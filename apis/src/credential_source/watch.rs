//! Single-thread projected-file watcher with fail-closed publication.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use arc_swap::ArcSwap;
use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};
use praxis_filter::FilterError;
use zeroize::Zeroizing;

use super::{
    CredentialSource, DiagnosticContext, SecretHandle, SecretSnapshot, TransformFn, resolve_raw_token,
};

/// Watcher-thread control message.
enum WatcherMessage {
    /// A filesystem event forwarded from the `notify` callback.
    Event(notify::Event),
    /// A runtime error reported by `notify`. The watcher can no longer trust
    /// its event stream, so it rescans every source (each reload fails closed
    /// on its own read) rather than continuing to serve a possibly-stale
    /// credential indefinitely.
    WatchError,
    /// Signal telling the watcher thread to stop and return.
    Shutdown,
}

/// One watched file source and its atomically-swappable snapshot.
struct WatchedSource {
    /// Configured path to the watched credential file.
    path: PathBuf,
    /// Canonicalized parent directory of `path`, used to match `notify` event
    /// paths back to this source. Filesystem backends report the real path
    /// (macOS `FSEvents` resolves symlinks to `/private/...`; Kubernetes atomic
    /// rotation swaps a `..data` symlink inside the directory), so comparing
    /// canonicalized directories — rather than the literal configured path —
    /// keeps rotation detection correct across symlinks.
    canonical_dir: PathBuf,
    /// Non-secret diagnostic context for byte-free error and reload logging.
    ctx: DiagnosticContext,
    /// Validation/shaping closure run on every (re)load.
    transform: TransformFn,
    /// Shared cell the watcher swaps atomically on reload.
    snapshot: Arc<ArcSwap<SecretSnapshot>>,
}

impl WatchedSource {
    /// Re-resolve this source and publish. On any failure, publish `None`
    /// (fail closed) and log a byte-free reason. Never logs the error message,
    /// so a careless adopter's `transform` error cannot leak token bytes.
    fn reload(&self) {
        let payload = match resolve_raw_token_and_transform(&self.path, &self.ctx, &self.transform) {
            Ok(payload) => Some(payload),
            Err(phase) => {
                tracing::warn!(
                    owner = self.ctx.owner,
                    reference = self.ctx.reference.as_deref().unwrap_or(""),
                    phase,
                    "credential source unavailable; failing closed"
                );
                None
            },
        };
        self.snapshot.store(Arc::new(SecretSnapshot { payload }));
    }
}

/// Resolve + transform, returning a static phase discriminator on error so the
/// caller can log a byte-free reason.
fn resolve_raw_token_and_transform(
    path: &Path,
    ctx: &DiagnosticContext,
    transform: &TransformFn,
) -> Result<Zeroizing<String>, &'static str> {
    let source = CredentialSource::File(path.to_string_lossy().into_owned());
    let Ok(raw) = resolve_raw_token(&source, ctx) else {
        return Err("read");
    };
    let Ok(payload) = transform(&raw) else {
        return Err("transform");
    };
    Ok(payload)
}

/// Owns the single watcher thread; `Drop` signals shutdown and joins.
pub struct WatchHandle {
    /// Sender used to deliver the shutdown signal to the watcher thread.
    shutdown: mpsc::Sender<WatcherMessage>,
    /// Join handle for the watcher thread, taken on `Drop` to join it.
    thread: Option<std::thread::JoinHandle<()>>,
}

impl WatchHandle {
    /// Number of live watcher threads (always 1 while alive) — test aid for the
    /// single-thread invariant.
    #[cfg(test)]
    #[must_use]
    pub fn thread_count(&self) -> usize {
        usize::from(self.thread.is_some())
    }

    /// Deliver a synthetic watcher-error message on the same channel the
    /// `notify` callback uses — test aid proving the loop rescans (fails closed)
    /// when the filesystem watcher reports a runtime error.
    #[cfg(test)]
    fn inject_watch_error_for_test(&self) {
        drop(self.shutdown.send(WatcherMessage::WatchError));
    }
}

impl Drop for WatchHandle {
    fn drop(&mut self) {
        drop(self.shutdown.send(WatcherMessage::Shutdown));
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }
}

impl std::fmt::Debug for WatchHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatchHandle").finish()
    }
}

/// Batch builder: registers many file sources under ONE watcher thread.
pub struct WatchRegistry {
    /// File sources registered so far, watched together by one thread.
    watched: Vec<WatchedSource>,
}

impl Default for WatchRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchRegistry {
    /// Create an empty registry that watches no file sources yet.
    #[must_use]
    pub fn new() -> Self {
        Self { watched: Vec::new() }
    }

    /// Register a `file` source: performs the authoritative initial read +
    /// transform (rejecting construction on failure) and returns its handle.
    ///
    /// The `transform` closure must return `Err` for every failure mode and must not
    /// panic: it runs on the watcher thread, where a panic terminates rotation and
    /// freezes the last-published snapshot (fail-stale) instead of failing closed.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when `source` is not a `File`, when the path is
    /// empty/oversized or unreadable, when the token is empty/oversized, or when
    /// the `transform` rejects the initial token.
    pub fn register(
        &mut self,
        ctx: DiagnosticContext,
        source: &CredentialSource,
        transform: impl Fn(&str) -> Result<Zeroizing<String>, FilterError> + Send + Sync + 'static,
    ) -> Result<SecretHandle, FilterError> {
        let CredentialSource::File(path) = source else {
            return Err(format!("{}: WatchRegistry only accepts a file source", ctx.owner).into());
        };
        super::file_validate_source_len_for_registry(path, ctx.owner)?;
        let transform: TransformFn = Arc::new(transform);
        // Initial read fails construction on a bad token (fail loud).
        let raw = resolve_raw_token(source, &ctx)?;
        let payload = transform(&raw)?;
        let handle = SecretHandle::new(SecretSnapshot { payload: Some(payload) });
        let path_buf = PathBuf::from(path);
        // The initial read above succeeded, so the parent directory exists and
        // canonicalizes; the `.` fallback is defensive for exotic paths.
        let canonical_dir = path_buf
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        self.watched.push(WatchedSource {
            path: path_buf,
            canonical_dir,
            ctx,
            transform,
            snapshot: handle.cell(),
        });
        Ok(handle)
    }

    /// Establish ONE watch per parent directory on ONE thread, then perform the
    /// authoritative re-read strictly after the watch is live. Returns `None`
    /// if no file source was registered.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the watcher cannot be created, a directory
    /// cannot be watched, the post-watch authoritative re-read fails, or the
    /// watcher does not report readiness within the timeout.
    #[expect(
        clippy::too_many_lines,
        reason = "watcher setup and lifecycle are intentionally bounded in one function"
    )]
    pub fn finish(self) -> Result<Option<WatchHandle>, FilterError> {
        if self.watched.is_empty() {
            return Ok(None);
        }
        let (ready_tx, ready_rx) = mpsc::channel();
        let (message_tx, message_rx) = mpsc::channel::<WatcherMessage>();
        let callback_tx = message_tx.clone();
        let watched = self.watched;
        let thread = std::thread::Builder::new()
            .name("credential-source-watcher".to_owned())
            .spawn(move || {
                let mut watcher: RecommendedWatcher = match notify::recommended_watcher(move |event| {
                    // Forward both success and failure: a discarded `notify`
                    // error would leave the watcher blind and pin the last
                    // published credential forever (fail-stale). The error value
                    // itself is never logged here — the rescan it triggers logs a
                    // byte-free per-source reason on any read failure.
                    match event {
                        Ok(event) => drop(callback_tx.send(WatcherMessage::Event(event))),
                        Err(_) => drop(callback_tx.send(WatcherMessage::WatchError)),
                    }
                }) {
                    Ok(watcher) => watcher,
                    Err(error) => {
                        drop(ready_tx.send(Err(format!("failed to create credential watcher: {error}"))));
                        return;
                    },
                };
                let mut directories = HashSet::new();
                for item in &watched {
                    let directory = item.canonical_dir.clone();
                    if directories.insert(directory.clone())
                        && watcher.watch(&directory, RecursiveMode::NonRecursive).is_err()
                    {
                        drop(ready_tx.send(Err(format!(
                            "failed to watch credential directory {}",
                            directory.display()
                        ))));
                        return;
                    }
                }
                // Authoritative re-read AFTER the watch is live (closes the
                // construct-read -> watch-register TOCTOU). Any event during the
                // read stays queued and is processed after readiness. Publish
                // only after every file re-reads OK, so readiness observes one
                // complete startup snapshot.
                let mut initial = Vec::with_capacity(watched.len());
                for item in &watched {
                    let Ok(payload) =
                        resolve_raw_token_and_transform(&item.path, &item.ctx, &item.transform)
                    else {
                        drop(ready_tx.send(Err(format!(
                            "{}: failed authoritative credential read",
                            item.ctx.owner
                        ))));
                        return;
                    };
                    initial.push((Arc::clone(&item.snapshot), payload));
                }
                for (snapshot, payload) in initial {
                    snapshot.store(Arc::new(SecretSnapshot { payload: Some(payload) }));
                }
                drop(ready_tx.send(Ok(())));
                while let Ok(message) = message_rx.recv() {
                    match message {
                        WatcherMessage::Event(event) => process_event(&event, &watched),
                        WatcherMessage::WatchError => {
                            // Byte-free: the notify error is not interpolated.
                            tracing::warn!("credential watcher reported an error; rescanning all sources");
                            reload_all(&watched);
                        },
                        WatcherMessage::Shutdown => break,
                    }
                }
            })
            .map_err(|error| format!("credential_source: failed to spawn watcher: {error}"))?;

        let ready = match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(ready) => ready,
            Err(error) => {
                drop(message_tx.send(WatcherMessage::Shutdown));
                drop(thread.join());
                return Err(
                    format!("credential_source: credential watcher did not initialize: {error}").into(),
                );
            },
        };
        if let Err(error) = ready {
            drop(message_tx.send(WatcherMessage::Shutdown));
            drop(thread.join());
            return Err(format!("credential_source: {error}").into());
        }
        Ok(Some(WatchHandle { shutdown: message_tx, thread: Some(thread) }))
    }
}

/// Reload every watched source unconditionally. Used when the watcher cannot
/// tell which files changed — a `notify` runtime error or a rescan hint — so it
/// re-resolves every source; each reload fails closed to `None` on its own read
/// error rather than serving a stale credential.
fn reload_all(watched: &[WatchedSource]) {
    for item in watched {
        item.reload();
    }
}

/// Reload the sources whose file/parent matches an event path; a rescan hint
/// reloads all.
fn process_event(event: &notify::Event, watched: &[WatchedSource]) {
    use notify::EventKind;
    if event.need_rescan() || matches!(event.kind, EventKind::Other) {
        reload_all(watched);
        return;
    }
    if !matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)) {
        return;
    }
    for item in watched {
        // Match cheaply on the literal path first (the common case where the
        // configured path has no symlink components), then fall back to
        // canonicalized parent-directory comparison so symlinked paths — macOS
        // FSEvents `/private/...`, Kubernetes `..data` swaps, or an atomic
        // sibling+rename — still resolve to this source. A removed file cannot
        // be canonicalized, so we canonicalize the (still-present) directory.
        let matched = event.paths.iter().any(|p| {
            p == &item.path
                || p.parent() == item.path.parent()
                || p.parent()
                    .and_then(|parent| parent.canonicalize().ok())
                    .is_some_and(|parent| parent == item.canonical_dir)
                || p.canonicalize().ok().is_some_and(|dir| dir == item.canonical_dir)
        });
        if matched {
            item.reload();
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;
    use crate::credential_source::{CredentialSource, DiagnosticContext};
    use std::time::{Duration, Instant};
    use zeroize::Zeroizing;

    fn identity() -> impl Fn(&str) -> Result<Zeroizing<String>, FilterError> + Send + Sync + 'static {
        |t: &str| Ok(Zeroizing::new(t.to_owned()))
    }
    fn ctx(reference: &str) -> DiagnosticContext {
        DiagnosticContext { owner: "credential_inject", reference: Some(reference.to_owned()) }
    }

    async fn wait_until<F: Fn() -> bool>(pred: F) -> bool {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if pred() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        false
    }

    #[test]
    fn file_source_reads_and_watches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "tok-a\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        let (handle, watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, identity()).unwrap();
        assert!(watch.is_some(), "file source must spawn a watcher");
        assert_eq!(handle.current().payload.as_deref().map(String::as_str), Some("tok-a"));
    }

    #[tokio::test]
    async fn rotation_swaps_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "tok-a\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        let (handle, _watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, identity()).unwrap();
        // Atomic replace (K8s projected-volume style): write sibling + rename.
        let tmp = dir.path().join("key.tmp");
        std::fs::write(&tmp, "tok-b\n").unwrap();
        std::fs::rename(&tmp, &path).unwrap();
        assert!(
            wait_until(|| handle.current().payload.as_deref().map(String::as_str) == Some("tok-b")).await
        );
    }

    #[tokio::test]
    async fn invalid_rotation_fails_closed_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "tok-a\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        let (handle, _watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, identity()).unwrap();
        std::fs::write(&path, "   \n").unwrap(); // now empty/invalid
        assert!(wait_until(|| handle.current().payload.is_none()).await);
    }

    #[tokio::test]
    async fn removal_fails_closed_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "tok-a\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        let (handle, _watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, identity()).unwrap();
        std::fs::remove_file(&path).unwrap(); // file disappears (rotation gap)
        assert!(wait_until(|| handle.current().payload.is_none()).await);
    }

    #[tokio::test]
    async fn transform_failure_on_rotation_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "ok\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        // transform rejects any token containing '!'
        let transform = |t: &str| -> Result<Zeroizing<String>, FilterError> {
            if t.contains('!') {
                return Err("bad token shape".to_owned().into());
            }
            Ok(Zeroizing::new(t.to_owned()))
        };
        let (handle, _watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, transform).unwrap();
        std::fs::write(&path, "bad!\n").unwrap();
        assert!(wait_until(|| handle.current().payload.is_none()).await);
    }

    #[test]
    fn missing_file_rejects_construction() {
        let src = CredentialSource::File("/no/such/dir/key".to_owned());
        assert!(crate::credential_source::build_one(ctx("a/b/c"), &src, identity()).is_err());
    }

    #[test]
    fn multiple_files_share_one_thread_with_dir_dedup() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("k1");
        let p2 = dir.path().join("k2");
        std::fs::write(&p1, "one\n").unwrap();
        std::fs::write(&p2, "two\n").unwrap();
        let mut reg = WatchRegistry::new();
        let h1 = reg
            .register(ctx("a/b/1"), &CredentialSource::File(p1.to_string_lossy().into_owned()), identity())
            .unwrap();
        let h2 = reg
            .register(ctx("a/b/2"), &CredentialSource::File(p2.to_string_lossy().into_owned()), identity())
            .unwrap();
        let watch = reg.finish().unwrap().expect("file sources produce a watch handle");
        assert_eq!(watch.thread_count(), 1, "both files share one watcher thread");
        assert_eq!(h1.current().payload.as_deref().map(String::as_str), Some("one"));
        assert_eq!(h2.current().payload.as_deref().map(String::as_str), Some("two"));
    }

    #[test]
    fn no_file_source_produces_no_watch_handle() {
        let reg = WatchRegistry::new();
        assert!(reg.finish().unwrap().is_none());
    }

    #[test]
    fn register_rejects_non_file_source() {
        let mut reg = WatchRegistry::new();
        let err = reg
            .register(ctx("a/b/c"), &CredentialSource::Value(Zeroizing::new("x".to_owned())), identity())
            .unwrap_err();
        assert!(err.to_string().contains("file"), "{err}");
    }

    #[test]
    fn watch_handle_drop_joins_quickly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "tok\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        let (_handle, watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, identity()).unwrap();
        let start = Instant::now();
        drop(watch);
        // `Drop` sends the shutdown signal and joins; the thread blocks on
        // `recv()` (not a polling `recv_timeout`), so it wakes at once rather
        // than after a poll interval. The bound is a load-tolerant proxy for
        // "woken immediately, not polling" — a true hang is unbounded and is
        // caught by the CI job timeout, not this assertion. macOS FSEvents
        // teardown plus thread join under parallel test load can take longer
        // than a tight 500ms, so allow 5s.
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "Drop must wake and join the watcher"
        );
    }

    #[test]
    fn authoritative_reread_replaces_pre_watch_snapshot() {
        // Write token-b, register (initial read = token-b), then overwrite with
        // token-c BEFORE finish() establishes the watch + authoritative re-read.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "token-b\n").unwrap();
        let mut reg = WatchRegistry::new();
        let handle = reg
            .register(ctx("a/b/c"), &CredentialSource::File(path.to_string_lossy().into_owned()), identity())
            .unwrap();
        std::fs::write(&path, "token-c\n").unwrap();
        let _watch = reg.finish().unwrap();
        // The authoritative re-read after the watch is live must observe token-c.
        assert_eq!(handle.current().payload.as_deref().map(String::as_str), Some("token-c"));
    }

    #[test]
    fn reload_all_republishes_valid_and_fails_missing_closed() {
        // Register two file sources, remove one, then rescan every source
        // directly (bypassing the OS watcher for determinism). The surviving
        // file stays valid; the removed one fails closed to `None`.
        let dir = tempfile::tempdir().unwrap();
        let present = dir.path().join("present");
        let missing = dir.path().join("missing");
        std::fs::write(&present, "live\n").unwrap();
        std::fs::write(&missing, "gone\n").unwrap();
        let mut reg = WatchRegistry::new();
        let mut register_file = |reference: &str, path: &Path| {
            reg.register(
                ctx(reference),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap()
        };
        let h_present = register_file("a/b/present", &present);
        let h_missing = register_file("a/b/missing", &missing);
        std::fs::remove_file(&missing).unwrap();
        reload_all(&reg.watched);
        assert_eq!(
            h_present.current().payload.as_deref().map(String::as_str),
            Some("live"),
            "a still-present source must re-resolve to its live value on rescan"
        );
        assert!(
            h_missing.current().payload.is_none(),
            "a removed source must fail closed to None on rescan"
        );
    }

    #[tokio::test]
    async fn watch_error_message_triggers_rescan() {
        // A synthetic watcher-error message (as the `notify` callback forwards on
        // a runtime error) must drive a full rescan that re-runs each source's
        // transform, proving errors are not discarded and left fail-stale.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "tok\n").unwrap();
        let src = CredentialSource::File(path.to_string_lossy().into_owned());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_in = Arc::clone(&calls);
        let transform = move |t: &str| -> Result<Zeroizing<String>, FilterError> {
            calls_in.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Zeroizing::new(t.to_owned()))
        };
        let (_handle, watch) = crate::credential_source::build_one(ctx("a/b/c"), &src, transform).unwrap();
        let watch = watch.expect("file source spawns a watcher");
        let before = calls.load(std::sync::atomic::Ordering::SeqCst);
        watch.inject_watch_error_for_test();
        assert!(
            wait_until(|| calls.load(std::sync::atomic::Ordering::SeqCst) > before).await,
            "a watcher-error message must trigger a rescan that re-runs the transform"
        );
    }
}
