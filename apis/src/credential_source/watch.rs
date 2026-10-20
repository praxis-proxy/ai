// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Single-thread projected-file watcher with fail-closed publication.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    time::Duration,
};

use arc_swap::ArcSwap;
use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};
use praxis_filter::FilterError;
use zeroize::Zeroizing;

use super::{CredentialSource, DiagnosticContext, SecretHandle, SecretSnapshot, TransformFn, resolve_raw_token};

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
    /// Stable lexical parent outside the configured directory's deepest
    /// symlink component, or the configured directory's parent when no symlink
    /// is present. Watching this anchor keeps directory deletion/recreation and
    /// symlink replacement observable even while the directory itself is absent.
    directory_anchor: Option<PathBuf>,
    /// Non-secret diagnostic context for byte-free error and reload logging.
    ctx: DiagnosticContext,
    /// Validation/shaping closure run on every (re)load.
    transform: TransformFn,
    /// Shared cell the watcher swaps atomically on reload.
    snapshot: Arc<ArcSwap<SecretSnapshot>>,
}

/// Filesystem identity of a watched directory. On Unix, device + inode detects
/// replacement at the same pathname; the portable fallback uses timestamps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectoryIdentity {
    #[cfg(unix)]
    /// Device containing the directory.
    device: u64,
    #[cfg(unix)]
    /// Inode identifying the directory on its device.
    inode: u64,
    #[cfg(not(unix))]
    /// Portable creation timestamp used when inode identity is unavailable.
    created: Option<std::time::SystemTime>,
    #[cfg(not(unix))]
    /// Portable modification timestamp used when inode identity is unavailable.
    modified: Option<std::time::SystemTime>,
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
        let configured_path = PathBuf::from(path);
        let path_buf = if configured_path.is_absolute() {
            configured_path
        } else {
            std::env::current_dir()
                .map_err(|error| format!("{}: cannot resolve credential file path: {error}", ctx.owner))?
                .join(configured_path)
        };
        // Keep the final component and configured directory lexical: projected
        // Secrets replace them atomically. The directory anchor remains stable
        // when that directory is removed or is a symlink retargeted elsewhere.
        let directory_anchor = stable_directory_anchor(&path_buf);
        self.watched.push(WatchedSource {
            path: path_buf,
            directory_anchor,
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
                let mut directories = HashMap::new();
                for directory in desired_watch_directories(&watched) {
                    if watcher.watch(&directory, RecursiveMode::NonRecursive).is_err() {
                        drop(ready_tx.send(Err(format!(
                            "failed to watch credential directory {}",
                            directory.display()
                        ))));
                        return;
                    }
                    let identity = match directory_identity(&directory) {
                        Ok(identity) => identity,
                        Err(error) => {
                            drop(ready_tx.send(Err(format!(
                                "failed to identify credential directory {}: {error}",
                                directory.display()
                            ))));
                            return;
                        },
                    };
                    directories.insert(directory, identity);
                }
                // Every stable anchor is now live. Recompute the topology once
                // more so a symlink retarget between discovery and watch setup
                // cannot leave only the old target subscribed while the
                // authoritative read publishes the new target's token.
                if let Err(error) = reconcile_watch_directories(&mut watcher, &watched, &mut directories) {
                    drop(ready_tx.send(Err(format!(
                        "failed to reconcile credential directories before readiness: {error}"
                    ))));
                    return;
                }
                // Authoritative re-read AFTER the watch is live (closes the
                // construct-read -> watch-register TOCTOU). Any event during the
                // read stays queued and is processed after readiness. Publish
                // only after every file re-reads OK, so readiness observes one
                // complete startup snapshot.
                let mut initial = Vec::with_capacity(watched.len());
                for item in &watched {
                    let Ok(payload) = resolve_raw_token_and_transform(&item.path, &item.ctx, &item.transform) else {
                        drop(ready_tx.send(Err(format!("{}: failed authoritative credential read", item.ctx.owner))));
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
                        WatcherMessage::Event(event) => {
                            if event_requires_reload(&event) {
                                if reconcile_watch_directories(&mut watcher, &watched, &mut directories).is_err() {
                                    tracing::warn!(
                                        "credential watcher could not follow a directory replacement; failing closed"
                                    );
                                    fail_closed_all(&watched);
                                } else {
                                    reload_all(&watched);
                                }
                            }
                        },
                        WatcherMessage::WatchError => {
                            // Byte-free: the notify error is not interpolated.
                            tracing::warn!("credential watcher reported an error; rescanning all sources");
                            if reconcile_watch_directories(&mut watcher, &watched, &mut directories).is_err() {
                                fail_closed_all(&watched);
                            } else {
                                reload_all(&watched);
                            }
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
                return Err(format!("credential_source: credential watcher did not initialize: {error}").into());
            },
        };
        if let Err(error) = ready {
            drop(message_tx.send(WatcherMessage::Shutdown));
            drop(thread.join());
            return Err(format!("credential_source: {error}").into());
        }
        Ok(Some(WatchHandle {
            shutdown: message_tx,
            thread: Some(thread),
        }))
    }
}

/// Choose a lexical directory that remains stable when the configured parent
/// is removed or any symlink component beneath the anchor is retargeted.
fn stable_directory_anchor(path: &Path) -> Option<PathBuf> {
    let configured_dir = path.parent()?;
    configured_dir
        .ancestors()
        .find(|candidate| {
            std::fs::symlink_metadata(candidate)
                .ok()
                .is_some_and(|metadata| metadata.file_type().is_symlink())
        })
        .and_then(Path::parent)
        .or_else(|| configured_dir.parent())
        .map(Path::to_path_buf)
}

/// Read a directory identity that changes when the same pathname is replaced.
fn directory_identity(path: &Path) -> Result<DirectoryIdentity, notify::Error> {
    let metadata = std::fs::metadata(path).map_err(|error| notify::Error::io(error).add_path(path.to_path_buf()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        Ok(DirectoryIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(DirectoryIdentity {
            created: metadata.created().ok(),
            modified: metadata.modified().ok(),
        })
    }
}

/// Resolve every symlink hop in a credential path and return the directories
/// needed to observe link replacement, target-file changes, and target-directory
/// recreation.
///
/// A projected credential may traverse file symlinks, directory symlinks, or
/// both. Each hop contributes the link's containing directory, the target
/// directory, and stable anchors outside them. The anchors remain watchable if
/// a target directory is removed, so recreation can re-arm the direct
/// subscription. A visited set and hop bound terminate malformed cycles without
/// affecting the startup read, which has already rejected an unreadable source.
fn symlink_chain_watch_directories(path: &Path) -> HashSet<PathBuf> {
    let mut directories = HashSet::new();
    let mut visited = HashSet::new();
    let mut current = path.to_path_buf();

    // Match the conventional filesystem symlink-resolution ceiling. The
    // startup read already rejects a loop; this bound also protects runtime
    // reconciliation if a previously valid chain is retargeted into one.
    for _ in 0..40 {
        if !visited.insert(current.clone()) {
            break;
        }
        let Some(link) = first_symlink_component(&current) else {
            break;
        };
        let Ok(target) = std::fs::read_link(&link) else {
            break;
        };
        let suffix = current.strip_prefix(&link).unwrap_or_else(|_| Path::new(""));
        let resolved_target = if target.is_absolute() {
            target
        } else if let Some(parent) = link.parent() {
            parent.join(target)
        } else {
            break;
        };
        insert_target_directory_watches(&mut directories, &link);
        current = resolved_target.join(suffix);
        insert_target_directory_watches(&mut directories, &current);
    }

    directories
}

/// Return the first symlink component in `path`, from the filesystem root
/// toward the final credential file.
fn first_symlink_component(path: &Path) -> Option<PathBuf> {
    let mut candidate = PathBuf::new();
    for component in path.components() {
        candidate.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&candidate).ok()?;
        if metadata.file_type().is_symlink() {
            return Some(candidate);
        }
    }
    None
}

/// Add direct and stable watches for one file-symlink target.
fn insert_target_directory_watches(directories: &mut HashSet<PathBuf>, target: &Path) {
    if let Some(parent) = target.parent().and_then(|parent| parent.canonicalize().ok()) {
        directories.insert(parent);
    }
    // Keep the immediate parent of the target directory subscribed even when
    // an unrelated higher ancestor is itself a symlink (for example, macOS
    // `/var`). This directory observes removal and recreation of the target
    // directory name.
    if let Some(parent_anchor) = target
        .parent()
        .and_then(Path::parent)
        .and_then(|parent| parent.canonicalize().ok())
    {
        directories.insert(parent_anchor);
    }
    if let Some(anchor) = stable_directory_anchor(target).and_then(|anchor| anchor.canonicalize().ok()) {
        directories.insert(anchor);
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

/// Publish an unavailable snapshot for every watched source without reading it.
fn fail_closed_all(watched: &[WatchedSource]) {
    for item in watched {
        item.snapshot.store(Arc::new(SecretSnapshot { payload: None }));
    }
}

/// Return the directories that must be watched for the current filesystem
/// topology. Each source watches its resolved configured directory, every
/// file-symlink target directory, and stable parents for all of those
/// directories. Stable parents observe directory recreation and symlink
/// replacement; recomputing the direct watches follows a new target after
/// rotation.
fn desired_watch_directories(watched: &[WatchedSource]) -> HashSet<PathBuf> {
    let mut directories = HashSet::new();
    for item in watched {
        if let Some(resolved) = item.path.parent().and_then(|parent| parent.canonicalize().ok()) {
            directories.insert(resolved);
        }
        if let Some(anchor) = item
            .directory_anchor
            .as_deref()
            .and_then(|parent| parent.canonicalize().ok())
        {
            directories.insert(anchor);
        }
        directories.extend(symlink_chain_watch_directories(&item.path));
    }
    directories
}

/// Reconcile `notify` subscriptions after a directory or symlink replacement.
/// New subscriptions are established before obsolete ones are removed so no
/// rotation window is introduced while following a new target.
fn reconcile_watch_directories(
    watcher: &mut RecommendedWatcher,
    watched: &[WatchedSource],
    active: &mut HashMap<PathBuf, DirectoryIdentity>,
) -> Result<(), notify::Error> {
    let desired = desired_watch_directories(watched);
    for directory in &desired {
        let identity = directory_identity(directory)?;
        if active.get(directory) != Some(&identity) {
            // An identical pathname can now refer to a replacement inode. Drop
            // any obsolete subscription before arming the current directory.
            if active.remove(directory).is_some() {
                drop(watcher.unwatch(directory));
            }
            watcher.watch(directory, RecursiveMode::NonRecursive)?;
            active.insert(directory.clone(), identity);
        }
    }
    let obsolete: Vec<_> = active
        .keys()
        .filter(|directory| !desired.contains(*directory))
        .cloned()
        .collect();
    for directory in obsolete {
        // A backend may already have discarded a watch after directory
        // removal. An unwatch failure only leaves a harmless obsolete watch;
        // failure to establish a new watch above is the fail-closed condition.
        drop(watcher.unwatch(&directory));
        active.remove(&directory);
    }
    Ok(())
}

/// Whether a filesystem event can affect a watched source. Every subscribed
/// directory is credential-related, so path matching is deliberately avoided:
/// re-reading a few sources is cheap and remains correct when a removed path can
/// no longer be canonicalized.
fn event_requires_reload(event: &notify::Event) -> bool {
    use notify::EventKind;
    event.need_rescan()
        || matches!(
            event.kind,
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Other
        )
}

/// Apply an event directly without the live watcher. Used by deterministic
/// tests for removal and rescan semantics.
#[cfg(test)]
fn process_event(event: &notify::Event, watched: &[WatchedSource]) {
    if event_requires_reload(event) {
        reload_all(watched);
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use std::time::{Duration, Instant};

    use zeroize::Zeroizing;

    use super::*;
    use crate::credential_source::{CredentialSource, DiagnosticContext};

    fn identity() -> impl Fn(&str) -> Result<Zeroizing<String>, FilterError> + Send + Sync + 'static {
        |t: &str| Ok(Zeroizing::new(t.to_owned()))
    }
    fn ctx(reference: &str) -> DiagnosticContext {
        DiagnosticContext {
            owner: "credential_inject",
            reference: Some(reference.to_owned()),
        }
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
        assert!(wait_until(|| handle.current().payload.as_deref().map(String::as_str) == Some("tok-b")).await);
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
            .register(
                ctx("a/b/1"),
                &CredentialSource::File(p1.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        let h2 = reg
            .register(
                ctx("a/b/2"),
                &CredentialSource::File(p2.to_string_lossy().into_owned()),
                identity(),
            )
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
            .register(
                ctx("a/b/c"),
                &CredentialSource::Value(Zeroizing::new("x".to_owned())),
                identity(),
            )
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
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
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

    #[test]
    fn bare_relative_path_matches_absolute_rotation_and_removal_events() {
        let cwd = std::env::current_dir().unwrap();
        let dir = tempfile::tempdir_in(&cwd).unwrap();
        let absolute_path = dir.path().join("key");
        let relative_path = absolute_path.strip_prefix(&cwd).unwrap().to_path_buf();
        std::fs::write(&absolute_path, "token-a\n").unwrap();

        let mut registry = WatchRegistry::new();
        let handle = registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(relative_path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        assert!(registry.watched.first().unwrap().path.is_absolute());

        std::fs::write(&absolute_path, "token-b\n").unwrap();
        let modified = notify::Event::new(notify::EventKind::Modify(notify::event::ModifyKind::Data(
            notify::event::DataChange::Any,
        )))
        .add_path(absolute_path.clone());
        process_event(&modified, &registry.watched);
        assert_eq!(handle.current().payload.as_deref().map(String::as_str), Some("token-b"));

        std::fs::remove_file(&absolute_path).unwrap();
        let removed =
            notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::File)).add_path(absolute_path);
        process_event(&removed, &registry.watched);
        assert!(handle.current().payload.is_none());
    }

    #[test]
    fn removed_parent_directory_event_fails_closed() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("secrets");
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("key");
        std::fs::write(&path, "token-a\n").unwrap();

        let mut registry = WatchRegistry::new();
        let handle = registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();

        std::fs::rename(&directory, root.path().join("old-secrets")).unwrap();
        let removed =
            notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::Folder)).add_path(directory);
        process_event(&removed, &registry.watched);
        assert!(
            handle.current().payload.is_none(),
            "removing the watched directory must not retain the previous credential"
        );
    }

    #[test]
    fn same_path_directory_replacement_changes_identity() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("secrets");
        std::fs::create_dir(&directory).unwrap();
        let before = directory_identity(&directory).unwrap();

        std::fs::rename(&directory, root.path().join("old-secrets")).unwrap();
        std::fs::create_dir(&directory).unwrap();
        let after = directory_identity(&directory).unwrap();

        assert_ne!(
            before, after,
            "same-path directory replacement must force the watcher subscription to re-arm"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_file_target_directory_is_watched_and_remains_on_removal() {
        let root = tempfile::tempdir().unwrap();
        let config_dir = root.path().join("config");
        let target_dir = root.path().join("projected-secret");
        std::fs::create_dir(&config_dir).unwrap();
        std::fs::create_dir(&target_dir).unwrap();
        let target = target_dir.join("key");
        std::fs::write(&target, "token-a\n").unwrap();
        let path = config_dir.join("token");
        std::os::unix::fs::symlink(&target, &path).unwrap();

        let mut registry = WatchRegistry::new();
        let handle = registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        assert!(desired_watch_directories(&registry.watched).contains(&target_dir.canonicalize().unwrap()));

        std::fs::write(&target, "token-b\n").unwrap();
        let modified = notify::Event::new(notify::EventKind::Modify(notify::event::ModifyKind::Data(
            notify::event::DataChange::Any,
        )))
        .add_path(target.clone());
        process_event(&modified, &registry.watched);
        assert_eq!(handle.current().payload.as_deref().map(String::as_str), Some("token-b"));

        std::fs::remove_file(&target).unwrap();
        assert!(desired_watch_directories(&registry.watched).contains(&target_dir.canonicalize().unwrap()));
        let removed = notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::File)).add_path(target);
        process_event(&removed, &registry.watched);
        assert!(handle.current().payload.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn multi_hop_file_symlink_watches_every_target_directory() {
        let root = tempfile::tempdir().unwrap();
        let config_dir = root.path().join("config");
        let links_dir = root.path().join("links");
        let target_dir = root.path().join("secrets/live");
        std::fs::create_dir(&config_dir).unwrap();
        std::fs::create_dir(&links_dir).unwrap();
        std::fs::create_dir_all(&target_dir).unwrap();
        let target = target_dir.join("key");
        std::fs::write(&target, "token-a\n").unwrap();
        let intermediate = links_dir.join("key");
        std::os::unix::fs::symlink(&target, &intermediate).unwrap();
        let path = config_dir.join("token");
        std::os::unix::fs::symlink(&intermediate, &path).unwrap();

        let mut registry = WatchRegistry::new();
        registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        let directories = desired_watch_directories(&registry.watched);

        assert!(
            directories.contains(&links_dir.canonicalize().unwrap()),
            "the intermediate file-symlink directory must remain observable"
        );
        assert!(
            directories.contains(&target_dir.canonicalize().unwrap()),
            "the ultimate token directory must be watched"
        );
    }

    #[cfg(unix)]
    #[test]
    #[expect(clippy::too_many_lines, reason = "complete chained-directory symlink setup")]
    fn multi_hop_directory_symlink_watches_intermediate_link_directory() {
        let root = tempfile::tempdir().unwrap();
        let config_dir = root.path().join("config");
        let secrets_dir = root.path().join("secrets");
        let version_a = secrets_dir.join("version-a");
        let version_b = secrets_dir.join("version-b");
        std::fs::create_dir(&config_dir).unwrap();
        std::fs::create_dir_all(&version_a).unwrap();
        std::fs::create_dir_all(&version_b).unwrap();
        std::fs::write(version_a.join("key"), "token-a\n").unwrap();
        std::fs::write(version_b.join("key"), "token-b\n").unwrap();
        let current = secrets_dir.join("current");
        std::os::unix::fs::symlink(&version_a, &current).unwrap();
        let link = config_dir.join("link");
        std::os::unix::fs::symlink(&current, &link).unwrap();
        let path = link.join("key");

        let mut registry = WatchRegistry::new();
        let handle = registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        let before = desired_watch_directories(&registry.watched);
        assert!(
            before.contains(&secrets_dir.canonicalize().unwrap()),
            "the directory containing an intermediate symlink must be watched"
        );

        std::fs::remove_file(&current).unwrap();
        std::os::unix::fs::symlink(&version_b, &current).unwrap();
        let after = desired_watch_directories(&registry.watched);
        assert!(after.contains(&version_b.canonicalize().unwrap()));
        assert!(!after.contains(&version_a.canonicalize().unwrap()));

        let replaced = notify::Event::new(notify::EventKind::Modify(notify::event::ModifyKind::Name(
            notify::event::RenameMode::Both,
        )))
        .add_path(current);
        process_event(&replaced, &registry.watched);
        assert_eq!(handle.current().payload.as_deref().map(String::as_str), Some("token-b"));
    }

    #[cfg(unix)]
    #[test]
    #[expect(clippy::too_many_lines, reason = "complete target-directory recreation setup")]
    fn symlink_target_directory_keeps_recreation_anchor() {
        let root = tempfile::tempdir().unwrap();
        let config_dir = root.path().join("config");
        let secrets_dir = root.path().join("secrets");
        let target_dir = secrets_dir.join("live");
        std::fs::create_dir(&config_dir).unwrap();
        std::fs::create_dir_all(&target_dir).unwrap();
        let target = target_dir.join("key");
        std::fs::write(&target, "token-a\n").unwrap();
        let path = config_dir.join("token");
        std::os::unix::fs::symlink(&target, &path).unwrap();

        let mut registry = WatchRegistry::new();
        registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        let anchor = secrets_dir.canonicalize().unwrap();
        let before = desired_watch_directories(&registry.watched);
        assert!(before.contains(&target_dir.canonicalize().unwrap()));
        assert!(
            before.contains(&anchor),
            "the target directory's parent must observe its removal and recreation"
        );

        std::fs::remove_dir_all(&target_dir).unwrap();
        let missing = desired_watch_directories(&registry.watched);
        assert!(
            missing.contains(&anchor),
            "the recreation anchor must remain watched while the target directory is absent"
        );
    }

    #[cfg(unix)]
    #[test]
    #[expect(clippy::too_many_lines, reason = "complete symlink retarget regression setup")]
    fn symlink_parent_retarget_changes_watch_topology_and_reloads() {
        let root = tempfile::tempdir().unwrap();
        let version_a = root.path().join("version-a");
        let version_b = root.path().join("version-b");
        std::fs::create_dir_all(version_a.join("nested")).unwrap();
        std::fs::create_dir_all(version_b.join("nested")).unwrap();
        std::fs::write(version_a.join("nested/key"), "token-a\n").unwrap();
        std::fs::write(version_b.join("nested/key"), "token-b\n").unwrap();
        let current = root.path().join("current");
        std::os::unix::fs::symlink(&version_a, &current).unwrap();
        let path = current.join("nested/key");

        let mut registry = WatchRegistry::new();
        let handle = registry
            .register(
                ctx("a/b/c"),
                &CredentialSource::File(path.to_string_lossy().into_owned()),
                identity(),
            )
            .unwrap();
        let before = desired_watch_directories(&registry.watched);
        assert!(before.contains(&version_a.join("nested").canonicalize().unwrap()));
        assert!(before.contains(&root.path().canonicalize().unwrap()));

        std::fs::remove_file(&current).unwrap();
        std::os::unix::fs::symlink(&version_b, &current).unwrap();
        let after = desired_watch_directories(&registry.watched);
        assert!(after.contains(&version_b.join("nested").canonicalize().unwrap()));
        assert!(!after.contains(&version_a.join("nested").canonicalize().unwrap()));

        let replaced = notify::Event::new(notify::EventKind::Modify(notify::event::ModifyKind::Name(
            notify::event::RenameMode::Both,
        )))
        .add_path(current);
        process_event(&replaced, &registry.watched);
        assert_eq!(handle.current().payload.as_deref().map(String::as_str), Some("token-b"));
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
