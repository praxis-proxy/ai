// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Reads a Kubernetes service account token and keeps the current value in memory.
//!
//! The token file is read at startup. A background watcher re-reads it when
//! Kubernetes rotates the file, including the usual `..data` shortcut swap.
//! A watcher error re-reads the file and when that error means the watch
//! itself is gone, it watches the folder again. Callers attach the current token
//! to the request they already send. If the file cannot be read, the saved
//! token is cleared and callers refuse the call instead of sending an old or
//! missing token.

use std::{
    io::Read as _,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread::JoinHandle,
    time::Duration,
};

use arc_swap::ArcSwap;
use notify::{ErrorKind, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher as _};
use praxis_filter::FilterError;
use zeroize::Zeroizing;

/// Maximum accepted token file size (16 KiB).
const MAX_TOKEN_BYTES: usize = 16 * 1024;

/// Read cap: one byte past the maximum so an oversized file is detected.
const MAX_TOKEN_READ_BYTES: usize = MAX_TOKEN_BYTES + 1;

/// In-memory copy of the token file.
struct TokenSnapshot {
    /// Current token. `None` means the token must not be sent.
    token: Option<Zeroizing<String>>,
}

/// Stops the watcher thread when dropped.
struct TokenReloadHandle {
    /// Channel used to stop the watcher.
    shutdown: Sender<WatcherMessage>,
    /// Watcher thread, joined on drop.
    thread: Option<JoinHandle<()>>,
}

impl Drop for TokenReloadHandle {
    fn drop(&mut self) {
        drop(self.shutdown.send(WatcherMessage::Shutdown));
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }
}

/// Message from the watcher thread.
enum WatcherMessage {
    /// A filesystem event.
    Event(Event),
    /// The watcher reported an error.
    NotifyError {
        /// The folder subscription is gone and must be established again.
        watch_lost: bool,
    },
    /// Stop the thread.
    Shutdown,
}

/// A Kubernetes service account token, refreshed when the file changes.
pub(crate) struct ServiceAccountToken {
    /// Latest token. Request handling only reads this.
    snapshot: Arc<ArcSwap<TokenSnapshot>>,
    /// Kept so the watcher stops when this value is dropped.
    _watcher: Mutex<TokenReloadHandle>,
}

impl std::fmt::Debug for ServiceAccountToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ServiceAccountToken")
    }
}

impl ServiceAccountToken {
    /// Read `path` and start watching its folder.
    ///
    /// Fails when the path is empty, the file is missing, empty, or larger
    /// than 16 KiB, or the watcher cannot start. The path is turned into a
    /// full path once so the first read and later change events use the same
    /// location.
    pub(crate) fn open(path: &Path) -> Result<Self, FilterError> {
        let display = path.display().to_string();
        if display.trim().is_empty() {
            return Err("service account token: token_file must not be empty".into());
        }
        let path = std::path::absolute(path).map_err(|error| -> FilterError {
            format!("service account token: failed to resolve token_file '{display}': {error}").into()
        })?;
        let token = read_token(&path)?;
        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot { token: Some(token) }));
        let watcher = spawn_token_watcher(path, Arc::clone(&snapshot))?;
        Ok(Self {
            snapshot,
            _watcher: Mutex::new(watcher),
        })
    }

    /// The current token, or `None` when it must not be sent.
    pub(crate) fn current(&self) -> Option<Zeroizing<String>> {
        self.snapshot.load().token.clone()
    }
}

/// Read the token file, rejecting empty and oversized contents.
fn read_token(path: &Path) -> Result<Zeroizing<String>, FilterError> {
    let file = std::fs::File::open(path).map_err(|error| -> FilterError {
        format!(
            "service account token: cannot read token file {}: {error}",
            path.display()
        )
        .into()
    })?;
    let mut content = Zeroizing::new(String::new());
    file.take(MAX_TOKEN_READ_BYTES as u64)
        .read_to_string(&mut content)
        .map_err(|error| -> FilterError {
            format!(
                "service account token: cannot read token file {}: {error}",
                path.display()
            )
            .into()
        })?;
    if content.len() > MAX_TOKEN_BYTES {
        return Err(format!(
            "service account token: token file {} exceeds {} bytes",
            path.display(),
            MAX_TOKEN_BYTES
        )
        .into());
    }
    let token = Zeroizing::new(content.trim().to_owned());
    if token.is_empty() {
        return Err(format!("service account token: token file {} is empty", path.display()).into());
    }
    Ok(token)
}

/// Re-read the token file. On any failure, clear the saved token.
fn reload_token(path: &Path, snapshot: &ArcSwap<TokenSnapshot>) {
    let token = match read_token(path) {
        Ok(token) => Some(token),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "service account token unavailable; failing closed");
            None
        },
    };
    snapshot.store(Arc::new(TokenSnapshot { token }));
}

/// Handle one change reported by the file watcher.
///
/// The mounted `token` file is a shortcut to `..data/token`. Kubernetes
/// rotates the token by replacing `..data` and leaves `token` itself alone,
/// so the change shows up as another file in the same folder.
fn process_watcher_event(event: &Event, path: &Path, snapshot: &ArcSwap<TokenSnapshot>) {
    if event.need_rescan() || matches!(event.kind, EventKind::Other) {
        reload_token(path, snapshot);
        return;
    }
    let relevant = matches!(
        event.kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    );
    if relevant
        && event
            .paths
            .iter()
            .any(|event_path| event_affects_token(event_path, path))
    {
        reload_token(path, snapshot);
    }
}

/// Whether the changed path is the token file or another file in its folder.
fn event_affects_token(event_path: &Path, token_path: &Path) -> bool {
    if event_path == token_path {
        return true;
    }
    let (Some(token_dir), Some(event_dir)) = (token_path.parent(), event_path.parent()) else {
        return false;
    };
    if event_dir == token_dir {
        return true;
    }
    let Some(token_real) = token_dir.canonicalize().ok() else {
        return false;
    };
    event_dir
        .canonicalize()
        .is_ok_and(|event_real| event_real == token_real)
}

/// Whether `error` means the folder subscription is gone.
///
/// A missing path or a missing watch cannot deliver the next rotation.
/// Other notify errors leave the existing subscription in place.
fn watch_is_lost(error: &notify::Error) -> bool {
    match &error.kind {
        ErrorKind::PathNotFound | ErrorKind::WatchNotFound => true,
        ErrorKind::Io(io_error) => io_error.kind() == std::io::ErrorKind::NotFound,
        ErrorKind::Generic(_) | ErrorKind::InvalidConfig(_) | ErrorKind::MaxFilesWatch => false,
    }
}

/// Watch `directory` again after the previous subscription was lost.
///
/// A failed watch leaves the token from the latest successful read in place.
/// Later rotations are missed until a watch is established.
fn restore_token_watch(watcher: &mut RecommendedWatcher, directory: &Path) {
    drop(watcher.unwatch(directory));
    if let Err(error) = watcher.watch(directory, RecursiveMode::NonRecursive) {
        tracing::warn!(
            path = %directory.display(),
            %error,
            "service account token watch could not be restored"
        );
    }
}

/// Apply one watcher message. Returns `false` when the watcher should stop.
///
/// A notify error re-reads the token file. The saved token is cleared only
/// when that read fails. When the error means the watch itself is gone, the
/// folder is watched again so the next rotation is observed.
fn handle_watcher_message(
    message: WatcherMessage,
    path: &Path,
    directory: &Path,
    snapshot: &ArcSwap<TokenSnapshot>,
    watcher: &mut RecommendedWatcher,
) -> bool {
    match message {
        WatcherMessage::Event(event) => {
            process_watcher_event(&event, path, snapshot);
            true
        },
        WatcherMessage::NotifyError { watch_lost } => {
            reload_token(path, snapshot);
            if watch_lost {
                restore_token_watch(watcher, directory);
            }
            true
        },
        WatcherMessage::Shutdown => false,
    }
}

/// Watch the token file's folder and refresh the in-memory copy.
fn spawn_token_watcher(path: PathBuf, snapshot: Arc<ArcSwap<TokenSnapshot>>) -> Result<TokenReloadHandle, FilterError> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (message_tx, message_rx) = mpsc::channel();
    let callback_tx = message_tx.clone();

    let thread = std::thread::Builder::new()
        .name("service-account-token-watcher".to_owned())
        .spawn(move || watcher_thread(path, snapshot, ready_tx, callback_tx, message_rx))
        .map_err(|error| -> FilterError {
            format!("service account token: failed to spawn watcher: {error}").into()
        })?;

    let ready = match ready_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(ready) => ready,
        Err(error) => {
            drop(message_tx.send(WatcherMessage::Shutdown));
            drop(thread.join());
            return Err(format!("service account token: watcher did not initialize: {error}").into());
        },
    };
    if let Err(error) = ready {
        drop(message_tx.send(WatcherMessage::Shutdown));
        drop(thread.join());
        return Err(format!("service account token: {error}").into());
    }

    Ok(TokenReloadHandle {
        shutdown: message_tx,
        thread: Some(thread),
    })
}

/// Watcher thread body: watch the folder, re-read the file, then apply events.
#[expect(
    clippy::too_many_lines,
    reason = "watcher setup and the event loop stay in one place"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "the watcher thread owns these values until it exits"
)]
fn watcher_thread(
    path: PathBuf,
    snapshot: Arc<ArcSwap<TokenSnapshot>>,
    ready_tx: Sender<Result<(), String>>,
    callback_tx: Sender<WatcherMessage>,
    message_rx: Receiver<WatcherMessage>,
) {
    let mut watcher: RecommendedWatcher = match notify::recommended_watcher(move |event| match event {
        Ok(event) => drop(callback_tx.send(WatcherMessage::Event(event))),
        Err(error) => {
            let watch_lost = watch_is_lost(&error);
            tracing::warn!(
                %error,
                watch_lost,
                "service account token watcher error; re-reading the token"
            );
            drop(callback_tx.send(WatcherMessage::NotifyError { watch_lost }));
        },
    }) {
        Ok(watcher) => watcher,
        Err(error) => {
            drop(ready_tx.send(Err(format!("failed to create token watcher: {error}"))));
            return;
        },
    };

    let directory = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    if let Err(error) = watcher.watch(&directory, RecursiveMode::NonRecursive) {
        drop(ready_tx.send(Err(format!(
            "failed to watch token directory {}: {error}",
            directory.display()
        ))));
        return;
    }

    // Start watching before re-reading so a rotation between the two is not missed.
    match read_token(&path) {
        Ok(token) => snapshot.store(Arc::new(TokenSnapshot { token: Some(token) })),
        Err(error) => {
            drop(ready_tx.send(Err(format!("failed authoritative token read: {error}"))));
            return;
        },
    }
    drop(ready_tx.send(Ok(())));

    while let Ok(message) = message_rx.recv() {
        if !handle_watcher_message(message, &path, &directory, &snapshot, &mut watcher) {
            break;
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "tests wait on the filesystem watcher"
)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::mpsc::{self, RecvTimeoutError},
        time::{Duration, Instant},
    };

    use super::*;

    fn write_token(dir: &tempfile::TempDir, contents: &str) -> PathBuf {
        let path = dir.path().join("token");
        fs::write(&path, contents).unwrap();
        path
    }

    fn wait_for(token: &ServiceAccountToken, pred: impl Fn(Option<&str>) -> bool) -> Option<Zeroizing<String>> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let current = token.current();
            if pred(current.as_ref().map(|value| value.as_str())) {
                return current;
            }
            assert!(Instant::now() < deadline, "timed out waiting for token snapshot");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn missing_file_rejected_at_construction() {
        let err = ServiceAccountToken::open(Path::new("/nonexistent/path/to/token")).unwrap_err();
        assert!(err.to_string().contains("cannot read"), "{err}");
    }

    #[test]
    fn empty_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "   \n");
        let err = ServiceAccountToken::open(&path).unwrap_err();
        assert!(err.to_string().contains("is empty"), "{err}");
    }

    #[test]
    fn oversized_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        fs::write(&path, vec![b't'; MAX_TOKEN_BYTES + 1]).unwrap();
        let err = ServiceAccountToken::open(&path).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn blank_path_rejected() {
        let err = ServiceAccountToken::open(Path::new("   ")).unwrap_err();
        assert!(err.to_string().contains("token_file"), "{err}");
    }

    #[test]
    fn startup_reread_keeps_the_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "first\n");
        fs::write(&path, "second\n").unwrap();
        let token = ServiceAccountToken::open(&path).unwrap();
        assert_eq!(token.current().as_ref().map(|value| value.as_str()), Some("second"));
    }

    #[tokio::test]
    async fn rotation_replaces_the_saved_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "first-token\n");
        let token = ServiceAccountToken::open(&path).unwrap();
        fs::write(&path, "second-token\n").unwrap();
        let current = wait_for(&token, |value| value == Some("second-token"));
        assert_eq!(current.as_ref().map(|value| value.as_str()), Some("second-token"));
    }

    #[tokio::test]
    async fn deleting_the_file_clears_the_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "live-token\n");
        let token = ServiceAccountToken::open(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let current = wait_for(&token, |value| value.is_none());
        assert!(current.is_none());
    }

    #[test]
    fn notify_error_rereads_a_valid_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "still-good\n");
        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot {
            token: Some(Zeroizing::new("stale".to_owned())),
        }));
        let mut watcher = notify::recommended_watcher(|_| {}).unwrap();
        assert!(handle_watcher_message(
            WatcherMessage::NotifyError { watch_lost: false },
            &path,
            dir.path(),
            &snapshot,
            &mut watcher,
        ));
        assert_eq!(
            snapshot.load().token.as_ref().map(|value| value.as_str()),
            Some("still-good")
        );
    }

    #[test]
    fn notify_error_clears_a_missing_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "gone\n");
        fs::remove_file(&path).unwrap();
        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot {
            token: Some(Zeroizing::new("gone".to_owned())),
        }));
        let mut watcher = notify::recommended_watcher(|_| {}).unwrap();
        assert!(handle_watcher_message(
            WatcherMessage::NotifyError { watch_lost: false },
            &path,
            dir.path(),
            &snapshot,
            &mut watcher,
        ));
        assert!(snapshot.load().token.is_none());
    }

    #[test]
    fn path_and_missing_watch_errors_mean_the_watch_is_gone() {
        assert!(watch_is_lost(&notify::Error::path_not_found()));
        assert!(watch_is_lost(&notify::Error::watch_not_found()));
        assert!(watch_is_lost(&notify::Error::io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing"
        ))));
        assert!(!watch_is_lost(&notify::Error::generic("queue overflow")));
        assert!(!watch_is_lost(&notify::Error::new(ErrorKind::MaxFilesWatch)));
    }

    fn drain_for(rx: &Receiver<Result<Event, notify::Error>>, window: Duration) {
        let quiet = Instant::now() + window;
        while Instant::now() < quiet {
            drop(rx.recv_timeout(Duration::from_millis(20)));
        }
    }

    fn watch_event_arrives(rx: &Receiver<Result<Event, notify::Error>>) -> bool {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(Ok(_)) => return true,
                Ok(Err(_)) | Err(RecvTimeoutError::Timeout) => {},
                Err(RecvTimeoutError::Disconnected) => return false,
            }
        }
        false
    }

    #[test]
    fn lost_watch_observes_a_later_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "first\n");
        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot {
            token: Some(Zeroizing::new("first".to_owned())),
        }));
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |event| {
            drop(tx.send(event));
        })
        .unwrap();
        assert!(handle_watcher_message(
            WatcherMessage::NotifyError { watch_lost: true },
            &path,
            dir.path(),
            &snapshot,
            &mut watcher,
        ));
        drain_for(&rx, Duration::from_millis(200));
        fs::write(&path, "second\n").unwrap();
        assert!(watch_event_arrives(&rx), "restored watch should observe a later write");
    }

    #[cfg(unix)]
    fn write_projected_token(root: &Path, contents: &str) -> PathBuf {
        let data = root.join("..data");
        fs::create_dir_all(&data).unwrap();
        fs::write(data.join("token"), contents).unwrap();
        let link = root.join("token");
        std::os::unix::fs::symlink("..data/token", &link).unwrap();
        link
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn data_shortcut_swap_replaces_the_saved_token() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("nemo");
        fs::create_dir_all(&root).unwrap();
        let path = write_projected_token(&root, "first-token\n");
        let token = ServiceAccountToken::open(&path).unwrap();

        let next = root.join("..data_tmp");
        fs::create_dir_all(&next).unwrap();
        fs::write(next.join("token"), "second-token\n").unwrap();
        let current_data = root.join("..data");
        fs::rename(&current_data, root.join("..data_old")).unwrap();
        fs::rename(&next, &current_data).unwrap();

        let current = wait_for(&token, |value| value == Some("second-token"));
        assert_eq!(current.as_ref().map(|value| value.as_str()), Some("second-token"));
    }
}
