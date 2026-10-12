// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`ServiceAccountTokenFilter`] — attaches a Kubernetes pod identity token
//! to every outgoing request.
//!
//! # Overview
//!
//! Kubernetes writes a short-lived login token to a file inside the pod and
//! replaces it automatically before it expires. This filter reads that file
//! once at startup, keeps the value in memory, and watches the file for
//! changes in the background. Every request gets the token from memory — no
//! disk access on the hot path. When Kubernetes swaps in a new token the
//! background watcher picks it up in milliseconds without a restart.
//!
//! Use this in an `outbound_chain` to prove Praxis's identity to a protected
//! upstream — for example, the auth proxy in front of a NeMo Guardrails
//! service.
//!
//! # Security
//!
//! - The token is never written to logs, traces, or error messages.
//! - If the file is missing, empty, or too large at any point, all requests
//!   are rejected with `503` rather than forwarded without a token.
//! - The target header cannot be one that Praxis manages internally or one
//!   that controls the HTTP connection (e.g. `host`, `proxy-authorization`).
//!
//! # YAML config
//!
//! ```yaml
//! filter: service_account_token
//! token_file: /var/run/secrets/tokens/nemo/token
//! header: Authorization   # optional, defaults to Authorization
//! prefix: "Bearer "       # optional, defaults to "Bearer "
//! ```

use std::{
    fs::File,
    io::Read as _,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    time::Duration,
};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use http::{HeaderName, HeaderValue, header::AUTHORIZATION};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher as _};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;
use zeroize::Zeroizing;

use super::descriptor::RESERVED_HEADER_PREFIXES;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Max length for the token file path.
const MAX_PATH_LEN: usize = 4096;

/// Max length for the header value prefix (e.g. `"Bearer "`).
const MAX_PREFIX_LEN: usize = 64;

/// Max token size accepted from the file.
const MAX_TOKEN_BYTES: usize = 16 * 1024;

/// Read one byte past the limit so we can detect oversized files without
/// loading them fully into memory.
const MAX_TOKEN_READ_BYTES: u64 = MAX_TOKEN_BYTES as u64 + 1;

/// Default prefix prepended to the token in the header value.
const DEFAULT_PREFIX: &str = "Bearer ";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `service_account_token` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceAccountTokenConfig {
    /// Path to the token file. Must exist and be non-empty at startup.
    /// The file is watched for changes after startup.
    token_file: String,

    /// Header to set on outgoing requests. Defaults to `Authorization`.
    /// Cannot be an internal Praxis header or a connection-level header
    /// (`host`, `content-length`, `proxy-authorization`, etc.).
    #[serde(default)]
    header: Option<String>,

    /// Text prepended to the token in the header value. Defaults to
    /// `"Bearer "`. Use `""` to send the raw token with no prefix.
    #[serde(default)]
    prefix: Option<String>,
}

// -----------------------------------------------------------------------------
// Internal types
// -----------------------------------------------------------------------------

/// The token value held in memory and swapped out on each rotation.
///
/// `None` means the last file read failed; requests are rejected with 503
/// until a good token is available again.
struct TokenSnapshot {
    /// The fully assembled header value (`"{prefix}{token}"`), or `None` if
    /// the last read failed. Zeroized on drop so the token doesn't linger.
    header_value: Option<Zeroizing<String>>,
}

/// Messages sent to the background watcher thread.
enum WatcherMessage {
    /// A filesystem change was detected.
    Event(notify::Event),
    /// Shut down — sent when the filter is dropped.
    Shutdown,
}

/// Keeps the background watcher thread alive for as long as the filter exists.
struct TokenReloadHandle {
    /// Sends the shutdown signal to the watcher thread.
    shutdown: mpsc::Sender<WatcherMessage>,
    /// The watcher thread; joined when this handle is dropped.
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for TokenReloadHandle {
    fn drop(&mut self) {
        drop(self.shutdown.send(WatcherMessage::Shutdown));
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }
}

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

/// Attaches a Kubernetes pod identity token to every outgoing request.
/// See the module documentation for the full behaviour description.
pub struct ServiceAccountTokenFilter {
    /// Header to set on each request.
    header_name: HeaderName,
    /// Latest token, updated in the background when the file changes.
    snapshot: Arc<ArcSwap<TokenSnapshot>>,
    /// Keeps the watcher thread alive; dropped after the snapshot.
    _reload_handle: TokenReloadHandle,
}

impl ServiceAccountTokenFilter {
    /// Build from YAML config. Fails if the token file can't be read, the
    /// header or prefix are invalid, or the background watcher can't start.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ServiceAccountTokenConfig = parse_filter_config("service_account_token", value)?;

        validate_bounded("token_file", &cfg.token_file, MAX_PATH_LEN)?;
        let header_name = resolve_header_name(cfg.header.as_deref())?;
        let prefix = cfg.prefix.unwrap_or_else(|| DEFAULT_PREFIX.to_owned());
        validate_prefix(&prefix)?;

        let token_file = PathBuf::from(&cfg.token_file);

        // Read the token now so a bad config fails at startup.
        let initial_token = read_token(&token_file)?;
        let initial_value = format!("{prefix}{}", initial_token.as_str());
        // Check the assembled value is a valid HTTP header value before storing it.
        HeaderValue::from_str(&initial_value).map_err(|e| -> FilterError {
            format!("service_account_token: assembled header value is not valid HTTP: {e}").into()
        })?;

        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot {
            header_value: Some(Zeroizing::new(initial_value)),
        }));

        let reload_handle = spawn_token_watcher(token_file, prefix, Arc::clone(&snapshot))?;

        Ok(Box::new(Self {
            header_name,
            snapshot,
            _reload_handle: reload_handle,
        }))
    }
}

#[async_trait]
impl HttpFilter for ServiceAccountTokenFilter {
    fn name(&self) -> &'static str {
        "service_account_token"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Lock-free snapshot load: no file I/O, no syscall on the hot path.
        let snapshot = self.snapshot.load_full();
        let Some(ref assembled) = snapshot.header_value else {
            tracing::warn!("service_account_token: token unavailable; failing closed");
            return Ok(FilterAction::Reject(Rejection::status(503)));
        };

        // The snapshot only holds values that already passed HeaderValue::from_str
        // in from_config and reload_token.
        let header_value = HeaderValue::from_str(assembled.as_str())
            .expect("snapshot contains a pre-validated header value");

        // Strip any caller-supplied value on this header before setting ours,
        // so the caller can't smuggle their own token through to the upstream.
        ctx.request_headers_to_remove.push(self.header_name.clone());
        ctx.request_headers_to_set
            .push((self.header_name.clone(), header_value));

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Watcher
// -----------------------------------------------------------------------------

/// Re-read the token file and publish a new snapshot, failing closed on any
/// error. The token value is never included in diagnostics — only the file
/// path and the error from the OS or HTTP layer appear in the warning.
fn reload_token(path: &Path, prefix: &str, snapshot: &ArcSwap<TokenSnapshot>) {
    let header_value = match read_token(path) {
        Ok(token) => {
            let assembled = format!("{prefix}{}", token.as_str());
            match HeaderValue::from_str(&assembled) {
                Ok(_) => Some(Zeroizing::new(assembled)),
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        %error,
                        "service_account_token: assembled header value is not valid HTTP; failing closed"
                    );
                    None
                },
            }
        },
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "service_account_token: token unavailable; failing closed"
            );
            None
        },
    };
    snapshot.store(Arc::new(TokenSnapshot { header_value }));
}

/// Apply one filesystem event from the notify watcher.
fn process_watcher_event(event: &notify::Event, path: &Path, prefix: &str, snapshot: &ArcSwap<TokenSnapshot>) {
    if event.need_rescan() || matches!(event.kind, EventKind::Other) {
        reload_token(path, prefix, snapshot);
        return;
    }
    if !matches!(
        event.kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    ) {
        return;
    }
    let affected = event
        .paths
        .iter()
        .any(|p| p == path || p.parent() == path.parent());
    if affected {
        reload_token(path, prefix, snapshot);
    }
}

/// Start watching the token file and keep the in-memory snapshot up to date.
///
/// Starts a background thread that:
/// 1. Watches the token file's directory for changes.
/// 2. Re-reads the file immediately to catch any rotation that happened
///    between construction and the watcher becoming active.
/// 3. Signals that startup finished — this function blocks until that point.
/// 4. Reloads the token whenever the file is created, changed, or deleted.
///
/// The thread stops cleanly when the returned [`TokenReloadHandle`] is dropped.
fn spawn_token_watcher(
    path: PathBuf,
    prefix: String,
    snapshot: Arc<ArcSwap<TokenSnapshot>>,
) -> Result<TokenReloadHandle, FilterError> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (message_tx, message_rx) = mpsc::channel();
    let callback_tx = message_tx.clone();

    let thread = std::thread::Builder::new()
        .name("service-account-token-watcher".to_owned())
        .spawn(move || {
            let mut watcher: RecommendedWatcher = match notify::recommended_watcher(move |event| {
                if let Ok(event) = event {
                    drop(callback_tx.send(WatcherMessage::Event(event)));
                }
            }) {
                Ok(w) => w,
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

            // Start watching before re-reading so we can't miss a rotation that
            // happens between the two. Any filesystem event that fires during
            // the read stays queued and is handled after startup completes.
            match read_token(&path) {
                Ok(token) => {
                    let assembled = format!("{prefix}{}", token.as_str());
                    match HeaderValue::from_str(&assembled) {
                        Ok(_) => {
                            snapshot.store(Arc::new(TokenSnapshot {
                                header_value: Some(Zeroizing::new(assembled)),
                            }));
                        },
                        Err(error) => {
                            drop(ready_tx.send(Err(format!(
                                "assembled header value is not valid HTTP: {error}"
                            ))));
                            return;
                        },
                    }
                },
                Err(error) => {
                    drop(ready_tx.send(Err(format!("failed authoritative token read: {error}"))));
                    return;
                },
            }
            drop(ready_tx.send(Ok(())));

            while let Ok(message) = message_rx.recv() {
                let WatcherMessage::Event(event) = message else {
                    break;
                };
                process_watcher_event(&event, &path, &prefix, &snapshot);
            }
        })
        .map_err(|error| format!("service_account_token: failed to spawn watcher: {error}"))?;

    let ready = match ready_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(ready) => ready,
        Err(error) => {
            drop(message_tx.send(WatcherMessage::Shutdown));
            drop(thread.join());
            return Err(format!("service_account_token: watcher did not initialize: {error}").into());
        },
    };
    if let Err(error) = ready {
        drop(message_tx.send(WatcherMessage::Shutdown));
        drop(thread.join());
        return Err(format!("service_account_token: {error}").into());
    }

    Ok(TokenReloadHandle {
        shutdown: message_tx,
        thread: Some(thread),
    })
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Pick the header to inject the token into, defaulting to `Authorization`.
///
/// Rejects headers that Praxis owns internally (they'd be stripped before
/// leaving the proxy, so the token would never arrive) and headers that
/// control the HTTP connection itself (writing to them corrupts routing or,
/// in the case of `proxy-authorization`, hands the token to a middleman
/// instead of the real destination).
fn resolve_header_name(configured: Option<&str>) -> Result<HeaderName, FilterError> {
    let Some(header) = configured else {
        return Ok(AUTHORIZATION);
    };
    validate_bounded("header", header, MAX_PATH_LEN)?;
    let name: HeaderName = header.parse().map_err(|error| -> FilterError {
        format!("service_account_token: invalid header '{header}': {error}").into()
    })?;
    if RESERVED_HEADER_PREFIXES
        .iter()
        .any(|prefix| name.as_str().starts_with(prefix))
    {
        return Err(
            format!("service_account_token: header '{header}' must not use a reserved internal header prefix").into(),
        );
    }
    if name != AUTHORIZATION && praxis_ai_apis::promotion::is_transport_controlled_header_lowercase(name.as_str()) {
        return Err(format!(
            "service_account_token: header '{header}' is transport-controlled and must not receive the service account token"
        )
        .into());
    }
    Ok(name)
}

/// Validate the configured header-value prefix.
fn validate_prefix(prefix: &str) -> Result<(), FilterError> {
    if prefix.len() > MAX_PREFIX_LEN {
        return Err(format!("service_account_token: prefix must be at most {MAX_PREFIX_LEN} bytes").into());
    }
    Ok(())
}

/// Validate a non-empty value against a byte limit.
fn validate_bounded(field: &str, value: &str, maximum: usize) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > maximum {
        return Err(format!("service_account_token: {field} must contain 1-{maximum} bytes").into());
    }
    Ok(())
}

/// Read and validate the complete token file.
///
/// Reads one byte beyond [`MAX_TOKEN_BYTES`] to distinguish an exact-limit
/// file from an oversized one without buffering an unbounded file in full.
/// The token value is never included in the returned error.
fn read_token(path: &Path) -> Result<Zeroizing<String>, FilterError> {
    let file = File::open(path).map_err(|error| -> FilterError {
        format!("service_account_token: cannot read '{}': {error}", path.display()).into()
    })?;
    let mut content = Zeroizing::new(String::new());
    file.take(MAX_TOKEN_READ_BYTES)
        .read_to_string(&mut content)
        .map_err(|error| -> FilterError {
            format!("service_account_token: cannot read '{}': {error}", path.display()).into()
        })?;
    if content.len() > MAX_TOKEN_BYTES {
        return Err(format!("service_account_token: token file exceeds {MAX_TOKEN_BYTES} bytes").into());
    }
    let token = Zeroizing::new(content.trim().to_owned());
    if token.is_empty() {
        return Err(format!("service_account_token: token file '{}' is empty", path.display()).into());
    }
    Ok(token)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use http::Method;

    use super::*;

    // -------------------------------------------------------------------------
    // Config Validation
    // -------------------------------------------------------------------------

    #[test]
    fn missing_file_rejected_at_construction() {
        let err = parse("token_file: /nonexistent/path/to/token")
            .err()
            .expect("must fail");
        assert!(err.to_string().contains("cannot read"), "{err}");
    }

    #[test]
    fn empty_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "   \n").unwrap();
        let yaml = format!("token_file: {}", path.display());
        let err = parse(&yaml).err().expect("empty file must be rejected");
        assert!(err.to_string().contains("is empty"), "{err}");
    }

    #[test]
    fn oversized_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, vec![b't'; MAX_TOKEN_BYTES + 1]).unwrap();
        let yaml = format!("token_file: {}", path.display());
        let err = parse(&yaml).err().expect("oversized file must be rejected");
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn blank_token_file_path_rejected() {
        let err = parse("token_file: ''").err().expect("must fail");
        assert!(err.to_string().contains("token_file"), "{err}");
    }

    #[test]
    fn reserved_prefix_header_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let yaml = format!("token_file: {}\nheader: x-praxis-internal", path.display());
        let err = parse(&yaml).err().expect("reserved-prefix header must be rejected");
        assert!(err.to_string().contains("reserved internal header prefix"), "{err}");
    }

    #[test]
    fn transport_controlled_header_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        for header in ["host", "content-length", "proxy-authorization"] {
            let yaml = format!("token_file: {}\nheader: {header}", path.display());
            let err = parse(&yaml)
                .err()
                .expect("transport-controlled header must be rejected");
            assert!(err.to_string().contains("transport-controlled"), "{header}: {err}");
        }
    }

    #[test]
    fn oversized_prefix_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let long_prefix = "a".repeat(MAX_PREFIX_LEN + 1);
        let yaml = format!("token_file: {}\nprefix: '{long_prefix}'", path.display());
        let err = parse(&yaml).err().expect("oversized prefix must be rejected");
        assert!(err.to_string().contains("prefix"), "{err}");
    }

    #[test]
    fn valid_minimal_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let yaml = format!("token_file: {}", path.display());
        assert!(parse(&yaml).is_ok(), "valid config must parse");
    }

    // -------------------------------------------------------------------------
    // Request-Time Injection
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn injects_default_authorization_bearer_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "sa-jwt-value");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.request_headers_to_set.len(), 1);
        let (name, value) = &ctx.request_headers_to_set[0];
        assert_eq!(*name, AUTHORIZATION);
        assert_eq!(value.to_str().unwrap(), "Bearer sa-jwt-value");
    }

    #[tokio::test]
    async fn strips_caller_supplied_authorization_before_setting() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "sa-jwt-value");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let mut req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        req.headers
            .insert(AUTHORIZATION, HeaderValue::from_static("Bearer smuggled"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        drop(f.on_request(&mut ctx).await.unwrap());
        assert!(ctx.request_headers_to_remove.contains(&AUTHORIZATION));
        assert_eq!(ctx.request_headers_to_set[0].1.to_str().unwrap(), "Bearer sa-jwt-value");
    }

    #[tokio::test]
    async fn custom_header_and_empty_prefix_injects_raw_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "raw-token");
        let yaml = format!("token_file: {}\nheader: x-service-token\nprefix: ''", path.display());
        let f = parse(&yaml).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        drop(f.on_request(&mut ctx).await.unwrap());
        assert_eq!(ctx.request_headers_to_set[0].0.as_str(), "x-service-token");
        assert_eq!(ctx.request_headers_to_set[0].1.to_str().unwrap(), "raw-token");
    }

    // -------------------------------------------------------------------------
    // Rotation / Watcher
    // -------------------------------------------------------------------------

    #[test]
    fn authoritative_startup_reread_replaces_pre_watcher_snapshot() {
        // Pre-populate the snapshot with token-b (simulates the value read at
        // construction), write token-c to the file, then start the watcher.
        // The authoritative re-read inside spawn_token_watcher must publish
        // token-c before returning.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "token-b\n").unwrap();

        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot {
            header_value: Some(Zeroizing::new("Bearer token-b".to_owned())),
        }));

        std::fs::write(&path, "token-c\n").unwrap();
        let handle = spawn_token_watcher(path, DEFAULT_PREFIX.to_owned(), Arc::clone(&snapshot)).unwrap();

        assert_eq!(
            snapshot.load().header_value.as_deref().unwrap(),
            "Bearer token-c",
            "authoritative re-read must overwrite the pre-watcher snapshot"
        );
        drop(handle);
    }

    #[tokio::test]
    async fn rotation_is_observed_without_restarting_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "token-a");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");

        let mut first = crate::test_utils::make_filter_context(&req);
        drop(f.on_request(&mut first).await.unwrap());
        assert_eq!(first.request_headers_to_set[0].1, "Bearer token-a");

        // Kubernetes rotates a projected token via an atomic rename.
        let replacement = dir.path().join("token.next");
        std::fs::write(&replacement, "token-b\n").unwrap();
        std::fs::rename(replacement, &path).unwrap();

        // Poll until the watcher publishes the new snapshot, or time out.
        let mut second = crate::test_utils::make_filter_context(&req);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                second.request_headers_to_set.clear();
                drop(f.on_request(&mut second).await.unwrap());
                if second.request_headers_to_set[0].1 == "Bearer token-b" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("watcher did not publish the rotated token");
        assert_eq!(second.request_headers_to_set[0].1, "Bearer token-b");
    }

    #[tokio::test]
    async fn token_deleted_after_construction_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "token-a");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        std::fs::remove_file(&path).unwrap();

        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        // Poll until the watcher publishes the fail-closed None snapshot.
        let mut ctx = crate::test_utils::make_filter_context(&req);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                ctx.request_headers_to_set.clear();
                let action = f.on_request(&mut ctx).await.unwrap();
                if matches!(action, FilterAction::Reject(ref r) if r.status == 503) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("watcher did not publish the fail-closed snapshot after file deletion");
        assert!(ctx.request_headers_to_set.is_empty());
    }

    #[test]
    fn watcher_shutdown_wakes_without_receive_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let filter = parse(&format!("token_file: {}", path.display())).unwrap();
        let started = std::time::Instant::now();
        drop(filter);
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "watcher teardown should be woken immediately, not wait on a timeout"
        );
    }

    #[tokio::test]
    async fn token_not_in_filter_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "super-secret-sa-token");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        drop(f.on_request(&mut ctx).await.unwrap());
        for value in ctx.filter_metadata.values() {
            assert!(
                !value.contains("super-secret-sa-token"),
                "token must not leak into filter_metadata: {value}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Security: token not in logs on invalid reload
    // -------------------------------------------------------------------------

    #[test]
    fn invalid_token_replacement_fails_closed_without_token_in_logs() {
        // A rotated token containing a NUL byte is valid UTF-8 but not a
        // valid HTTP header value. reload_token must set the snapshot to None
        // (fail closed) and must NOT include the token value in the warning —
        // only the path and the HTTP error appear (verified by inspection of
        // the tracing::warn! call in reload_token, which logs path and error
        // but not the assembled value).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        // A NUL byte is rejected by HeaderValue::from_str.
        std::fs::write(&path, b"valid-prefix\x00invalid-suffix\n").unwrap();

        let snapshot = Arc::new(ArcSwap::from_pointee(TokenSnapshot {
            header_value: Some(Zeroizing::new("Bearer old-token".to_owned())),
        }));

        reload_token(&path, DEFAULT_PREFIX, &snapshot);

        assert!(
            snapshot.load().header_value.is_none(),
            "snapshot must be None after a reload with an invalid HTTP header value"
        );
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn write_token(dir: &tempfile::TempDir, token: &str) -> PathBuf {
        let path = dir.path().join("token");
        std::fs::write(&path, format!("{token}\n")).unwrap();
        path
    }

    fn parse(yaml: &str) -> Result<Box<dyn HttpFilter>, FilterError> {
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        ServiceAccountTokenFilter::from_config(&val)
    }
}
