// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Scripted HTTP/1 replay backend for inference fixture tests.

use std::{
    collections::VecDeque,
    convert::Infallible,
    future::Future,
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    pin::Pin,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use futures::stream;
use http::{HeaderValue, Request, Response, StatusCode, header};
use http_body_util::{BodyExt as _, Full, Limited, StreamBody, combinators::BoxBody};
use hyper::{
    body::{Body as _, Frame, Incoming},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::{
    net::TcpListener,
    sync::{Notify, watch},
    task::JoinSet,
    time::Instant,
};

use super::{
    FixtureError, RecordedBody, RecordedRequest, RecordedResponse, SseFrame,
    bounds::{MAX_SCENARIO_REQUEST_BODY_BYTES, body_has_rendered_content, validate_response_body},
    header_policy::{http_fixture_headers, recorded_transport_headers, validate_recorded_headers},
};

/// Maximum time allowed for the runtime thread to report startup completion.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum graceful drain allowed after the listener is closed.
const CONNECTION_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Maximum time allowed for aborted connection tasks to report completion.
const FINAL_ABORT_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Maximum time allowed for the dedicated runtime thread to report completion.
const THREAD_COMPLETION_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum time the credential-bearing recorder waits for request headers.
const RECORDING_HEADER_READ_TIMEOUT: Duration = Duration::from_secs(1);

/// Async connection-drain limits used by one server thread.
#[derive(Clone, Copy)]
struct ConnectionTimeouts {
    /// Graceful shutdown window before task abort.
    graceful: Duration,
    /// Final bounded window for aborted tasks to report cancellation.
    aborted: Duration,
}

/// Connection-task drain and optional internal completion observation.
struct ConnectionRuntimeOptions {
    /// Async connection-drain limits.
    timeouts: ConnectionTimeouts,
    /// Optional internal connection-task completion notification.
    connection_finished: Option<Arc<Notify>>,
    /// Optional active connection-task ceiling.
    max_active_connections: Option<usize>,
    /// Optional request-header read timeout.
    header_read_timeout: Option<Duration>,
}

/// Production connection-drain limits.
const CONNECTION_TIMEOUTS: ConnectionTimeouts = ConnectionTimeouts {
    graceful: CONNECTION_DRAIN_TIMEOUT,
    aborted: FINAL_ABORT_DRAIN_TIMEOUT,
};

/// Fixed response returned after all scripted responses have been consumed.
const EXHAUSTED_BODY: &str = r#"{"error":"scripted response exhausted"}"#;

/// Type-erased response body shared by replay and recording HTTP/1 servers.
type ServerBody = BoxBody<Bytes, io::Error>;

/// Replay response body alias retained for the rendering boundary.
type ReplayBody = ServerBody;

/// Live-recorder response body alias retained for the request-handler boundary.
pub(super) type RecordingBody = ServerBody;

/// Recorder callback invoked at the response-body and HTTP/1 connection boundaries.
pub(super) trait ResponseDelivery: Send + Sync {
    /// Hyper consumed the complete response body.
    fn body_delivered(&self);
    /// Hyper flushed and closed the HTTP/1 connection without error.
    fn connection_succeeded(&self);
    /// Delivery ended before both completion boundaries succeeded.
    fn delivery_failed(&self);
}

/// Type-erased response-delivery callback stored only in response extensions.
#[derive(Clone)]
struct DeliveryObserver(Arc<dyn ResponseDelivery>);

/// Connection-local delivery ownership and shutdown ordering.
struct ConnectionDelivery {
    /// Recorder observer installed by the active response, if any.
    observer: Mutex<Option<DeliveryObserver>>,
    /// Whether server shutdown started before the connection completed.
    shutting_down: AtomicBool,
}

impl ConnectionDelivery {
    /// Creates empty per-connection delivery state.
    fn new() -> Self {
        Self {
            observer: Mutex::new(None),
            shutting_down: AtomicBool::new(false),
        }
    }

    /// Registers one response observer or fails it if shutdown already began.
    fn register(&self, observer: DeliveryObserver) {
        if self.shutting_down.load(Ordering::Acquire) {
            observer.0.delivery_failed();
            return;
        }
        *self.observer.lock().unwrap_or_else(PoisonError::into_inner) = Some(observer);
        if self.shutting_down.load(Ordering::Acquire)
            && let Some(observer) = self.observer.lock().unwrap_or_else(PoisonError::into_inner).take()
        {
            observer.0.delivery_failed();
        }
    }

    /// Marks server-driven shutdown and fails an active attempt.
    fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let observer = self.observer.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(observer) = observer {
            observer.0.delivery_failed();
        }
    }

    /// Reports the final Hyper result unless shutdown already failed delivery.
    fn finish(&self, succeeded: bool) {
        let observer = self.observer.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(observer) = observer {
            if succeeded && !self.shutting_down.load(Ordering::Acquire) {
                observer.0.connection_succeeded();
            } else {
                observer.0.delivery_failed();
            }
        }
    }
}

/// Attaches one recorder delivery observer without exposing it on the wire.
pub(super) fn observe_response_delivery(
    mut response: Response<RecordingBody>,
    observer: Arc<dyn ResponseDelivery>,
) -> Response<RecordingBody> {
    response.extensions_mut().insert(DeliveryObserver(observer));
    response
}

/// Body wrapper that distinguishes EOF from cancellation/drop.
struct DeliveryBody {
    /// Original response body.
    inner: RecordingBody,
    /// Observer notified at most once by the body boundary.
    observer: Option<DeliveryObserver>,
    /// Whether body EOF was observed.
    delivered: bool,
}

impl DeliveryBody {
    /// Wraps one response body and handles an already-empty body immediately.
    fn new(inner: RecordingBody, observer: Option<DeliveryObserver>) -> Self {
        let delivered = inner.is_end_stream();
        if delivered && let Some(observer) = &observer {
            observer.0.body_delivered();
        }
        Self {
            inner,
            observer,
            delivered,
        }
    }

    /// Records the first terminal body outcome.
    fn finish_body(&mut self, delivered: bool) {
        if self.delivered {
            return;
        }
        self.delivered = delivered;
        if let Some(observer) = self.observer.take() {
            if delivered {
                observer.0.body_delivered();
            } else {
                observer.0.delivery_failed();
            }
        }
    }
}

impl hyper::body::Body for DeliveryBody {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match Pin::new(&mut self.inner).poll_frame(cx) {
            std::task::Poll::Ready(None) => {
                self.finish_body(true);
                std::task::Poll::Ready(None)
            },
            std::task::Poll::Ready(Some(Err(error))) => {
                self.finish_body(false);
                std::task::Poll::Ready(Some(Err(error)))
            },
            outcome => outcome,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for DeliveryBody {
    fn drop(&mut self) {
        self.finish_body(false);
    }
}

/// Future returned by the live recording server's request handler.
type RecordingHandlerFuture = Pin<Box<dyn Future<Output = Response<RecordingBody>> + Send>>;

/// Shared request handler installed on the live recording server.
pub(super) type RecordingHandler = Arc<dyn Fn(Request<Incoming>) -> RecordingHandlerFuture + Send + Sync>;

/// Mutable script and capture state shared by connection tasks and the guard.
struct ReplayState {
    /// Requests captured in arrival order.
    exchanges: Vec<RecordedRequest>,
    /// Responses still available to serve in order.
    responses: VecDeque<Arc<RecordedResponse>>,
    /// Whether one proxy readiness `GET /` should be answered out of band.
    ignore_readiness_probe: bool,
    /// Count of requests that could not be represented faithfully.
    malformed_requests: usize,
}

/// Shared synchronization primitives for the server and waiters.
struct SharedState {
    /// Script and capture state; no guard is held across an await point.
    replay: Mutex<ReplayState>,
    /// Notifies waiters after a request has been captured.
    captured: Notify,
}

/// Test-selectable startup behavior used to exercise partial initialization.
enum StartupMode {
    /// Normal listener startup.
    Normal,
    /// Publish a bound address, report failure, and wait for caller shutdown.
    #[cfg(test)]
    FailAfterBind {
        /// Address acquired before the injected failure.
        acquired_addr: Arc<Mutex<Option<SocketAddr>>>,
        /// Set only when the runtime thread is about to return.
        thread_exited: Arc<AtomicBool>,
    },
}

/// Which public wrapper owns one shared dedicated server.
#[derive(Clone, Copy)]
enum ServerKind {
    /// Scripted replay backend.
    Replay,
    /// Live recording proxy.
    Recording,
}

/// Test-selectable completion behavior for retryable timeout coverage.
enum CompletionMode {
    /// Publish completion immediately after async server shutdown.
    Normal,
    /// Wait on a deterministic barrier before publishing completion.
    #[cfg(test)]
    Hold(Arc<std::sync::Barrier>),
}

/// Test-selectable runtime teardown behavior.
enum RuntimeTeardownMode {
    /// Drop the runtime without injected blocking work.
    Normal,
    /// Hold one runtime blocking task until the test releases it.
    #[cfg(test)]
    Hold {
        /// Signals that the blocking task entered its deterministic gate.
        entered: std::sync::mpsc::SyncSender<()>,
        /// Gate that prevents runtime teardown from completing.
        release: Arc<std::sync::Barrier>,
    },
}

/// Test-selectable dedicated OS-thread launch behavior.
enum ThreadSpawnMode {
    /// Launch the runtime thread normally.
    Normal,
    /// Return the same error as an OS thread launch failure.
    #[cfg(test)]
    Fail,
}

/// Owned startup settings, including deterministic test seams.
struct DedicatedStartOptions {
    /// Wrapper-specific error vocabulary.
    kind: ServerKind,
    /// Optional injected startup behavior.
    startup_mode: StartupMode,
    /// Optional completion publication gate.
    completion_mode: CompletionMode,
    /// Optional blocking-runtime teardown gate.
    runtime_teardown_mode: RuntimeTeardownMode,
    /// Optional OS-thread launch failure injection.
    thread_spawn_mode: ThreadSpawnMode,
    /// Optional internal connection-task completion notification.
    connection_finished: Option<Arc<Notify>>,
    /// Optional internal notification emitted before each stop completion wait.
    stop_attempted: Option<std::sync::mpsc::SyncSender<()>>,
    /// OS-thread completion wait.
    completion_timeout: Duration,
    /// Async connection-drain limits.
    connection_timeouts: ConnectionTimeouts,
    /// Optional active connection-task ceiling.
    max_active_connections: Option<usize>,
    /// Optional request-header read timeout.
    header_read_timeout: Option<Duration>,
}

/// Values moved together into the dedicated runtime thread.
struct DedicatedThreadStart {
    /// Caller-bound nonblocking listener.
    listener: std::net::TcpListener,
    /// Shared request handler.
    handler: RecordingHandler,
    /// One-shot server shutdown receiver.
    shutdown: tokio::sync::oneshot::Receiver<()>,
    /// Bounded startup result sender.
    ready: std::sync::mpsc::SyncSender<Result<SocketAddr, FixtureError>>,
    /// Optional injected startup behavior.
    startup_mode: StartupMode,
    /// Wrapper-specific error vocabulary.
    kind: ServerKind,
    /// Async connection-drain limits.
    connection_timeouts: ConnectionTimeouts,
    /// Optional internal connection-task completion notification.
    connection_finished: Option<Arc<Notify>>,
    /// Optional active connection-task ceiling.
    max_active_connections: Option<usize>,
    /// Optional request-header read timeout.
    header_read_timeout: Option<Duration>,
}

/// Opaque lifecycle failure category rendered per public wrapper.
enum LifecycleFailure {
    /// Tokio runtime construction failed.
    Runtime,
    /// Loopback listener binding or nonblocking setup failed.
    Bind,
    /// Bound address inspection failed.
    Address,
    /// OS runtime thread creation failed.
    Spawn,
    /// Readiness was not reported before the fixed deadline.
    StartupTimeout,
    /// Readiness channel closed unexpectedly.
    StartupChannel,
    /// Completion was not reported before the fixed deadline.
    CompletionTimeout,
    /// Completion channel closed without a result.
    CompletionChannel,
    /// Dedicated runtime thread panicked after reporting completion.
    ThreadPanic,
    /// Listener accept failed.
    Accept,
    /// A connection task panicked.
    ConnectionTask,
    /// Connections did not drain before graceful timeout.
    ConnectionDrain,
    /// Aborted connections did not report completion before final timeout.
    FinalAbortDrain,
}

/// One owned dedicated-thread HTTP/1 lifecycle shared by replay and recording.
///
/// A successful explicit stop observes completion only after Tokio runtime
/// teardown, then joins the OS thread before returning. `Drop` makes the same
/// bounded attempt as best-effort cleanup and retries once when completion
/// ownership remains. After two unrecoverable timeouts Rust cannot force-join
/// the thread, so dropping its retained handle may detach it.
struct DedicatedHttp1Server {
    /// Bound loopback address.
    addr: SocketAddr,
    /// Wrapper-specific opaque errors and tracing labels.
    kind: ServerKind,
    /// Shutdown signal sent at most once.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    /// Completion received before joining the OS thread.
    completion: Mutex<std::sync::mpsc::Receiver<Result<(), FixtureError>>>,
    /// Runtime thread retained across a completion timeout.
    thread: Option<JoinHandle<()>>,
    /// Test-overridable completion wait.
    completion_timeout: Duration,
    /// Optional internal notification emitted before each stop completion wait.
    stop_attempted: Option<std::sync::mpsc::SyncSender<()>>,
}

impl DedicatedHttp1Server {
    /// Starts one validated handler on a loopback-only dedicated runtime.
    fn start(handler: RecordingHandler, kind: ServerKind, startup_mode: StartupMode) -> Result<Self, FixtureError> {
        Self::start_with_options(
            handler,
            DedicatedStartOptions {
                kind,
                startup_mode,
                completion_mode: CompletionMode::Normal,
                runtime_teardown_mode: RuntimeTeardownMode::Normal,
                thread_spawn_mode: ThreadSpawnMode::Normal,
                connection_finished: None,
                stop_attempted: None,
                completion_timeout: THREAD_COMPLETION_TIMEOUT,
                connection_timeouts: CONNECTION_TIMEOUTS,
                max_active_connections: None,
                header_read_timeout: None,
            },
        )
    }

    /// Shared constructor with deterministic test seams for lifecycle edges.
    #[expect(
        clippy::too_many_lines,
        reason = "listener, channels, owned thread launch, and readiness cleanup are one startup transaction"
    )]
    fn start_with_options(handler: RecordingHandler, options: DedicatedStartOptions) -> Result<Self, FixtureError> {
        let DedicatedStartOptions {
            kind,
            startup_mode,
            completion_mode,
            runtime_teardown_mode,
            thread_spawn_mode,
            connection_finished,
            stop_attempted,
            completion_timeout,
            connection_timeouts,
            max_active_connections,
            header_read_timeout,
        } = options;
        let listener = std::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
            .map_err(|_source| lifecycle_error(kind, LifecycleFailure::Bind))?;
        listener
            .set_nonblocking(true)
            .map_err(|_source| lifecycle_error(kind, LifecycleFailure::Bind))?;
        let addr = listener
            .local_addr()
            .map_err(|_source| lifecycle_error(kind, LifecycleFailure::Address))?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (completion_tx, completion_rx) = std::sync::mpsc::sync_channel(1);
        let thread_main = move || {
            let result = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => {
                    let result = runtime.block_on(run_dedicated_server(DedicatedThreadStart {
                        listener,
                        handler,
                        shutdown: shutdown_rx,
                        ready: ready_tx,
                        startup_mode,
                        kind,
                        connection_timeouts,
                        connection_finished,
                        max_active_connections,
                        header_read_timeout,
                    }));
                    #[cfg(test)]
                    start_runtime_teardown_gate(&runtime, runtime_teardown_mode);
                    #[cfg(not(test))]
                    let RuntimeTeardownMode::Normal = runtime_teardown_mode;
                    drop(runtime);
                    result
                },
                Err(_source) => {
                    let _sent = ready_tx.send(Err(lifecycle_error(kind, LifecycleFailure::Runtime)));
                    Err(lifecycle_error(kind, LifecycleFailure::Runtime))
                },
            };
            wait_for_completion_gate(&completion_mode);
            let _sent = completion_tx.send(result);
        };
        let thread = match thread_spawn_mode {
            ThreadSpawnMode::Normal => std::thread::Builder::new()
                .name(
                    match kind {
                        ServerKind::Replay => "inference-replay-backend",
                        ServerKind::Recording => "inference-recording-proxy",
                    }
                    .to_owned(),
                )
                .spawn(thread_main),
            #[cfg(test)]
            ThreadSpawnMode::Fail => Err(io::Error::other("injected thread spawn failure")),
        }
        .map_err(|_source| lifecycle_error(kind, LifecycleFailure::Spawn))?;
        let mut server = Self {
            addr,
            kind,
            shutdown: Some(shutdown_tx),
            completion: Mutex::new(completion_rx),
            thread: Some(thread),
            completion_timeout,
            stop_attempted,
        };
        match ready_rx.recv_timeout(STARTUP_TIMEOUT) {
            Ok(Ok(ready_addr)) if ready_addr == addr => Ok(server),
            Ok(Ok(_)) => {
                server.stop()?;
                Err(lifecycle_error(kind, LifecycleFailure::Address))
            },
            Ok(Err(error)) => {
                server.stop()?;
                Err(error)
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                server.stop()?;
                Err(lifecycle_error(kind, LifecycleFailure::StartupTimeout))
            },
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                server.stop()?;
                Err(lifecycle_error(kind, LifecycleFailure::StartupChannel))
            },
        }
    }

    /// Returns the bound loopback address.
    const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Sends shutdown and joins only after bounded completion acknowledgement.
    fn stop(&mut self) -> Result<(), FixtureError> {
        if let Some(shutdown) = self.shutdown.take() {
            let _sent = shutdown.send(());
        }
        if self.thread.is_none() {
            return Ok(());
        }
        if let Some(stop_attempted) = &self.stop_attempted {
            let _sent = stop_attempted.try_send(());
        }
        let completion = self.completion.lock().unwrap_or_else(PoisonError::into_inner);
        let result = match completion.recv_timeout(self.completion_timeout) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(lifecycle_error(self.kind, LifecycleFailure::CompletionTimeout));
            },
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err(lifecycle_error(self.kind, LifecycleFailure::CompletionChannel))
            },
        };
        drop(completion);
        let thread = self.thread.take().expect("BUG: checked dedicated thread ownership");
        thread
            .join()
            .map_err(|_panic| lifecycle_error(self.kind, LifecycleFailure::ThreadPanic))?;
        result
    }
}

/// Consumes the optional test gate before thread completion is published.
fn wait_for_completion_gate(mode: &CompletionMode) {
    match mode {
        CompletionMode::Normal => {},
        #[cfg(test)]
        CompletionMode::Hold(barrier) => {
            barrier.wait();
        },
    }
}

/// Starts optional test-only blocking work whose completion gates runtime drop.
#[cfg(test)]
fn start_runtime_teardown_gate(runtime: &tokio::runtime::Runtime, mode: RuntimeTeardownMode) {
    match mode {
        RuntimeTeardownMode::Normal => {},
        #[cfg(test)]
        RuntimeTeardownMode::Hold { entered, release } => {
            let _blocking = runtime.spawn_blocking(move || {
                let _sent = entered.send(());
                release.wait();
            });
        },
    }
}

impl Drop for DedicatedHttp1Server {
    fn drop(&mut self) {
        let first_failed = self.stop().is_err();
        let cleanup_failed = first_failed
            && if self.thread.is_some() {
                self.stop().is_err()
            } else {
                true
            };
        if cleanup_failed {
            tracing::warn!(addr = %self.addr, "bounded inference HTTP/1 server drop cleanup was incomplete");
        }
    }
}

/// RAII guard for one loopback-only scripted HTTP/1 backend.
///
/// Successful [`ScriptedHttpServer::finish`] tears down the Tokio runtime,
/// joins its OS thread, and releases the listener before returning. Guard drop
/// is bounded best-effort cleanup and logs rather than claiming that guarantee.
pub(super) struct ScriptedHttpServer {
    /// Shared dedicated listener/runtime ownership.
    server: DedicatedHttp1Server,
    /// Shared script, captures, and notification state.
    shared: Arc<SharedState>,
}

/// Dedicated-thread loopback HTTP/1 server used by the live recorder.
///
/// A successful explicit stop tears down and joins the runtime thread, making
/// listener and connection cleanup observable. Drop remains bounded best effort.
pub(super) struct RecordingHttpServer {
    /// Shared dedicated listener/runtime ownership.
    server: DedicatedHttp1Server,
}

impl RecordingHttpServer {
    /// Starts a loopback HTTP/1 server after its handler state is validated.
    ///
    /// # Errors
    ///
    /// Returns an opaque runtime error if the runtime thread or listener cannot
    /// be started within the bounded readiness window.
    pub(super) fn start(
        handler: RecordingHandler,
        connection_finished: Option<Arc<Notify>>,
    ) -> Result<Self, FixtureError> {
        let server = DedicatedHttp1Server::start_with_options(
            handler,
            DedicatedStartOptions {
                kind: ServerKind::Recording,
                startup_mode: StartupMode::Normal,
                completion_mode: CompletionMode::Normal,
                runtime_teardown_mode: RuntimeTeardownMode::Normal,
                thread_spawn_mode: ThreadSpawnMode::Normal,
                connection_finished,
                stop_attempted: None,
                completion_timeout: THREAD_COMPLETION_TIMEOUT,
                connection_timeouts: CONNECTION_TIMEOUTS,
                max_active_connections: Some(1),
                header_read_timeout: Some(RECORDING_HEADER_READ_TIMEOUT),
            },
        )?;
        Ok(Self { server })
    }

    /// Returns the bound loopback address.
    pub(super) const fn addr(&self) -> SocketAddr {
        self.server.addr()
    }

    /// Signals shutdown and joins only after runtime teardown completes.
    pub(super) fn stop(&mut self) -> Result<(), FixtureError> {
        self.server.stop()
    }
}

impl ScriptedHttpServer {
    /// Starts a loopback-only HTTP/1 backend with ordered scripted responses.
    ///
    /// # Errors
    ///
    /// Returns an opaque replay runtime error if the runtime thread, listener,
    /// or startup coordination channel cannot be created.
    #[cfg(test)]
    pub(super) fn start(responses: Vec<RecordedResponse>) -> Result<Self, FixtureError> {
        Self::start_shared(responses.into_iter().map(Arc::new).collect())
    }

    /// Starts a backend whose script shares response ownership with its caller.
    ///
    /// This is used by the scenario runner so a response body can be rendered
    /// on the wire and later moved into the fixture without cloning provider or
    /// streaming payload data.
    ///
    /// # Errors
    ///
    /// Returns the same opaque startup errors as the other scripted-server constructors.
    #[cfg(test)]
    pub(super) fn start_shared(responses: Vec<Arc<RecordedResponse>>) -> Result<Self, FixtureError> {
        Self::start_shared_inner(responses, false)
    }

    /// Starts a shared script while reserving `GET /` for proxy readiness.
    ///
    /// The existing proxy test guard performs a synchronous readiness request.
    /// If that request reaches this backend it must not consume or appear as a
    /// scenario exchange.
    ///
    /// # Errors
    ///
    /// Returns the same opaque startup errors as the other scripted-server constructors.
    pub(super) fn start_for_proxy(responses: Vec<Arc<RecordedResponse>>) -> Result<Self, FixtureError> {
        Self::start_shared_inner(responses, true)
    }

    /// Shared startup implementation with optional proxy-readiness handling.
    fn start_shared_inner(
        responses: Vec<Arc<RecordedResponse>>,
        ignore_readiness_probe: bool,
    ) -> Result<Self, FixtureError> {
        Self::start_shared_inner_with_mode(responses, ignore_readiness_probe, StartupMode::Normal)
    }

    /// Starts a backend that deterministically fails after binding its port.
    #[cfg(test)]
    fn start_with_injected_post_bind_failure(
        acquired_addr: Arc<Mutex<Option<SocketAddr>>>,
        thread_exited: Arc<AtomicBool>,
    ) -> Result<Self, FixtureError> {
        Self::start_shared_inner_with_mode(
            Vec::new(),
            false,
            StartupMode::FailAfterBind {
                acquired_addr,
                thread_exited,
            },
        )
    }

    /// Shared startup implementation with deterministic cleanup on every exit.
    fn start_shared_inner_with_mode(
        responses: Vec<Arc<RecordedResponse>>,
        ignore_readiness_probe: bool,
        startup_mode: StartupMode,
    ) -> Result<Self, FixtureError> {
        validate_scripted_responses(&responses)?;
        let shared = Arc::new(SharedState {
            replay: Mutex::new(ReplayState {
                exchanges: Vec::new(),
                responses: responses.into(),
                ignore_readiness_probe,
                malformed_requests: 0,
            }),
            captured: Notify::new(),
        });
        let server_shared = Arc::clone(&shared);
        let handler: RecordingHandler = Arc::new(move |request| {
            let shared = Arc::clone(&server_shared);
            Box::pin(async move {
                match handle_request(request, shared).await {
                    Ok(response) => response,
                    Err(never) => match never {},
                }
            })
        });
        let server = DedicatedHttp1Server::start(handler, ServerKind::Replay, startup_mode)?;

        Ok(Self { server, shared })
    }

    /// Returns the loopback socket address of the scripted backend.
    pub(super) const fn addr(&self) -> SocketAddr {
        self.server.addr()
    }

    /// Ends the one-request proxy readiness exemption after startup completes.
    pub(super) fn finish_proxy_readiness(&self) {
        self.shared
            .replay
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .ignore_readiness_probe = false;
    }

    /// Waits until at least `count` requests are captured or `timeout` elapses.
    ///
    /// The notification future is created before checking the count, preventing
    /// a capture between the state check and waiter registration from being
    /// missed.
    ///
    /// # Errors
    ///
    /// Returns an opaque replay runtime error if the requested count is not
    /// reached before the bounded timeout.
    pub(super) async fn wait_for_exchanges(&self, count: usize, timeout: Duration) -> Result<(), FixtureError> {
        self.wait_for_exchanges_inner(count, timeout, None::<fn()>).await
    }

    /// Test seam that runs a barrier hook after the first false count check.
    #[cfg(test)]
    async fn wait_for_exchanges_after_check<F>(
        &self,
        count: usize,
        timeout: Duration,
        after_check: F,
    ) -> Result<(), FixtureError>
    where
        F: FnOnce() + Send,
    {
        self.wait_for_exchanges_inner(count, timeout, Some(after_check)).await
    }

    /// Wait implementation that enables each notification before inspecting state.
    async fn wait_for_exchanges_inner<F>(
        &self,
        count: usize,
        timeout: Duration,
        mut after_check: Option<F>,
    ) -> Result<(), FixtureError>
    where
        F: FnOnce() + Send,
    {
        let deadline = Instant::now() + timeout;
        loop {
            let notified = self.shared.captured.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let reached = self
                .shared
                .replay
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .exchanges
                .len()
                >= count;
            if reached {
                return Ok(());
            }
            if let Some(after_check) = after_check.take() {
                after_check();
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(runtime_error("scripted backend timed out waiting for exchange count"));
            }
        }
    }

    /// Moves all captured requests out of the backend in arrival order.
    #[cfg(test)]
    pub(super) fn take_exchanges(&self) -> Vec<RecordedRequest> {
        let mut state = self.shared.replay.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut state.exchanges)
    }

    /// Stops the server, then returns captures only after exact accounting.
    pub(super) fn finish(mut self, expected_exchanges: usize) -> Result<Vec<RecordedRequest>, FixtureError> {
        self.stop()?;
        let mut state = self.shared.replay.lock().unwrap_or_else(PoisonError::into_inner);
        if state.malformed_requests != 0 {
            return Err(runtime_error("scripted backend observed a malformed request"));
        }
        if !state.responses.is_empty() {
            return Err(runtime_error("scripted backend did not consume every response"));
        }
        if state.exchanges.len() != expected_exchanges {
            return Err(runtime_error("scripted backend captured an unexpected request count"));
        }
        Ok(std::mem::take(&mut state.exchanges))
    }

    /// Signals shutdown and joins only after runtime teardown completes.
    fn stop(&mut self) -> Result<(), FixtureError> {
        self.server.stop()
    }
}

/// Validates every script before a listener or runtime thread is created.
fn validate_scripted_responses(responses: &[Arc<RecordedResponse>]) -> Result<(), FixtureError> {
    for response in responses {
        let status = StatusCode::from_u16(response.status)
            .map_err(|_source| runtime_error("scripted response status is invalid"))?;
        validate_recorded_headers(&response.headers)?;
        validate_response_body(&response.body)?;
        if status.is_informational() {
            return Err(runtime_error("scripted response status cannot be terminal"));
        }
        if matches!(
            status,
            StatusCode::NO_CONTENT | StatusCode::RESET_CONTENT | StatusCode::NOT_MODIFIED
        ) && body_has_rendered_content(&response.body)
        {
            return Err(runtime_error("scripted response status forbids content"));
        }
    }
    Ok(())
}

/// Converts the caller-bound listener and drives the shared async server.
#[expect(
    clippy::too_many_lines,
    reason = "runtime listener conversion, readiness publication, and injected startup cleanup are one transaction"
)]
async fn run_dedicated_server(start: DedicatedThreadStart) -> Result<(), FixtureError> {
    let DedicatedThreadStart {
        listener,
        handler,
        shutdown,
        ready,
        startup_mode,
        kind,
        connection_timeouts,
        connection_finished,
        max_active_connections,
        header_read_timeout,
    } = start;
    let listener = TcpListener::from_std(listener).map_err(|_source| lifecycle_error(kind, LifecycleFailure::Bind))?;
    let addr = listener
        .local_addr()
        .map_err(|_source| lifecycle_error(kind, LifecycleFailure::Address))?;
    #[cfg(not(test))]
    let StartupMode::Normal = startup_mode;
    #[cfg(test)]
    if let StartupMode::FailAfterBind {
        acquired_addr,
        thread_exited,
    } = startup_mode
    {
        *acquired_addr.lock().unwrap_or_else(PoisonError::into_inner) = Some(addr);
        let _sent = ready.send(Err(runtime_error("injected scripted backend startup failure")));
        let _shutdown = shutdown.await;
        drop(listener);
        thread_exited.store(true, Ordering::SeqCst);
        return Ok(());
    }
    if ready.send(Ok(addr)).is_err() {
        return Ok(());
    }
    serve_http1(
        listener,
        handler,
        shutdown,
        kind,
        ConnectionRuntimeOptions {
            timeouts: connection_timeouts,
            connection_finished,
            max_active_connections,
            header_read_timeout,
        },
    )
    .await
}

/// Accepts HTTP/1 connections, signals graceful shutdown, then drains boundedly.
#[expect(
    clippy::too_many_lines,
    reason = "accept, graceful shutdown, and two bounded drain phases form one lifecycle contract"
)]
async fn serve_http1(
    listener: TcpListener,
    handler: RecordingHandler,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
    kind: ServerKind,
    options: ConnectionRuntimeOptions,
) -> Result<(), FixtureError> {
    let ConnectionRuntimeOptions {
        timeouts,
        connection_finished,
        max_active_connections,
        header_read_timeout,
    } = options;
    let mut connections = JoinSet::new();
    let (connection_shutdown, _) = watch::channel(false);
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => break,
            joined = connections.join_next(), if !connections.is_empty() => {
                if joined.is_some_and(|result| result.is_err()) {
                    return Err(lifecycle_error(kind, LifecycleFailure::ConnectionTask));
                }
            },
            accepted = listener.accept(), if max_active_connections.is_none_or(|max| connections.len() < max) => {
                let (stream, peer_addr) = accepted
                    .map_err(|_source| lifecycle_error(kind, LifecycleFailure::Accept))?;
                let handler = Arc::clone(&handler);
                let connection_shutdown = connection_shutdown.subscribe();
                let connection_finished = connection_finished.clone();
                connections.spawn(async move {
                    run_http1_connection(stream, peer_addr, handler, connection_shutdown, header_read_timeout).await;
                    if let Some(connection_finished) = connection_finished {
                        connection_finished.notify_one();
                    }
                });
            },
        }
    }
    drop(listener);
    let _sent = connection_shutdown.send(true);
    if tokio::time::timeout(timeouts.graceful, drain_connections(&mut connections))
        .await
        .is_ok_and(|drained| drained)
    {
        return Ok(());
    }
    connections.abort_all();
    if tokio::time::timeout(timeouts.aborted, drain_connections(&mut connections))
        .await
        .is_err()
    {
        return Err(lifecycle_error(kind, LifecycleFailure::FinalAbortDrain));
    }
    Err(lifecycle_error(kind, LifecycleFailure::ConnectionDrain))
}

/// Serves one response and reports body, shutdown, and Hyper completion edges.
#[expect(
    clippy::too_many_lines,
    reason = "one connection owns observer registration, graceful shutdown, and final delivery reporting"
)]
async fn run_http1_connection(
    stream: tokio::net::TcpStream,
    peer_addr: SocketAddr,
    handler: RecordingHandler,
    mut shutdown: watch::Receiver<bool>,
    header_read_timeout: Option<Duration>,
) {
    let delivery = Arc::new(ConnectionDelivery::new());
    let service_delivery = Arc::clone(&delivery);
    let service = service_fn(move |request| {
        let handler = Arc::clone(&handler);
        let delivery = Arc::clone(&service_delivery);
        async move {
            let mut response = handler(request).await;
            let observer = response.extensions_mut().remove::<DeliveryObserver>();
            if let Some(observer) = &observer {
                delivery.register(observer.clone());
            }
            Ok::<_, Infallible>(response.map(|body| DeliveryBody::new(body, observer).boxed()))
        }
    });
    let mut builder = http1::Builder::new();
    builder.keep_alive(false);
    if let Some(timeout) = header_read_timeout {
        builder.timer(TokioTimer::new()).header_read_timeout(timeout);
    }
    let connection = builder.serve_connection(TokioIo::new(stream), service);
    tokio::pin!(connection);
    let result = tokio::select! {
        result = &mut connection => result,
        _ = shutdown.changed() => {
            delivery.begin_shutdown();
            connection.as_mut().graceful_shutdown();
            connection.await
        },
    };
    delivery.finish(result.is_ok());
    if let Err(error) = result {
        tracing::debug!(%peer_addr, error = %error, "inference HTTP/1 connection ended");
    }
}

/// Drains a connection set and reports whether every task joined without panic.
async fn drain_connections(connections: &mut JoinSet<()>) -> bool {
    while let Some(joined) = connections.join_next().await {
        if joined.is_err() {
            return false;
        }
    }
    true
}

/// Maps shared lifecycle failures to stable wrapper-specific opaque errors.
#[expect(
    clippy::too_many_lines,
    reason = "the exhaustive wrapper-by-failure mapping keeps every public error opaque and stable"
)]
fn lifecycle_error(kind: ServerKind, failure: LifecycleFailure) -> FixtureError {
    let message = match (kind, failure) {
        (ServerKind::Replay, LifecycleFailure::Runtime) => "scripted backend failed to create Tokio runtime",
        (ServerKind::Recording, LifecycleFailure::Runtime) => "recording proxy failed to create Tokio runtime",
        (ServerKind::Replay, LifecycleFailure::Bind) => "scripted backend failed to bind loopback listener",
        (ServerKind::Recording, LifecycleFailure::Bind) => "recording proxy failed to bind loopback listener",
        (ServerKind::Replay, LifecycleFailure::Address) => "scripted backend failed to read listener address",
        (ServerKind::Recording, LifecycleFailure::Address) => "recording proxy failed to read listener address",
        (ServerKind::Replay, LifecycleFailure::Spawn) => "scripted backend failed to spawn runtime thread",
        (ServerKind::Recording, LifecycleFailure::Spawn) => "recording proxy failed to spawn runtime thread",
        (ServerKind::Replay, LifecycleFailure::StartupTimeout) => "scripted backend startup timed out",
        (ServerKind::Recording, LifecycleFailure::StartupTimeout) => "recording proxy startup timed out",
        (ServerKind::Replay, LifecycleFailure::StartupChannel) => "scripted backend startup channel closed",
        (ServerKind::Recording, LifecycleFailure::StartupChannel) => "recording proxy startup channel closed",
        (ServerKind::Replay, LifecycleFailure::CompletionTimeout) => "scripted backend shutdown timed out",
        (ServerKind::Recording, LifecycleFailure::CompletionTimeout) => "recording proxy shutdown timed out",
        (ServerKind::Replay, LifecycleFailure::CompletionChannel) => "scripted backend completion channel closed",
        (ServerKind::Recording, LifecycleFailure::CompletionChannel) => "recording proxy completion channel closed",
        (ServerKind::Replay, LifecycleFailure::ThreadPanic) => {
            "scripted backend runtime thread panicked during shutdown"
        },
        (ServerKind::Recording, LifecycleFailure::ThreadPanic) => {
            "recording proxy runtime thread panicked during shutdown"
        },
        (ServerKind::Replay, LifecycleFailure::Accept) => "scripted backend listener accept failed",
        (ServerKind::Recording, LifecycleFailure::Accept) => "recording proxy listener accept failed",
        (ServerKind::Replay, LifecycleFailure::ConnectionTask) => "scripted backend connection task panicked",
        (ServerKind::Recording, LifecycleFailure::ConnectionTask) => "recording proxy connection task panicked",
        (ServerKind::Replay, LifecycleFailure::ConnectionDrain | LifecycleFailure::FinalAbortDrain) => {
            "scripted backend connection drain was incomplete"
        },
        (ServerKind::Recording, LifecycleFailure::ConnectionDrain | LifecycleFailure::FinalAbortDrain) => {
            "recording proxy connection drain was incomplete"
        },
    };
    runtime_error(message)
}

/// Captures one bounded request before selecting its scripted response.
#[expect(
    clippy::too_many_lines,
    reason = "capture-before-selection ordering and opaque boundary failures stay visible together"
)]
async fn handle_request(
    request: Request<Incoming>,
    shared: Arc<SharedState>,
) -> Result<Response<ReplayBody>, Infallible> {
    let (parts, incoming) = request.into_parts();
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok());
    let Ok(collected) = Limited::new(incoming, MAX_SCENARIO_REQUEST_BODY_BYTES).collect().await else {
        record_malformed_request(&shared);
        return Ok(opaque_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeded replay limit",
        ));
    };
    let body = collected.to_bytes();
    let (body, parse_failed) = match RecordedBody::from_http(content_type, &body) {
        Ok(body) => (body, false),
        Err(_) => (
            RecordedBody::Base64 {
                data: STANDARD.encode(&body),
            },
            true,
        ),
    };
    let captured = RecordedRequest {
        method: parts.method.as_str().to_owned(),
        path: parts
            .uri
            .path_and_query()
            .map_or_else(|| "/".to_owned(), |path| path.as_str().to_owned()),
        headers: if let Ok(headers) = http_fixture_headers(&parts.headers) {
            headers
        } else {
            record_malformed_request(&shared);
            return Ok(opaque_response(
                StatusCode::BAD_REQUEST,
                "request headers could not be captured",
            ));
        },
        body,
    };
    let is_readiness_probe = captured.method == "GET" && captured.path == "/";
    let response = {
        let mut state = shared.replay.lock().unwrap_or_else(PoisonError::into_inner);
        if state.ignore_readiness_probe && is_readiness_probe {
            state.ignore_readiness_probe = false;
            return Ok(opaque_empty_response(StatusCode::NO_CONTENT));
        }
        state.exchanges.push(captured);
        if parse_failed {
            state.malformed_requests = state.malformed_requests.saturating_add(1);
            None
        } else {
            state.responses.pop_front()
        }
    };
    shared.captured.notify_waiters();

    if parse_failed {
        return Ok(opaque_response(
            StatusCode::BAD_REQUEST,
            "request body could not be parsed",
        ));
    }
    Ok(response.map_or_else(
        || opaque_response(StatusCode::INTERNAL_SERVER_ERROR, "scripted response exhausted"),
        |recorded| {
            render_scripted_response(&recorded).unwrap_or_else(|_| {
                opaque_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "scripted response violated its preflight validation",
                )
            })
        },
    ))
}

/// Records a request that could not be represented as a complete capture.
fn record_malformed_request(shared: &SharedState) {
    let mut state = shared.replay.lock().unwrap_or_else(PoisonError::into_inner);
    state.malformed_requests = state.malformed_requests.saturating_add(1);
    drop(state);
    shared.captured.notify_waiters();
}

/// Converts one recorded response into an HTTP response without fixture framing headers.
fn render_scripted_response(recorded: &Arc<RecordedResponse>) -> Result<Response<ReplayBody>, FixtureError> {
    let status = StatusCode::from_u16(recorded.status)
        .map_err(|_source| runtime_error("scripted response status violated preflight validation"))?;
    let mut response = match &recorded.body {
        RecordedBody::Sse { .. } => Response::new(streaming_sse_body(Arc::clone(recorded))),
        body => Response::new(
            Full::new(Bytes::from(body.render().map_err(|_source| {
                runtime_error("scripted response body violated preflight validation")
            })?))
            .map_err(|never| match never {})
            .boxed(),
        ),
    };
    *response.status_mut() = status;
    response
        .headers_mut()
        .extend(recorded_transport_headers(&recorded.headers)?);
    Ok(response)
}

/// Emits each SSE event and the terminal marker as a distinct body frame.
fn streaming_sse_body(recorded: Arc<RecordedResponse>) -> ReplayBody {
    let (sender, receiver) = tokio::sync::mpsc::channel(1);
    tokio::spawn(async move {
        let RecordedBody::Sse { frames, done } = &recorded.body else {
            return;
        };
        for frame in frames {
            let frame = Frame::data(Bytes::from(render_sse_frame(frame)));
            if sender.send(Ok::<_, io::Error>(frame)).await.is_err() {
                return;
            }
        }
        if *done {
            let terminal = Frame::data(Bytes::from_static(b"data: [DONE]\n\n"));
            let _sent = sender.send(Ok::<_, io::Error>(terminal)).await;
        }
    });
    let frames = stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|frame| (frame, receiver))
    });
    StreamBody::new(frames).boxed()
}

/// Renders one canonical SSE event without combining it with adjacent frames.
fn render_sse_frame(frame: &SseFrame) -> Vec<u8> {
    let mut rendered = Vec::new();
    if let Some(event) = &frame.event {
        push_sse_field(&mut rendered, "event", event);
    }
    for data_line in frame.data.split('\n') {
        push_sse_field(&mut rendered, "data", data_line);
    }
    if let Some(id) = &frame.id {
        push_sse_field(&mut rendered, "id", id);
    }
    if let Some(retry) = frame.retry {
        push_sse_field(&mut rendered, "retry", &retry.to_string());
    }
    rendered.push(b'\n');
    rendered
}

/// Appends one canonical SSE field line.
fn push_sse_field(output: &mut Vec<u8>, name: &str, value: &str) {
    output.extend_from_slice(name.as_bytes());
    output.extend_from_slice(b": ");
    output.extend_from_slice(value.as_bytes());
    output.push(b'\n');
}

/// Builds one deterministic JSON error response without request or fixture data.
fn opaque_response(status: StatusCode, message: &'static str) -> Response<ReplayBody> {
    let body = if message == "scripted response exhausted" {
        Bytes::from_static(EXHAUSTED_BODY.as_bytes())
    } else {
        Bytes::from(serde_json::json!({"error": message}).to_string())
    };
    let mut response = Response::new(Full::new(body).map_err(|never| match never {}).boxed());
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

/// Builds an empty response used only for proxy readiness coordination.
fn opaque_empty_response(status: StatusCode) -> Response<ReplayBody> {
    let mut response = Response::new(Full::new(Bytes::new()).map_err(|never| match never {}).boxed());
    *response.status_mut() = status;
    response
}

/// Creates an opaque replay runtime error with no fixture content.
fn runtime_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use std::{
        collections::BTreeMap,
        io::{Read as _, Write as _},
        net::{Ipv4Addr, TcpListener, TcpStream},
        sync::{
            Arc, Barrier, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use http::{StatusCode, header::HeaderValue};
    use http_body_util::BodyExt as _;
    use serde_json::json;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    use super::{
        super::{RecordedBody, RecordedResponse, SseFrame},
        CONNECTION_TIMEOUTS, CompletionMode, ConnectionTimeouts, DedicatedHttp1Server, DedicatedStartOptions,
        RecordingBody, RecordingHandler, RuntimeTeardownMode, ScriptedHttpServer, ServerKind, StartupMode,
        ThreadSpawnMode, opaque_empty_response,
    };

    async fn assert_listener_released(addr: std::net::SocketAddr) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            match TcpListener::bind(addr) {
                Ok(listener) => {
                    drop(listener);
                    return;
                },
                Err(error)
                    if error.kind() == std::io::ErrorKind::AddrInUse && tokio::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                },
                Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
                    panic!("test listener address remained in use after bounded cleanup");
                },
                Err(error) => panic!("test listener address could not be rebound: {error}"),
            }
        }
    }

    #[tokio::test]
    async fn listener_release_check_tolerates_brief_reuse_by_another_test() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let releaser = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            drop(listener);
        });

        assert_listener_released(addr).await;

        releaser.await.unwrap();
    }

    #[tokio::test]
    async fn opaque_response_json_escapes_message_content() {
        // Catches interpolating a future static message without JSON escaping.
        let response = super::opaque_response(StatusCode::BAD_REQUEST, "quoted \"message\"\nnext line");

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).expect("opaque response body must be valid JSON");

        assert_eq!(value, json!({"error": "quoted \"message\"\nnext line"}));
    }

    #[tokio::test]
    async fn serves_ordered_responses_and_captures_exact_safe_request() {
        let server = ScriptedHttpServer::start(vec![
            RecordedResponse {
                status: 201,
                headers: BTreeMap::from([
                    ("Connection".to_owned(), vec!["x-remove".to_owned()]),
                    ("connection".to_owned(), vec!["x-remove-too".to_owned()]),
                    ("x-remove".to_owned(), vec!["must-not-leak".to_owned()]),
                    ("x-remove-too".to_owned(), vec!["must-not-leak".to_owned()]),
                    ("Authorization".to_owned(), vec!["Bearer must-not-leak".to_owned()]),
                    ("Set-Cookie".to_owned(), vec!["session=must-not-leak".to_owned()]),
                    ("X-Goog-Api-Key".to_owned(), vec!["must-not-leak".to_owned()]),
                    (
                        "x-replay-value".to_owned(),
                        vec!["first".to_owned(), "second".to_owned()],
                    ),
                    ("content-type".to_owned(), vec!["application/json".to_owned()]),
                ]),
                body: RecordedBody::Json {
                    value: json!({"turn": 1}),
                },
            },
            RecordedResponse {
                status: 204,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
        ])
        .expect("scripted server should start");
        assert!(server.addr().ip().is_loopback(), "server must bind only loopback");

        let client = reqwest::Client::new();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.append("content-type", HeaderValue::from_static("application/json"));
        headers.append("x-request-id", HeaderValue::from_static("trace-a"));
        headers.append("x-request-id", HeaderValue::from_static("trace-b"));
        headers.append("authorization", HeaderValue::from_static("Bearer must-not-capture"));
        headers.append("connection", HeaderValue::from_static("x-hop"));
        headers.append("x-hop", HeaderValue::from_static("must-not-capture"));

        let first = client
            .post(format!("http://{}/first?x=a%2Fb&x=2", server.addr()))
            .headers(headers)
            .body(r#"{"message":"hello"}"#)
            .send()
            .await
            .expect("first request should complete");
        assert_eq!(first.status(), 201);
        assert_eq!(
            first
                .headers()
                .get_all("x-replay-value")
                .iter()
                .map(|value| value.to_str().expect("fixture header should be text"))
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert!(
            first
                .headers()
                .get_all("connection")
                .iter()
                .all(|value| !value.as_bytes().eq_ignore_ascii_case(b"x-remove")
                    && !value.as_bytes().eq_ignore_ascii_case(b"x-remove-too")),
            "the protocol stack may add `connection: close`, but scripted nominations must not survive"
        );
        assert!(!first.headers().contains_key("x-remove"));
        assert!(!first.headers().contains_key("x-remove-too"));
        assert!(!first.headers().contains_key("authorization"));
        assert!(!first.headers().contains_key("set-cookie"));
        assert!(!first.headers().contains_key("x-goog-api-key"));
        assert_eq!(first.json::<serde_json::Value>().await.unwrap(), json!({"turn": 1}));

        let second = client
            .put(format!("http://{}/second", server.addr()))
            .send()
            .await
            .expect("second request should complete");
        assert_eq!(second.status(), 204);
        assert!(second.bytes().await.unwrap().is_empty());

        server
            .wait_for_exchanges(2, Duration::from_secs(1))
            .await
            .expect("both exchanges should be captured");
        let exchanges = server.take_exchanges();
        assert_eq!(exchanges.len(), 2);
        assert_eq!(exchanges[0].method, "POST");
        assert_eq!(exchanges[0].path, "/first?x=a%2Fb&x=2");
        assert_eq!(
            exchanges[0].headers.get("content-type"),
            Some(&vec!["application/json".to_owned()])
        );
        assert_eq!(
            exchanges[0].headers.get("x-request-id"),
            Some(&vec!["trace-a".to_owned(), "trace-b".to_owned()])
        );
        assert!(!exchanges[0].headers.contains_key("authorization"));
        assert!(!exchanges[0].headers.contains_key("connection"));
        assert!(!exchanges[0].headers.contains_key("x-hop"));
        assert_eq!(
            exchanges[0].body,
            RecordedBody::Json {
                value: json!({"message": "hello"})
            }
        );
        assert_eq!(exchanges[1].method, "PUT");
        assert_eq!(exchanges[1].path, "/second");
    }

    #[tokio::test]
    async fn sse_is_emitted_as_multiple_canonical_http_chunks() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 200,
            headers: BTreeMap::from([
                ("content-type".to_owned(), vec!["text/event-stream".to_owned()]),
                ("content-length".to_owned(), vec!["999".to_owned()]),
                ("transfer-encoding".to_owned(), vec!["identity".to_owned()]),
            ]),
            body: RecordedBody::Sse {
                frames: vec![
                    SseFrame {
                        event: Some("start".to_owned()),
                        data: r#"{"n":1}"#.to_owned(),
                        id: None,
                        retry: None,
                    },
                    SseFrame {
                        event: Some("stop".to_owned()),
                        data: "line one\nline two".to_owned(),
                        id: Some("event-2".to_owned()),
                        retry: Some(250),
                    },
                ],
                done: true,
            },
        }])
        .expect("scripted server should start");

        let raw = raw_http_request(
            server.addr(),
            "GET /stream HTTP/1.1\r\nHost: replay\r\nConnection: close\r\n\r\n",
        )
        .await;
        let (head, encoded_body) = raw.split_once("\r\n\r\n").expect("response should have a header block");
        assert!(head.to_ascii_lowercase().contains("transfer-encoding: chunked"));
        assert!(!head.to_ascii_lowercase().contains("content-length:"));
        assert!(
            nonempty_chunk_count(encoded_body) >= 3,
            "each SSE frame and DONE marker should be framed"
        );
        assert_eq!(
            decode_chunked(encoded_body),
            "event: start\ndata: {\"n\":1}\n\nevent: stop\ndata: line one\ndata: line two\nid: event-2\nretry: 250\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn startup_rejects_invalid_status_header_base64_and_sse_scripts() {
        let invalid = [
            RecordedResponse {
                status: 99,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
            RecordedResponse {
                status: 200,
                headers: BTreeMap::from([("bad header".to_owned(), vec!["value".to_owned()])]),
                body: RecordedBody::Empty,
            },
            RecordedResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: RecordedBody::Base64 {
                    data: "not-valid-%%%".to_owned(),
                },
            },
            RecordedResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: RecordedBody::Sse {
                    frames: vec![SseFrame {
                        event: Some("event\ninjection".to_owned()),
                        data: "{}".to_owned(),
                        id: None,
                        retry: None,
                    }],
                    done: false,
                },
            },
        ];

        for response in invalid {
            let error = ScriptedHttpServer::start(vec![response])
                .err()
                .expect("invalid script must fail before binding");
            assert!(!format!("{error}\n{error:?}").contains("not-valid-%%%"));
        }
    }

    #[test]
    fn startup_rejects_informational_and_body_bearing_no_content_responses() {
        for status in [100, 101, 199] {
            let error = ScriptedHttpServer::start(vec![RecordedResponse {
                status,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            }])
            .err()
            .expect("informational status cannot be a terminal scripted response");
            assert_eq!(error.to_string(), "scripted response status cannot be terminal");
        }

        for status in [204, 205, 304] {
            let error = ScriptedHttpServer::start(vec![RecordedResponse {
                status,
                headers: BTreeMap::new(),
                body: RecordedBody::Json {
                    value: json!({"must_not_reach_wire": true}),
                },
            }])
            .err()
            .expect("a no-content status cannot discard scripted content");
            let rendered = format!("{error}\n{error:?}");
            assert_eq!(error.to_string(), "scripted response status forbids content");
            assert!(!rendered.contains("must_not_reach_wire"));
        }
    }

    #[test]
    fn startup_accepts_empty_no_content_and_ordinary_responses() {
        for response in [
            RecordedResponse {
                status: 204,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
            RecordedResponse {
                status: 205,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
            RecordedResponse {
                status: 304,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
            RecordedResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            },
            RecordedResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: RecordedBody::Json {
                    value: json!({"ordinary": true}),
                },
            },
        ] {
            drop(ScriptedHttpServer::start(vec![response]).expect("wire-compatible response should start"));
        }
    }

    #[tokio::test]
    async fn exhaustion_returns_opaque_500_after_capturing_request() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("scripted server should start");
        let client = reqwest::Client::new();

        let first = client
            .get(format!("http://{}/one", server.addr()))
            .send()
            .await
            .unwrap();
        let exhausted = client
            .get(format!("http://{}/two", server.addr()))
            .send()
            .await
            .unwrap();

        assert_eq!(first.status(), 200);
        assert_eq!(exhausted.status(), 500);
        assert_eq!(
            exhausted.json::<serde_json::Value>().await.unwrap(),
            json!({"error": "scripted response exhausted"})
        );
        server.wait_for_exchanges(2, Duration::from_secs(1)).await.unwrap();
        let exchanges = server.take_exchanges();
        assert_eq!(
            exchanges
                .iter()
                .map(|request| request.path.as_str())
                .collect::<Vec<_>>(),
            ["/one", "/two"]
        );
    }

    #[tokio::test]
    async fn finish_rejects_malformed_request_without_hiding_unused_script() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("scripted server should start");
        let addr = server.addr();
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/malformed"))
            .header("content-type", "application/json")
            .body("{not-json")
            .send()
            .await
            .expect("malformed request should receive a controlled response");
        assert_eq!(response.status(), 400);

        let error = server
            .finish(1)
            .expect_err("malformed capture must fail final accounting");

        assert_eq!(error.to_string(), "scripted backend observed a malformed request");
        assert_listener_released(addr).await;
    }

    #[tokio::test]
    async fn finish_rejects_unused_script_and_releases_listener() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("scripted server should start");
        let addr = server.addr();

        let error = server.finish(0).expect_err("unused script must fail final accounting");

        assert_eq!(error.to_string(), "scripted backend did not consume every response");
        assert_listener_released(addr).await;
    }

    #[tokio::test]
    async fn finish_rejects_late_extra_request_after_stopping_server() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("scripted server should start");
        let addr = server.addr();
        let client = reqwest::Client::new();
        assert_eq!(
            client
                .get(format!("http://{addr}/expected"))
                .send()
                .await
                .unwrap()
                .status(),
            204
        );
        assert_eq!(
            client.get(format!("http://{addr}/late")).send().await.unwrap().status(),
            500
        );

        let error = server
            .finish(1)
            .expect_err("late extra request must fail final accounting");

        assert_eq!(
            error.to_string(),
            "scripted backend captured an unexpected request count"
        );
        assert_listener_released(addr).await;
    }

    #[tokio::test]
    async fn ending_proxy_readiness_preserves_a_scenario_get_root() {
        let server = ScriptedHttpServer::start_for_proxy(vec![Arc::new(RecordedResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: RecordedBody::Json {
                value: json!({"scenario": true}),
            },
        })])
        .expect("scripted server should start");
        server.finish_proxy_readiness();

        let response = reqwest::get(format!("http://{}/", server.addr()))
            .await
            .expect("scenario root request should complete");

        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json::<serde_json::Value>().await.unwrap(),
            json!({"scenario": true})
        );
        server.wait_for_exchanges(1, Duration::from_secs(1)).await.unwrap();
        assert_eq!(server.take_exchanges()[0].path, "/");
    }

    #[tokio::test]
    async fn exchange_wait_is_race_free_and_times_out_boundedly() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("scripted server should start");
        let client = reqwest::Client::new();
        let send = client.get(format!("http://{}/race", server.addr())).send();
        let wait = server.wait_for_exchanges(1, Duration::from_secs(1));

        let (response, waited) = tokio::join!(send, wait);

        assert_eq!(response.unwrap().status(), 200);
        waited.expect("waiter must not miss a concurrent capture notification");
        server
            .wait_for_exchanges(1, Duration::from_millis(10))
            .await
            .expect("already-reached counts should return immediately");
        let error = server
            .wait_for_exchanges(2, Duration::from_millis(20))
            .await
            .expect_err("unreached count should time out");
        assert_eq!(
            error.to_string(),
            "scripted backend timed out waiting for exchange count"
        );
    }

    #[tokio::test]
    async fn exchange_wait_observes_capture_between_false_check_and_await() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("scripted server should start");
        let release_capture = Arc::new(Barrier::new(2));
        let capture_complete = Arc::new(Barrier::new(2));
        let sender_release = Arc::clone(&release_capture);
        let sender_complete = Arc::clone(&capture_complete);
        let addr = server.addr();
        let sender = std::thread::spawn(move || {
            sender_release.wait();
            let mut stream = TcpStream::connect(addr).expect("barrier client should connect");
            stream
                .write_all(b"GET /between HTTP/1.1\r\nHost: replay\r\nConnection: close\r\n\r\n")
                .expect("barrier request should write");
            let mut response = Vec::new();
            stream.read_to_end(&mut response).expect("barrier response should read");
            sender_complete.wait();
        });

        server
            .wait_for_exchanges_after_check(1, Duration::from_secs(1), || {
                release_capture.wait();
                capture_complete.wait();
            })
            .await
            .expect("enabled notification must retain the in-between wakeup");

        sender.join().expect("barrier client should finish");
        assert_eq!(server.take_exchanges()[0].path, "/between");
    }

    #[tokio::test]
    async fn dropping_guard_stops_server_and_releases_port() {
        let server = ScriptedHttpServer::start(Vec::new()).expect("scripted server should start");
        let addr = server.addr();

        drop(server);

        let rebound = TcpListener::bind(addr).expect("drop should close the listener before returning");
        assert_eq!(rebound.local_addr().unwrap(), addr);
    }

    #[tokio::test]
    async fn thread_spawn_failure_inside_runtime_returns_opaque_error_without_panicking() {
        // Catches creating a nested runtime before OS-thread launch, whose failed-launch drop panics in async context.
        let handler: RecordingHandler =
            Arc::new(|_request| Box::pin(async { opaque_empty_response(StatusCode::NO_CONTENT) }));

        let result = DedicatedHttp1Server::start_with_options(
            handler,
            DedicatedStartOptions {
                kind: ServerKind::Replay,
                startup_mode: StartupMode::Normal,
                completion_mode: CompletionMode::Normal,
                runtime_teardown_mode: RuntimeTeardownMode::Normal,
                thread_spawn_mode: ThreadSpawnMode::Fail,
                connection_finished: None,
                stop_attempted: None,
                completion_timeout: Duration::from_secs(1),
                connection_timeouts: CONNECTION_TIMEOUTS,
                max_active_connections: None,
                header_read_timeout: None,
            },
        );

        let Err(error) = result else {
            panic!("injected OS-thread launch failure must return an error");
        };
        assert_eq!(error.to_string(), "scripted backend failed to spawn runtime thread");
    }

    #[test]
    fn drop_retries_one_bounded_timeout_before_releasing_owned_thread() {
        let completion_gate = Arc::new(Barrier::new(2));
        let (attempted_tx, attempted_rx) = std::sync::mpsc::sync_channel(2);
        let handler: RecordingHandler =
            Arc::new(|_request| Box::pin(async { opaque_empty_response(StatusCode::NO_CONTENT) }));
        let server = DedicatedHttp1Server::start_with_options(
            handler,
            DedicatedStartOptions {
                kind: ServerKind::Replay,
                startup_mode: StartupMode::Normal,
                completion_mode: CompletionMode::Hold(Arc::clone(&completion_gate)),
                runtime_teardown_mode: RuntimeTeardownMode::Normal,
                thread_spawn_mode: ThreadSpawnMode::Normal,
                connection_finished: None,
                stop_attempted: Some(attempted_tx),
                completion_timeout: Duration::from_millis(50),
                connection_timeouts: CONNECTION_TIMEOUTS,
                max_active_connections: None,
                header_read_timeout: None,
            },
        )
        .expect("dedicated server should start");
        let addr = server.addr();
        let (dropped_tx, dropped_rx) = std::sync::mpsc::sync_channel(1);
        let dropper = std::thread::spawn(move || {
            drop(server);
            let _sent = dropped_tx.send(());
        });
        attempted_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("Drop should make its first bounded stop attempt");

        let second_attempt = attempted_rx.recv_timeout(Duration::from_secs(1));
        completion_gate.wait();
        dropped_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("released Drop cleanup should return");
        dropper.join().expect("Drop cleanup thread should join");
        second_attempt.expect("Drop should retry once before detaching its owned runtime thread");
        let rebound = TcpListener::bind(addr).expect("successful Drop retry must release the listener");
        assert_eq!(rebound.local_addr().unwrap(), addr);
    }

    #[test]
    fn completion_timeout_retains_thread_ownership_for_retry_and_join() {
        let completion_gate = Arc::new(Barrier::new(2));
        let handler: RecordingHandler =
            Arc::new(|_request| Box::pin(async { opaque_empty_response(StatusCode::NO_CONTENT) }));
        let mut server = DedicatedHttp1Server::start_with_options(
            handler,
            DedicatedStartOptions {
                kind: ServerKind::Replay,
                startup_mode: StartupMode::Normal,
                completion_mode: CompletionMode::Hold(Arc::clone(&completion_gate)),
                runtime_teardown_mode: RuntimeTeardownMode::Normal,
                thread_spawn_mode: ThreadSpawnMode::Normal,
                connection_finished: None,
                stop_attempted: None,
                completion_timeout: Duration::ZERO,
                connection_timeouts: CONNECTION_TIMEOUTS,
                max_active_connections: None,
                header_read_timeout: None,
            },
        )
        .expect("dedicated server should start");
        let addr = server.addr();

        let error = server.stop().expect_err("held completion must time out");
        assert_eq!(error.to_string(), "scripted backend shutdown timed out");
        assert!(server.thread.is_some(), "timed-out stop must retain the JoinHandle");

        completion_gate.wait();
        server.completion_timeout = Duration::from_secs(1);
        server.stop().expect("retry should receive completion and join");
        assert!(
            server.thread.is_none(),
            "successful retry must consume the joined thread"
        );
        let rebound = TcpListener::bind(addr).expect("joined retry must release the listener");
        assert_eq!(rebound.local_addr().unwrap(), addr);
    }

    #[test]
    fn completion_timeout_covers_runtime_drop_before_join_and_retry_releases_port() {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let release = Arc::new(Barrier::new(2));
        let handler: RecordingHandler =
            Arc::new(|_request| Box::pin(async { opaque_empty_response(StatusCode::NO_CONTENT) }));
        let server = DedicatedHttp1Server::start_with_options(
            handler,
            DedicatedStartOptions {
                kind: ServerKind::Replay,
                startup_mode: StartupMode::Normal,
                completion_mode: CompletionMode::Normal,
                runtime_teardown_mode: RuntimeTeardownMode::Hold {
                    entered: entered_tx,
                    release: Arc::clone(&release),
                },
                thread_spawn_mode: ThreadSpawnMode::Normal,
                connection_finished: None,
                stop_attempted: None,
                completion_timeout: Duration::from_millis(50),
                connection_timeouts: CONNECTION_TIMEOUTS,
                max_active_connections: None,
                header_read_timeout: None,
            },
        )
        .expect("dedicated server should start");
        let addr = server.addr();
        let (stopped_tx, stopped_rx) = std::sync::mpsc::sync_channel(1);
        let stopper = std::thread::spawn(move || {
            let mut server = server;
            let result = server.stop();
            let _sent = stopped_tx.send((result, server));
        });
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("blocking runtime task should enter before teardown");

        let first_stop = stopped_rx.recv_timeout(Duration::from_secs(1));
        release.wait();
        let (result, mut server) = match first_stop {
            Ok(outcome) => outcome,
            Err(error) => {
                let _cleanup = stopped_rx
                    .recv_timeout(Duration::from_secs(1))
                    .expect("released legacy join should return for cleanup");
                stopper.join().expect("stopper thread should finish after cleanup");
                panic!("stop blocked in join instead of returning its bounded timeout: {error}");
            },
        };
        stopper.join().expect("bounded stopper thread should finish");
        assert_eq!(result.unwrap_err().to_string(), "scripted backend shutdown timed out");
        assert!(server.thread.is_some(), "timeout must retain runtime-thread ownership");

        server.completion_timeout = Duration::from_secs(1);
        server
            .stop()
            .expect("retry should observe post-runtime-drop completion");
        let rebound = TcpListener::bind(addr).expect("joined retry must release the listener");
        assert_eq!(rebound.local_addr().unwrap(), addr);
    }

    #[test]
    fn stuck_connection_is_aborted_joined_and_releases_port_with_bounded_failure() {
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let handler: RecordingHandler = Arc::new(move |_request| {
            let entered_tx = entered_tx.clone();
            Box::pin(async move {
                let _sent = entered_tx.send(());
                std::future::pending::<http::Response<RecordingBody>>().await
            })
        });
        let mut server = DedicatedHttp1Server::start_with_options(
            handler,
            DedicatedStartOptions {
                kind: ServerKind::Replay,
                startup_mode: StartupMode::Normal,
                completion_mode: CompletionMode::Normal,
                runtime_teardown_mode: RuntimeTeardownMode::Normal,
                thread_spawn_mode: ThreadSpawnMode::Normal,
                connection_finished: None,
                stop_attempted: None,
                completion_timeout: Duration::from_secs(1),
                connection_timeouts: ConnectionTimeouts {
                    graceful: Duration::ZERO,
                    aborted: Duration::from_secs(1),
                },
                max_active_connections: None,
                header_read_timeout: None,
            },
        )
        .expect("dedicated server should start");
        let addr = server.addr();
        let mut client = TcpStream::connect(addr).expect("raw client should connect");
        client
            .write_all(b"GET /stuck HTTP/1.1\r\nHost: replay\r\n\r\n")
            .expect("raw request should write");
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("handler should reach its deterministic gate");

        let error = server
            .stop()
            .expect_err("stuck graceful drain must fail after aborting");
        assert_eq!(error.to_string(), "scripted backend connection drain was incomplete");
        assert!(
            server.thread.is_none(),
            "failed drain must still join its runtime thread"
        );
        drop(client);
        let rebound = TcpListener::bind(addr).expect("aborted server must release the listener");
        assert_eq!(rebound.local_addr().unwrap(), addr);
    }

    #[test]
    fn post_bind_startup_failure_joins_thread_and_releases_port_before_returning() {
        let acquired_addr = Arc::new(Mutex::new(None));
        let thread_exited = Arc::new(AtomicBool::new(false));

        let result = ScriptedHttpServer::start_with_injected_post_bind_failure(
            Arc::clone(&acquired_addr),
            Arc::clone(&thread_exited),
        );

        assert!(result.is_err(), "injected startup failure must be returned");
        assert!(thread_exited.load(Ordering::SeqCst), "runtime thread must be joined");
        let addr = acquired_addr
            .lock()
            .unwrap()
            .expect("injected failure must publish its acquired address");
        let rebound = TcpListener::bind(addr).expect("failed startup must release its listener before returning");
        assert_eq!(rebound.local_addr().unwrap(), addr);
    }

    async fn raw_http_request(addr: std::net::SocketAddr, request: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("raw client should connect");
        stream
            .write_all(request.as_bytes())
            .await
            .expect("raw request should write");
        let mut bytes = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), stream.read_to_end(&mut bytes))
            .await
            .expect("raw response should complete before timeout")
            .expect("raw response should read");
        String::from_utf8(bytes).expect("test response should be UTF-8")
    }

    fn nonempty_chunk_count(mut body: &str) -> usize {
        let mut count = 0;
        while let Some((size, rest)) = body.split_once("\r\n") {
            let size = usize::from_str_radix(size.trim(), 16).expect("chunk size should be hexadecimal");
            if size == 0 {
                break;
            }
            count += 1;
            body = rest.get(size + 2..).expect("chunk should contain bytes and CRLF");
        }
        count
    }

    fn decode_chunked(mut body: &str) -> String {
        let mut decoded = String::new();
        while let Some((size, rest)) = body.split_once("\r\n") {
            let size = usize::from_str_radix(size.trim(), 16).expect("chunk size should be hexadecimal");
            if size == 0 {
                break;
            }
            decoded.push_str(rest.get(..size).expect("chunk should contain declared bytes"));
            body = rest.get(size + 2..).expect("chunk should end with CRLF");
        }
        decoded
    }
}
