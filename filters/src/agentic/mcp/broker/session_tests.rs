// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for current-profile backend session routing, driven through a
//! scripted [`McpBackendClient`] mock.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderValue;
use praxis_filter::{FilterAction, HttpFilter as _};

use super::{
    BackendSessions, McpBrokerFilter,
    backend::{BackendCallout, BackendInit, BackendTarget, McpBackendClient, McpBackendError},
    config::{McpBrokerConfig, build_config},
};

// -----------------------------------------------------------------------------
// Mock Client
// -----------------------------------------------------------------------------

/// Scripted backend client recording every lifecycle exchange.
#[derive(Default)]
struct MockBackend {
    /// Recorded `(exchange, cluster, detail)` tuples.
    calls: Mutex<Vec<(String, String, String)>>,
    /// Sessions minted so far.
    minted: AtomicUsize,
    /// Error returned by `initialize`, when set.
    initialize_error: Mutex<Option<McpBackendError>>,
    /// Error returned by `send_initialized`, when set.
    initialized_error: Mutex<Option<McpBackendError>>,
    /// Delay applied to `initialize`.
    initialize_delay: Option<std::time::Duration>,
    /// Mint no session id (stateless backend).
    sessionless: bool,
}

impl MockBackend {
    fn count(&self, exchange: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(e, ..)| e == exchange)
            .count()
    }

    fn record(&self, exchange: &str, target: BackendTarget<'_>, detail: String) {
        self.calls
            .lock()
            .unwrap()
            .push((exchange.to_owned(), target.cluster.to_string(), detail));
    }
}

#[async_trait]
impl McpBackendClient for MockBackend {
    async fn initialize(
        &self,
        target: BackendTarget<'_>,
        protocol_version: &str,
        callout: &BackendCallout,
    ) -> Result<BackendInit, McpBackendError> {
        let auth = callout
            .forwarded_headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        self.record("initialize", target, format!("{protocol_version}|{auth}"));
        if let Some(delay) = self.initialize_delay {
            tokio::time::sleep(delay).await;
        }
        let scripted = self.initialize_error.lock().unwrap().clone();
        if let Some(error) = scripted {
            return Err(error);
        }
        let n = self.minted.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(BackendInit {
            protocol_version: Box::from("2025-06-18"),
            session_id: (!self.sessionless).then(|| HeaderValue::from_str(&format!("backend-{n}")).unwrap()),
            endpoint: Some(Arc::from(format!("10.0.0.{n}:3001"))),
        })
    }

    async fn send_initialized(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        _callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        self.record("initialized", target, session_label(session));
        self.initialized_error.lock().unwrap().clone().map_or(Ok(()), Err)
    }

    async fn terminate(
        &self,
        target: BackendTarget<'_>,
        session: &BackendInit,
        _callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        self.record("terminate", target, session_label(session));
        Ok(())
    }
}

fn session_label(session: &BackendInit) -> String {
    session
        .session_id
        .as_ref()
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

const SESSION_BROKER_YAML: &str = r#"
backend_sessions:
  max_sessions: 16
servers:
  - name: weather
    cluster: weather-mcp
    path: /weather/mcp
    tool_prefix: "weather_"
    tools:
      - name: get_weather
  - name: calendar
    cluster: calendar-mcp
    path: /mcp
    tool_prefix: "cal_"
    tools:
      - name: create_event
"#;

fn broker_with(mock: Arc<MockBackend>) -> McpBrokerFilter {
    let cfg: McpBrokerConfig = serde_yaml::from_str(SESSION_BROKER_YAML).unwrap();
    let servers = cfg.servers.clone();
    let (validated, catalog) = build_config(cfg).unwrap();
    let settings = validated.backend_sessions.clone().unwrap();
    let backend = BackendSessions::new(mock, Some(crate::isolated_subrequest_client(1)), &settings);
    McpBrokerFilter::build(&validated, catalog, &servers, Some(backend))
}

fn call_request(session: Option<&str>, auth: Option<&str>) -> praxis_filter::Request {
    let mut req = crate::test_utils::make_request(http::Method::POST, "/mcp");
    req.headers.insert("content-type", "application/json".parse().unwrap());
    req.headers
        .insert("mcp-protocol-version", "2025-03-26".parse().unwrap());
    if let Some(session) = session {
        req.headers.insert("mcp-session-id", session.parse().unwrap());
    }
    if let Some(auth) = auth {
        req.headers.insert("authorization", auth.parse().unwrap());
    }
    req
}

fn call_body(tool: &str) -> Bytes {
    Bytes::from(format!(
        r#"{{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{{"name":"{tool}","arguments":{{"city":"Paris"}}}}}}"#
    ))
}

/// Outcome of one routed call.
struct Routed {
    action: FilterAction,
    body: Option<Bytes>,
    cluster: Option<String>,
    path: Option<String>,
    set_headers: Vec<(String, String)>,
    removed_headers: Vec<String>,
    pinned_endpoint: Option<String>,
}

async fn route(filter: &McpBrokerFilter, req: &praxis_filter::Request, tool: &str) -> Routed {
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.current_filter_id = Some(0);
    let mut body = Some(call_body(tool));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    // Core runs a pre-read body before the header phase and rebuilds
    // `pinned_endpoint_address` empty for it; emulate both.
    ctx.pinned_endpoint_address = None;
    if matches!(action, FilterAction::Release) {
        let header_action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(header_action, FilterAction::Continue),
            "header phase must continue"
        );
    }
    Routed {
        action,
        body,
        cluster: ctx.cluster.as_deref().map(str::to_owned),
        path: ctx.rewritten_path.clone(),
        set_headers: ctx
            .request_headers_to_set
            .iter()
            .map(|(n, v)| (n.as_str().to_owned(), v.to_str().unwrap().to_owned()))
            .collect(),
        removed_headers: ctx
            .request_headers_to_remove
            .iter()
            .map(|n| n.as_str().to_owned())
            .collect(),
        pinned_endpoint: ctx.pinned_endpoint_address.as_deref().map(str::to_owned),
    }
}

/// Route a call, then deliver an upstream response with `status`.
async fn route_with_response(
    filter: &McpBrokerFilter,
    req: &praxis_filter::Request,
    tool: &str,
    status: http::StatusCode,
) {
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.current_filter_id = Some(0);
    let mut body = Some(call_body(tool));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let mut response = crate::test_utils::make_response();
    response.status = status;
    ctx.response_header = Some(&mut response);
    let response_action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Release), "call must be forwarded");
    assert!(
        matches!(response_action, FilterAction::Continue),
        "response must pass through"
    );
}

fn header<'a>(routed: &'a Routed, name: &str) -> Option<&'a str> {
    routed
        .set_headers
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

fn rejection_json(action: &FilterAction) -> (u16, serde_json::Value) {
    match action {
        FilterAction::Reject(rejection) => (
            rejection.status,
            serde_json::from_slice(rejection.body.as_ref().unwrap()).unwrap(),
        ),
        _ => panic!("expected a rejection"),
    }
}

// -----------------------------------------------------------------------------
// Handshake and Routing
// -----------------------------------------------------------------------------

#[tokio::test]
async fn missing_backend_session_runs_handshake_then_forwards() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));

    let routed = route(&filter, &call_request(Some("gw-1"), None), "weather_get_weather").await;

    assert!(matches!(routed.action, FilterAction::Release), "call must be forwarded");
    let calls = mock.calls.lock().unwrap().clone();
    assert_eq!(
        calls.iter().map(|(e, ..)| e.as_str()).collect::<Vec<_>>(),
        vec!["initialize", "initialized"],
        "handshake must run initialize then notifications/initialized"
    );
    assert_eq!(calls[0].1, "weather-mcp", "handshake must target the server cluster");
    assert_eq!(
        calls[1].2, "backend-1",
        "initialized must carry the new backend session"
    );
    assert_eq!(
        routed.cluster.as_deref(),
        Some("weather-mcp"),
        "call must route to the cluster"
    );
    assert_eq!(
        routed.path.as_deref(),
        Some("/weather/mcp"),
        "call must use the backend path"
    );
    assert_eq!(
        header(&routed, "mcp-session-id"),
        Some("backend-1"),
        "gateway session must be swapped for the backend session"
    );
    assert_eq!(
        header(&routed, "mcp-protocol-version"),
        Some("2025-06-18"),
        "negotiated backend version must be forwarded"
    );
    let forwarded: serde_json::Value = serde_json::from_slice(routed.body.as_ref().unwrap()).unwrap();
    assert_eq!(
        forwarded["params"]["name"], "get_weather",
        "tool prefix must be stripped"
    );
    assert_eq!(
        forwarded["params"]["arguments"]["city"], "Paris",
        "arguments must be preserved"
    );
}

#[tokio::test]
async fn existing_backend_session_is_reused() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    route(&filter, &req, "weather_get_weather").await;
    let second = route(&filter, &req, "weather_get_weather").await;

    assert_eq!(mock.count("initialize"), 1, "an existing session must not re-handshake");
    assert_eq!(
        header(&second, "mcp-session-id"),
        Some("backend-1"),
        "session must be reused"
    );
}

#[tokio::test]
async fn sessions_are_scoped_per_gateway_session_and_server() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));

    let a = route(&filter, &call_request(Some("gw-a"), None), "weather_get_weather").await;
    let b = route(&filter, &call_request(Some("gw-b"), None), "weather_get_weather").await;
    let c = route(&filter, &call_request(Some("gw-a"), None), "cal_create_event").await;

    assert_eq!(
        mock.count("initialize"),
        3,
        "each (gateway, server) pair needs its own handshake"
    );
    assert_ne!(
        header(&a, "mcp-session-id"),
        header(&b, "mcp-session-id"),
        "gateways must not share"
    );
    assert_ne!(
        header(&a, "mcp-session-id"),
        header(&c, "mcp-session-id"),
        "servers must not share"
    );
    assert_eq!(
        c.cluster.as_deref(),
        Some("calendar-mcp"),
        "calendar call must route to calendar"
    );
}

#[tokio::test]
async fn backend_404_invalidates_and_next_call_reinitializes() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    route_with_response(&filter, &req, "weather_get_weather", http::StatusCode::NOT_FOUND).await;
    let next = route(&filter, &req, "weather_get_weather").await;

    assert_eq!(
        mock.count("initialize"),
        2,
        "a backend 404 must force a fresh handshake"
    );
    assert_eq!(
        header(&next, "mcp-session-id"),
        Some("backend-2"),
        "the new session must be used"
    );
}

#[tokio::test]
async fn forwarded_calls_are_pinned_to_the_session_endpoint() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    let first = route(&filter, &req, "weather_get_weather").await;
    let second = route(&filter, &req, "weather_get_weather").await;

    assert_eq!(
        first.pinned_endpoint.as_deref(),
        Some("10.0.0.1:3001"),
        "the call must reach the endpoint that created the backend session"
    );
    assert_eq!(
        second.pinned_endpoint, first.pinned_endpoint,
        "reused sessions must stay on the same endpoint"
    );
}

#[tokio::test]
async fn backend_502_invalidates_and_next_call_reinitializes() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    route_with_response(&filter, &req, "weather_get_weather", http::StatusCode::BAD_GATEWAY).await;
    let next = route(&filter, &req, "weather_get_weather").await;

    assert_eq!(
        mock.count("initialize"),
        2,
        "an unreachable pinned endpoint must force a fresh handshake"
    );
    assert_eq!(
        next.pinned_endpoint.as_deref(),
        Some("10.0.0.2:3001"),
        "the new session's endpoint must be pinned"
    );
}

#[tokio::test]
async fn successful_response_keeps_mapping() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    route_with_response(&filter, &req, "weather_get_weather", http::StatusCode::OK).await;
    route(&filter, &req, "weather_get_weather").await;

    assert_eq!(mock.count("initialize"), 1, "a 200 must not invalidate the session");
}

#[tokio::test]
async fn credential_change_rebinds_backend_session() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));

    let first = route(
        &filter,
        &call_request(Some("gw-1"), Some("Bearer alice")),
        "weather_get_weather",
    )
    .await;
    let same = route(
        &filter,
        &call_request(Some("gw-1"), Some("Bearer alice")),
        "weather_get_weather",
    )
    .await;
    let other = route(
        &filter,
        &call_request(Some("gw-1"), Some("Bearer bob")),
        "weather_get_weather",
    )
    .await;

    assert_eq!(
        mock.count("initialize"),
        2,
        "a different credential must not reuse the session"
    );
    assert_eq!(
        header(&first, "mcp-session-id"),
        header(&same, "mcp-session-id"),
        "same credential reuses"
    );
    assert_ne!(
        header(&first, "mcp-session-id"),
        header(&other, "mcp-session-id"),
        "credential rebinds"
    );
    let calls = mock.calls.lock().unwrap().clone();
    assert!(
        calls[0].2.ends_with("|Bearer alice"),
        "forwarded credential must reach the handshake"
    );
}

#[tokio::test]
async fn concurrent_first_calls_share_one_handshake() {
    let mock = Arc::new(MockBackend {
        initialize_delay: Some(std::time::Duration::from_millis(50)),
        ..MockBackend::default()
    });
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    let (a, b) = tokio::join!(
        route(&filter, &req, "weather_get_weather"),
        route(&filter, &req, "weather_get_weather")
    );

    assert_eq!(
        mock.count("initialize"),
        1,
        "concurrent first calls must initialize once"
    );
    assert_eq!(
        header(&a, "mcp-session-id"),
        header(&b, "mcp-session-id"),
        "both calls share the session"
    );
}

#[tokio::test]
async fn sessionless_backend_strips_gateway_session_header() {
    let mock = Arc::new(MockBackend {
        sessionless: true,
        ..MockBackend::default()
    });
    let filter = broker_with(Arc::clone(&mock));

    let routed = route(&filter, &call_request(Some("gw-1"), None), "weather_get_weather").await;

    assert!(matches!(routed.action, FilterAction::Release), "call must be forwarded");
    assert!(
        header(&routed, "mcp-session-id").is_none(),
        "no backend session to send"
    );
    assert!(
        routed.removed_headers.iter().any(|n| n == "mcp-session-id"),
        "the gateway session id must never reach the backend"
    );
    assert!(
        routed.pinned_endpoint.is_none(),
        "a sessionless backend holds no endpoint state, so the load balancer chooses freely"
    );
}

#[tokio::test]
async fn client_protocol_version_is_offered_to_backend() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));

    route(&filter, &call_request(Some("gw-1"), None), "weather_get_weather").await;
    let mut req = call_request(Some("gw-2"), None);
    req.headers
        .insert("mcp-protocol-version", "1999-01-01".parse().unwrap());
    route(&filter, &req, "weather_get_weather").await;

    let calls = mock.calls.lock().unwrap().clone();
    assert!(
        calls[0].2.starts_with("2025-03-26|"),
        "supported client version must be offered"
    );
    assert!(
        calls[2].2.starts_with(&format!("{}|", filter.default_version)),
        "unsupported client version must fall back to the default: {}",
        calls[2].2
    );
}

// -----------------------------------------------------------------------------
// Failures
// -----------------------------------------------------------------------------

#[tokio::test]
async fn missing_gateway_session_is_invalid_request() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));

    let routed = route(&filter, &call_request(None, None), "weather_get_weather").await;

    let (status, body) = rejection_json(&routed.action);
    assert_eq!(status, 400, "missing session must be a 400");
    assert_eq!(body["error"]["code"], -32600, "missing session must be -32600");
    assert_eq!(mock.count("initialize"), 0, "no handshake without a gateway session");
}

#[tokio::test]
async fn malformed_gateway_session_is_invalid_request() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    let long = "a".repeat(257);

    let routed = route(&filter, &call_request(Some(&long), None), "weather_get_weather").await;

    let (status, _) = rejection_json(&routed.action);
    assert_eq!(status, 400, "oversized session id must be rejected");
}

#[tokio::test]
async fn unknown_tool_is_invalid_params() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));

    let routed = route(&filter, &call_request(Some("gw-1"), None), "nope").await;

    let (_, body) = rejection_json(&routed.action);
    assert_eq!(body["error"]["code"], -32602, "unknown tool must be -32602");
    assert_eq!(mock.count("initialize"), 0, "no handshake for an unknown tool");
}

#[tokio::test]
async fn handshake_failure_is_structured_and_not_cached() {
    let mock = Arc::new(MockBackend::default());
    *mock.initialize_error.lock().unwrap() = Some(McpBackendError::ResponseTooLarge { limit: 1024 });
    let filter = broker_with(Arc::clone(&mock));
    let req = call_request(Some("gw-1"), None);

    let routed = route(&filter, &req, "weather_get_weather").await;

    let (status, body) = rejection_json(&routed.action);
    assert_eq!(status, 200, "current-profile errors are JSON-RPC over 200");
    assert_eq!(body["id"], 9, "request id must be echoed");
    assert_eq!(body["error"]["code"], -32603, "handshake failure must be -32603");
    assert_eq!(body["error"]["data"]["server"], "weather", "server must be reported");
    assert_eq!(
        body["error"]["data"]["reason"], "response_too_large",
        "failure class must be preserved"
    );

    *mock.initialize_error.lock().unwrap() = None;
    let retried = route(&filter, &req, "weather_get_weather").await;
    assert!(
        matches!(retried.action, FilterAction::Release),
        "a failed handshake must not be cached"
    );
}

#[tokio::test]
async fn rejected_handshake_reports_backend_status() {
    let mock = Arc::new(MockBackend::default());
    *mock.initialize_error.lock().unwrap() = Some(McpBackendError::Rejected { status: 401 });
    let filter = broker_with(Arc::clone(&mock));

    let routed = route(&filter, &call_request(Some("gw-1"), None), "weather_get_weather").await;

    let (_, body) = rejection_json(&routed.action);
    assert_eq!(
        body["error"]["data"]["reason"], "rejected",
        "rejection must be classified"
    );
    assert_eq!(body["error"]["data"]["status"], 401, "backend status must be reported");
}

#[tokio::test]
async fn initialized_failure_terminates_new_session() {
    let mock = Arc::new(MockBackend::default());
    *mock.initialized_error.lock().unwrap() = Some(McpBackendError::Timeout);
    let filter = broker_with(Arc::clone(&mock));

    let routed = route(&filter, &call_request(Some("gw-1"), None), "weather_get_weather").await;

    let (_, body) = rejection_json(&routed.action);
    assert_eq!(body["error"]["data"]["reason"], "timeout", "timeout must be classified");
    assert_eq!(
        mock.count("terminate"),
        1,
        "a half-initialized session must be terminated"
    );
}

#[tokio::test]
async fn delete_terminates_backend_sessions_of_that_gateway_session() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    route(&filter, &call_request(Some("gw-1"), None), "weather_get_weather").await;
    route(&filter, &call_request(Some("gw-1"), None), "cal_create_event").await;
    route(&filter, &call_request(Some("gw-2"), None), "weather_get_weather").await;

    let mut req = crate::test_utils::make_request(http::Method::DELETE, "/mcp");
    req.headers.insert("mcp-session-id", "gw-1".parse().unwrap());
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 204),
        "DELETE must still return 204"
    );
    assert_eq!(
        mock.count("terminate"),
        2,
        "both backend sessions of gw-1 must be terminated"
    );
    route(&filter, &call_request(Some("gw-2"), None), "weather_get_weather").await;
    assert_eq!(mock.count("initialize"), 3, "other gateway sessions must survive");
}

#[tokio::test]
async fn delete_with_other_credential_keeps_backend_sessions() {
    let mock = Arc::new(MockBackend::default());
    let filter = broker_with(Arc::clone(&mock));
    route(
        &filter,
        &call_request(Some("gw-1"), Some("Bearer alice")),
        "weather_get_weather",
    )
    .await;

    let mut req = crate::test_utils::make_request(http::Method::DELETE, "/mcp");
    req.headers.insert("mcp-session-id", "gw-1".parse().unwrap());
    req.headers.insert("authorization", "Bearer mallory".parse().unwrap());
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 204),
        "DELETE must still return 204"
    );

    assert_eq!(
        mock.count("terminate"),
        0,
        "another caller must not terminate the session"
    );
    route(
        &filter,
        &call_request(Some("gw-1"), Some("Bearer alice")),
        "weather_get_weather",
    )
    .await;
    assert_eq!(mock.count("initialize"), 1, "the owner's session must survive");
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

#[test]
fn backend_sessions_require_current_profile() {
    let yaml = format!("protocol_profile: stateless\n{SESSION_BROKER_YAML}");
    let cfg: McpBrokerConfig = serde_yaml::from_str(&yaml).unwrap();
    let err = build_config(cfg).unwrap_err();
    assert!(err.to_string().contains("requires protocol_profile 'current'"), "{err}");
}

#[test]
fn backend_sessions_without_binding_fail() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(SESSION_BROKER_YAML).unwrap();
    let err = McpBrokerFilter::from_config(&yaml).err().expect("must fail");
    assert!(err.to_string().contains("chain-binding"), "{err}");
}

#[test]
fn backend_sessions_reject_reserved_forward_headers() {
    for name in [
        "mcp-session-id",
        "host",
        "x-praxis-secret",
        "connection",
        "content-length",
    ] {
        let yaml = format!("backend_sessions:\n  forward_headers: [{name}]\nservers: []\n");
        let cfg: McpBrokerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(build_config(cfg).is_err(), "'{name}' must not be forwardable");
    }
}

#[test]
fn backend_sessions_reject_out_of_range_limits() {
    for field in [
        "max_sessions: 0",
        "idle_timeout_ms: 0",
        "handshake_timeout_ms: 0",
        "max_response_bytes: 0",
        "max_response_bytes: 999999999",
    ] {
        let yaml = format!("backend_sessions:\n  {field}\nservers: []\n");
        let cfg: McpBrokerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(build_config(cfg).is_err(), "'{field}' must be rejected");
    }
}

#[test]
fn backend_sessions_reject_operator_load_balancer() {
    let yaml = "backend_sessions:\n  outbound_filters:\n    - filter: load_balancer\n      clusters: []\nservers: []\n";
    let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
    let err = build_config(cfg).unwrap_err();
    assert!(err.to_string().contains("load_balancer"), "{err}");
}

/// Build a one-filter pipeline the way the server does, with optional
/// top-level clusters.
fn build_server_pipeline(
    filter_yaml: &str,
    clusters_yaml: &str,
) -> Result<praxis_filter::FilterPipeline, praxis_filter::FilterError> {
    let registry = crate::build_ai_registry();
    let mut entries: Vec<praxis_core::config::FilterEntry> = serde_yaml::from_str(filter_yaml).unwrap();
    let clusters: Vec<praxis_core::config::Cluster> = serde_yaml::from_str(clusters_yaml).unwrap();
    let insecure: praxis_core::config::InsecureOptions = serde_yaml::from_str("allow_private_endpoints: true").unwrap();
    praxis_filter::FilterPipeline::build_with_chains(
        &mut entries,
        &registry,
        &std::collections::HashMap::new(),
        &insecure,
        &clusters,
    )
}

/// `SESSION_BROKER_YAML` as a single-entry filter list.
fn session_broker_entry() -> String {
    let indented: String = SESSION_BROKER_YAML
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| format!("  {line}\n"))
        .collect();
    format!("- filter: mcp\n{indented}")
}

#[test]
fn handshake_chain_binds_against_top_level_clusters() {
    let clusters = r#"
- name: weather-mcp
  endpoints: ["127.0.0.1:3001"]
- name: calendar-mcp
  endpoints: ["127.0.0.1:3002"]
"#;
    assert!(
        build_server_pipeline(&session_broker_entry(), clusters).is_ok(),
        "handshake chain must bind when clusters are declared at top level"
    );
}

#[test]
fn handshake_chain_requires_top_level_clusters() {
    let err = build_server_pipeline(&session_broker_entry(), "[]")
        .err()
        .expect("undeclared clusters must fail");
    assert!(
        err.to_string().contains("top-level"),
        "error must explain the cluster requirement: {err}"
    );
}

#[test]
fn handshake_chain_accepts_operator_outbound_filters() {
    let entry = session_broker_entry().replace(
        "    max_sessions: 16\n",
        "    max_sessions: 16\n    outbound_filters:\n      - filter: headers\n        request_add:\n          - name: x-broker\n            value: handshake\n",
    );
    let clusters = r#"
- name: weather-mcp
  endpoints: ["127.0.0.1:3001"]
- name: calendar-mcp
  endpoints: ["127.0.0.1:3002"]
"#;
    let result = build_server_pipeline(&entry, clusters);
    assert!(result.is_ok(), "operator filters must bind: {:?}", result.err());
}

#[test]
fn server_build_keeps_plain_broker_working() {
    let entry = "- filter: mcp\n  servers:\n    - name: a\n      cluster: c\n      tools: [{name: t}]\n";
    assert!(
        build_server_pipeline(entry, "[]").is_ok(),
        "a broker without backend_sessions needs no top-level clusters"
    );
}
