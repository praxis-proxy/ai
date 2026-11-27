// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for per-caller tool discovery, driven through a scripted
//! backend that answers the handshake and serves `tools/list` pages.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderValue;
use praxis_ai_apis::mcp_session::McpToolPage;
use praxis_filter::{FilterAction, HttpFilter as _};
use serde_json::{Value, json};

use super::{
    BackendSessions, McpBrokerFilter,
    backend::{BackendCallout, BackendInit, BackendTarget, McpBackendClient, McpBackendError},
    config::{McpBrokerConfig, build_config},
    discovery::{Discovery, ListingScope, METRIC_DISCOVERY_FAILURES_TOTAL, McpToolPages},
};

// -----------------------------------------------------------------------------
// Scripted Backend
// -----------------------------------------------------------------------------

/// Scripted listing for one cluster.
enum Script {
    /// Serve these pages in order, then the last one again.
    Pages(Vec<Vec<Value>>),
    /// Fail every page.
    Fail(McpBackendError),
}

/// One `tools/list` page request.
#[derive(Clone, Debug)]
struct PageRequest {
    cluster: String,
    authorization: String,
    cursor: Option<String>,
    /// Backend session id, or `stateless:<version>`.
    scope: String,
}

/// Backend answering the handshake and serving scripted listings.
#[derive(Default)]
struct ScriptedBackend {
    initialized: Mutex<Vec<String>>,
    pages: Mutex<Vec<PageRequest>>,
    scripts: Mutex<HashMap<String, Script>>,
    /// Pending one-off failures returned before the script, by cluster.
    once: Mutex<HashMap<String, McpBackendError>>,
}

impl ScriptedBackend {
    fn serve(&self, cluster: &str, tools: &[&str]) {
        let page = tools
            .iter()
            .map(|name| json!({"name": name, "description": format!("{name} from {cluster}"), "inputSchema": {"type": "object"}}))
            .collect();
        self.scripts
            .lock()
            .unwrap()
            .insert(cluster.to_owned(), Script::Pages(vec![page]));
    }

    fn serve_pages(&self, cluster: &str, pages: Vec<Vec<Value>>) {
        self.scripts
            .lock()
            .unwrap()
            .insert(cluster.to_owned(), Script::Pages(pages));
    }

    fn fail(&self, cluster: &str, error: McpBackendError) {
        self.scripts
            .lock()
            .unwrap()
            .insert(cluster.to_owned(), Script::Fail(error));
    }

    fn page_requests(&self, cluster: &str) -> Vec<PageRequest> {
        self.pages
            .lock()
            .unwrap()
            .iter()
            .filter(|page| page.cluster == cluster)
            .cloned()
            .collect()
    }

    fn initialize_count(&self, cluster: &str) -> usize {
        self.initialized
            .lock()
            .unwrap()
            .iter()
            .filter(|c| *c == cluster)
            .count()
    }
}

fn authorization(callout: &BackendCallout) -> String {
    callout
        .forwarded_headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

#[async_trait]
impl McpBackendClient for ScriptedBackend {
    async fn initialize(
        &self,
        target: BackendTarget<'_>,
        protocol_version: &str,
        _callout: &BackendCallout,
    ) -> Result<BackendInit, McpBackendError> {
        let mut initialized = self.initialized.lock().unwrap();
        initialized.push(target.cluster.to_string());
        let n = initialized.len();
        drop(initialized);
        Ok(BackendInit {
            protocol_version: Box::from(protocol_version),
            session_id: Some(HeaderValue::from_str(&format!("{}-{n}", target.cluster)).unwrap()),
            endpoint: Some(Arc::from("10.0.0.1:3001")),
        })
    }

    async fn send_initialized(
        &self,
        _target: BackendTarget<'_>,
        _session: &BackendInit,
        _callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        Ok(())
    }

    async fn terminate(
        &self,
        _target: BackendTarget<'_>,
        _session: &BackendInit,
        _callout: &BackendCallout,
    ) -> Result<(), McpBackendError> {
        Ok(())
    }
}

#[async_trait]
impl McpToolPages for ScriptedBackend {
    async fn list_tools_page(
        &self,
        target: BackendTarget<'_>,
        scope: ListingScope<'_>,
        cursor: Option<String>,
        callout: &BackendCallout,
    ) -> Result<McpToolPage<Value>, McpBackendError> {
        let cluster = target.cluster.to_string();
        let scope = match scope {
            ListingScope::Session(init) => init.session_id.as_ref().unwrap().to_str().unwrap().to_owned(),
            ListingScope::Stateless { protocol_version } => format!("stateless:{protocol_version}"),
        };
        self.pages.lock().unwrap().push(PageRequest {
            cluster: cluster.clone(),
            authorization: authorization(callout),
            cursor: cursor.clone(),
            scope,
        });
        let once = self.once.lock().unwrap().remove(&cluster);
        if let Some(error) = once {
            return Err(error);
        }
        let scripts = self.scripts.lock().unwrap();
        match scripts.get(&cluster) {
            Some(Script::Pages(pages)) => {
                let index = cursor.map_or(0, |c| c.parse::<usize>().unwrap());
                let page = pages.get(index).cloned().unwrap_or_default();
                let next_cursor = (index + 1 < pages.len()).then(|| (index + 1).to_string());
                Ok(McpToolPage {
                    tools: page,
                    next_cursor,
                })
            },
            Some(Script::Fail(error)) => Err(error.clone()),
            None => Err(McpBackendError::Transport),
        }
    }
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

const CURRENT_YAML: &str = r#"
backend_sessions:
  max_sessions: 16
servers:
  - name: weather
    cluster: weather-mcp
    tool_prefix: "weather_"
    discover: true
    tools:
      - name: forecast
        description: operator description
      - name: admin
        hidden: true
  - name: calendar
    cluster: calendar-mcp
    tool_prefix: "cal_"
    tools:
      - name: create_event
"#;

fn current_broker(backend: &Arc<ScriptedBackend>, yaml: &str) -> McpBrokerFilter {
    let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
    let servers = cfg.servers.clone();
    let (validated, catalog) = build_config(cfg).unwrap();
    let settings = validated.backend_sessions.clone().unwrap();
    let sessions = BackendSessions::new(
        Arc::<ScriptedBackend>::clone(backend),
        Some(crate::isolated_subrequest_client(1)),
        &settings,
        validated.supported_versions.clone(),
    )
    .unwrap();
    let discovery = validated.discovery.as_ref().map(|settings| {
        Discovery::new(
            settings,
            Arc::<ScriptedBackend>::clone(backend),
            Some(crate::isolated_subrequest_client(1)),
        )
    });
    McpBrokerFilter::build(&validated, catalog, &servers, Some(sessions), discovery)
}

fn stateless_broker(backend: &Arc<ScriptedBackend>) -> McpBrokerFilter {
    let yaml = r#"
protocol_profile: stateless
servers:
  - name: weather
    cluster: weather-mcp
    tool_prefix: "weather_"
    discover: true
"#;
    let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
    let servers = cfg.servers.clone();
    let (validated, catalog) = build_config(cfg).unwrap();
    let discovery = Discovery::new(
        validated.discovery.as_ref().unwrap(),
        Arc::<ScriptedBackend>::clone(backend),
        Some(crate::isolated_subrequest_client(1)),
    );
    McpBrokerFilter::build(&validated, catalog, &servers, None, Some(discovery))
}

fn request(session: Option<&str>, auth: &str) -> praxis_filter::Request {
    let mut req = crate::test_utils::make_request(http::Method::POST, "/mcp");
    req.headers.insert("content-type", "application/json".parse().unwrap());
    req.headers
        .insert("mcp-protocol-version", "2025-03-26".parse().unwrap());
    req.headers.insert("authorization", auth.parse().unwrap());
    if let Some(session) = session {
        req.headers.insert("mcp-session-id", session.parse().unwrap());
    }
    req
}

/// Run one request body; return the action, the forwarded body, and the
/// routed cluster and backend session header.
async fn send(
    filter: &McpBrokerFilter,
    req: &praxis_filter::Request,
    body: &str,
) -> (FilterAction, Option<Bytes>, Option<String>, Option<String>) {
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.current_filter_id = Some(0);
    let mut body = Some(Bytes::from(body.to_owned()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let session = ctx
        .request_headers_to_set
        .iter()
        .find(|(name, _)| name.as_str() == "mcp-session-id")
        .map(|(_, value)| value.to_str().unwrap().to_owned());
    (action, body, ctx.cluster.as_deref().map(str::to_owned), session)
}

async fn issue(filter: &McpBrokerFilter) -> String {
    let req = crate::test_utils::make_request(http::Method::POST, "/mcp");
    let (action, ..) = send(
        filter,
        &req,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}"#,
    )
    .await;
    let FilterAction::Reject(rejection) = action else {
        panic!("initialize must answer locally");
    };
    rejection
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("mcp-session-id"))
        .map(|(_, value)| value.clone())
        .unwrap()
}

fn json_of(action: &FilterAction) -> (u16, Value) {
    let FilterAction::Reject(rejection) = action else {
        panic!("expected a local answer");
    };
    (
        rejection.status,
        serde_json::from_slice(rejection.body.as_ref().unwrap()).unwrap(),
    )
}

const TOOLS_LIST: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;

fn call(tool: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"{tool}","arguments":{{}}}}}}"#)
}

async fn listed_names(filter: &McpBrokerFilter, req: &praxis_filter::Request) -> Vec<String> {
    let (action, ..) = send(filter, req, TOOLS_LIST).await;
    let (status, body) = json_of(&action);
    assert_eq!(status, 200, "{body}");
    body["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap().to_owned())
        .collect()
}

// -----------------------------------------------------------------------------
// Current Profile
// -----------------------------------------------------------------------------

#[tokio::test]
async fn tools_list_discovers_inside_the_callers_backend_session() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["forecast", "radar", "admin"]);
    let filter = current_broker(&backend, CURRENT_YAML);
    let gateway = issue(&filter).await;
    let req = request(Some(&gateway), "Bearer alice");

    let (action, ..) = send(&filter, &req, TOOLS_LIST).await;
    let (_, body) = json_of(&action);
    let tools = body["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["weather_forecast", "weather_radar", "cal_create_event"]);
    assert_eq!(
        tools[0]["description"], "operator description",
        "static description wins"
    );
    assert_eq!(tools[1]["description"], "radar from weather-mcp");

    assert_eq!(
        backend.initialize_count("weather-mcp"),
        1,
        "discovery opens the session"
    );
    assert_eq!(
        backend.initialize_count("calendar-mcp"),
        0,
        "static servers are not contacted"
    );
    let pages = backend.page_requests("weather-mcp");
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].scope, "weather-mcp-1", "listed inside the backend session");
    assert_eq!(pages[0].authorization, "Bearer alice", "with the caller's credential");

    let (action, body, cluster, session) = send(&filter, &req, &call("weather_radar")).await;
    assert!(matches!(action, FilterAction::Release), "the discovered tool routes");
    assert_eq!(cluster.as_deref(), Some("weather-mcp"));
    assert_eq!(session.as_deref(), Some("weather-mcp-1"), "the call reuses the session");
    let forwarded: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(forwarded["params"]["name"], "radar", "the prefix is stripped");
    assert_eq!(backend.initialize_count("weather-mcp"), 1);
    assert_eq!(
        backend.page_requests("weather-mcp").len(),
        1,
        "the call uses the cached listing"
    );

    let (action, ..) = send(&filter, &req, &call("weather_admin")).await;
    let (_, body) = json_of(&action);
    assert_eq!(
        body["error"]["message"], "unknown tool",
        "a hidden tool cannot be called"
    );
}

#[tokio::test]
async fn the_catalog_is_cached_per_caller() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    let filter = current_broker(&backend, CURRENT_YAML);
    let gateway = issue(&filter).await;

    listed_names(&filter, &request(Some(&gateway), "Bearer alice")).await;
    listed_names(&filter, &request(Some(&gateway), "Bearer alice")).await;
    assert_eq!(
        backend.page_requests("weather-mcp").len(),
        1,
        "a fresh listing is reused"
    );

    listed_names(&filter, &request(Some(&gateway), "Bearer bob")).await;
    let pages = backend.page_requests("weather-mcp");
    assert_eq!(pages.len(), 2, "another credential discovers its own catalog");
    assert_eq!(pages[1].authorization, "Bearer bob");
}

#[tokio::test]
async fn a_call_to_an_unlisted_name_discovers_first() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    let filter = current_broker(&backend, CURRENT_YAML);
    let gateway = issue(&filter).await;
    let req = request(Some(&gateway), "Bearer alice");

    let (action, ..) = send(&filter, &req, &call("cal_create_event")).await;
    assert!(matches!(action, FilterAction::Release));
    assert!(
        backend.page_requests("weather-mcp").is_empty(),
        "a name the catalog already exposes needs no discovery"
    );

    let (action, _, cluster, _) = send(&filter, &req, &call("weather_radar")).await;
    assert!(matches!(action, FilterAction::Release), "discovery finds the name");
    assert_eq!(cluster.as_deref(), Some("weather-mcp"));
    assert_eq!(backend.page_requests("weather-mcp").len(), 1);
}

#[tokio::test]
async fn pagination_follows_next_cursor() {
    let backend = Arc::new(ScriptedBackend::default());
    let tool = |name: &str| json!({"name": name, "inputSchema": {"type": "object"}});
    backend.serve_pages("weather-mcp", vec![vec![tool("a")], vec![tool("b")], vec![tool("c")]]);
    let filter = current_broker(&backend, CURRENT_YAML);
    let gateway = issue(&filter).await;

    let names = listed_names(&filter, &request(Some(&gateway), "Bearer alice")).await;
    assert_eq!(
        names,
        [
            "weather_a",
            "weather_b",
            "weather_c",
            "weather_forecast",
            "cal_create_event"
        ]
    );
    let cursors: Vec<Option<String>> = backend
        .page_requests("weather-mcp")
        .into_iter()
        .map(|page| page.cursor)
        .collect();
    assert_eq!(cursors, [None, Some("1".to_owned()), Some("2".to_owned())]);
}

#[tokio::test]
async fn a_failing_server_keeps_the_last_good_tools_and_is_counted() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    let yaml = CURRENT_YAML.replace("backend_sessions:", "discovery:\n  max_age_ms: 20\nbackend_sessions:");
    let filter = current_broker(&backend, &yaml);
    let gateway = issue(&filter).await;
    let req = request(Some(&gateway), "Bearer alice");
    let before = listed_names(&filter, &req).await;
    assert!(before.contains(&"weather_radar".to_owned()));

    backend.fail("weather-mcp", McpBackendError::Rejected { status: 503 });
    tokio::time::sleep(Duration::from_millis(30)).await;
    let recorder = metrics_util::debugging::DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let (action, ..) = {
        let _guard = metrics::set_default_local_recorder(&recorder);
        send(&filter, &req, TOOLS_LIST).await
    };
    let (_, body) = json_of(&action);
    let names: Vec<&str> = body["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, before, "the stale listing is served while the server fails");
    assert_eq!(
        backend.page_requests("weather-mcp").len(),
        2,
        "the stale listing was retried"
    );
    let failures: Vec<String> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .filter(|(key, ..)| key.key().name() == METRIC_DISCOVERY_FAILURES_TOTAL)
        .flat_map(|(key, ..)| {
            key.key()
                .labels()
                .map(|label| format!("{}={}", label.key(), label.value()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(failures, ["server=weather", "reason=rejected"]);

    let other = listed_names(&filter, &request(Some(&gateway), "Bearer bob")).await;
    assert_eq!(
        other,
        ["weather_forecast", "cal_create_event"],
        "a caller without a good listing sees the static entries and the other servers"
    );
}

#[tokio::test]
async fn a_changed_listing_replaces_the_catalog_after_max_age() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    let yaml = CURRENT_YAML.replace("backend_sessions:", "discovery:\n  max_age_ms: 20\nbackend_sessions:");
    let filter = current_broker(&backend, &yaml);
    let gateway = issue(&filter).await;
    let req = request(Some(&gateway), "Bearer alice");
    listed_names(&filter, &req).await;

    backend.serve("weather-mcp", &["alerts"]);
    tokio::time::sleep(Duration::from_millis(30)).await;
    let names = listed_names(&filter, &req).await;
    assert_eq!(names, ["weather_alerts", "weather_forecast", "cal_create_event"]);

    let (action, ..) = send(&filter, &req, &call("weather_radar")).await;
    let (_, body) = json_of(&action);
    assert_eq!(
        body["error"]["message"], "unknown tool",
        "a tool that left the listing gets the unknown-tool error"
    );
}

#[tokio::test]
async fn an_expired_backend_session_is_reopened_once() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    backend
        .once
        .lock()
        .unwrap()
        .insert("weather-mcp".to_owned(), McpBackendError::Rejected { status: 404 });
    let filter = current_broker(&backend, CURRENT_YAML);
    let gateway = issue(&filter).await;
    let req = request(Some(&gateway), "Bearer alice");

    let names = listed_names(&filter, &req).await;
    assert!(names.contains(&"weather_radar".to_owned()));
    let scopes: Vec<String> = backend
        .page_requests("weather-mcp")
        .into_iter()
        .map(|page| page.scope)
        .collect();
    assert_eq!(
        scopes,
        ["weather-mcp-1", "weather-mcp-2"],
        "the listing moves to a new session"
    );

    let (_, _, _, session) = send(&filter, &req, &call("weather_radar")).await;
    assert_eq!(session.as_deref(), Some("weather-mcp-2"), "calls use the new session");
}

#[tokio::test]
async fn discovery_without_a_gateway_session_is_rejected() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    let filter = current_broker(&backend, CURRENT_YAML);

    let (action, ..) = send(&filter, &request(None, "Bearer alice"), TOOLS_LIST).await;
    let (status, body) = json_of(&action);
    assert_eq!(status, 400, "{body}");
    assert!(backend.page_requests("weather-mcp").is_empty());
}

#[tokio::test]
async fn a_server_without_discovery_behaves_as_before() {
    let backend = Arc::new(ScriptedBackend::default());
    let filter = current_broker(
        &backend,
        "
backend_sessions: {}
servers:
  - name: calendar
    cluster: calendar-mcp
    tools:
      - name: create_event
",
    );
    assert!(filter.discovery.is_none());
    let names = listed_names(&filter, &request(None, "Bearer alice")).await;
    assert_eq!(names, ["create_event"]);
    assert!(backend.pages.lock().unwrap().is_empty());
}

// -----------------------------------------------------------------------------
// Stateless Profile
// -----------------------------------------------------------------------------

fn stateless_request(method: &str, name: Option<&str>) -> praxis_filter::Request {
    let mut req = crate::test_utils::make_request(http::Method::POST, "/mcp");
    req.headers.insert("content-type", "application/json".parse().unwrap());
    req.headers
        .insert("mcp-protocol-version", "2026-07-28".parse().unwrap());
    req.headers.insert("mcp-method", method.parse().unwrap());
    req.headers.insert("authorization", "Bearer alice".parse().unwrap());
    if let Some(name) = name {
        req.headers.insert("mcp-name", name.parse().unwrap());
    }
    req
}

fn stateless_body(method: &str, params: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":4,"method":"{method}","params":{{{params}"_meta":{{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{{"name":"t","version":"1"}},"io.modelcontextprotocol/clientCapabilities":{{}}}}}}}}"#
    )
}

#[tokio::test]
async fn stateless_discovery_lists_without_a_session() {
    let backend = Arc::new(ScriptedBackend::default());
    backend.serve("weather-mcp", &["radar"]);
    let filter = stateless_broker(&backend);

    let (action, ..) = send(
        &filter,
        &stateless_request("tools/list", None),
        &stateless_body("tools/list", ""),
    )
    .await;
    let (status, body) = json_of(&action);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["tools"][0]["name"], "weather_radar");
    assert_eq!(
        body["result"]["cacheScope"], "private",
        "per-caller listings are private"
    );
    let pages = backend.page_requests("weather-mcp");
    assert_eq!(pages[0].scope, "stateless:2026-07-28");
    assert_eq!(pages[0].authorization, "Bearer alice");
    assert_eq!(backend.initialize_count("weather-mcp"), 0, "no session is opened");

    let (action, body, cluster, _) = send(
        &filter,
        &stateless_request("tools/call", Some("weather_radar")),
        &stateless_body("tools/call", r#""name":"weather_radar","arguments":{},"#),
    )
    .await;
    assert!(matches!(action, FilterAction::Release), "the discovered tool routes");
    assert_eq!(cluster.as_deref(), Some("weather-mcp"));
    let forwarded: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(forwarded["params"]["name"], "radar");
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

fn config_error(yaml: &str) -> String {
    let cfg: McpBrokerConfig = serde_yaml::from_str(yaml).unwrap();
    build_config(cfg).err().unwrap().to_string()
}

#[test]
fn discovery_config_is_validated() {
    let cases = [
        (
            "servers:\n  - {name: s, cluster: c, discover: true}\n",
            "discover: true requires backend_sessions",
        ),
        (
            "servers:\n  - {name: s, cluster: c, tool_policy: read_only}\n",
            "tool_policy requires discover: true",
        ),
        (
            "servers:\n  - name: s\n    cluster: c\n    tools: [{name: t, hidden: true}]\n",
            "hidden requires discover: true",
        ),
        (
            "discovery: {}\nservers:\n  - {name: s, cluster: c}\n",
            "discovery requires at least one server",
        ),
        (
            "backend_sessions: {}\ndiscovery: {timeout_ms: 10}\nservers:\n  - {name: s, cluster: c, discover: true}\n",
            "discovery.timeout_ms requires protocol_profile 'stateless'",
        ),
        (
            "backend_sessions: {}\ndiscovery: {max_entries: 0}\nservers:\n  - {name: s, cluster: c, discover: true}\n",
            "discovery.max_entries must be between",
        ),
        (
            "protocol_profile: stateless\ncache_scope: public\nservers:\n  - {name: s, cluster: c, discover: true}\n",
            "cache_scope 'public' conflicts",
        ),
        (
            "protocol_profile: stateless\ndiscovery: {forward_headers: [mcp-session-id]}\nservers:\n  - {name: s, \
             cluster: c, discover: true}\n",
            "discovery.forward_headers must not include",
        ),
    ];
    for (yaml, expected) in cases {
        let error = config_error(yaml);
        assert!(error.contains(expected), "{yaml}: {error}");
    }
}

#[test]
fn discovery_defaults_apply_without_a_discovery_block() {
    let cfg: McpBrokerConfig = serde_yaml::from_str(
        "protocol_profile: stateless\nservers:\n  - {name: s, cluster: c, discover: true, tool_policy: read_only}\n",
    )
    .unwrap();
    let (validated, _) = build_config(cfg).unwrap();
    let discovery = validated.discovery.unwrap();
    assert_eq!(discovery.max_entries, 4_096);
    assert_eq!(discovery.max_age, Duration::from_secs(300));
    assert_eq!(discovery.idle_timeout, Duration::from_secs(600));
    assert_eq!(discovery.max_concurrency, 8);
    assert_eq!(discovery.max_tools, 512);
    let stateless = discovery.stateless.unwrap();
    assert_eq!(stateless.forward_headers, ["authorization"]);
    assert_eq!(stateless.timeout, Duration::from_secs(5));
    assert!(discovery.servers[0].is_some());
}
