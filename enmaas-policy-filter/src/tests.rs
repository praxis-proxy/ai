#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]

use praxis_filter::FilterRegistry;

use crate::{config::EnmaasPolicyConfig, filter::EnmaasPolicyFilter, policy::PolicyDecision};

#[test]
fn exports_enmaas_policy() {
    let mut registry = FilterRegistry::with_builtins();
    crate::register_filters(&mut registry);
    assert!(registry.available_filters().contains(&"enmaas_policy"));
}

#[test]
fn parses_minimal_config() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("policy_url: http://127.0.0.1:1\nallow_private_endpoint: true\n").expect("config yaml");
    let filter = EnmaasPolicyFilter::from_config(&yaml).expect("filter builds");
    assert_eq!(filter.name(), "enmaas_policy");
}

#[test]
fn parses_policy_decision() {
    let decision: PolicyDecision = serde_json::from_str(
        r#"{
            "version": 1,
            "hasAccess": true,
            "modelAllowed": false,
            "qosClass": "ratelimit",
            "qosSubject": "subject-1",
            "qosRetryAfter": 12,
            "qosKeyAlias": "main-0",
            "cacheTtlSeconds": 3
        }"#,
    )
    .expect("decision parses");
    assert!(decision.has_access);
    assert!(!decision.model_allowed);
    assert_eq!(decision.normalized_class("main"), "ratelimit");
    assert_eq!(decision.qos_retry_after, Some(12));
}

#[test]
fn rejects_unknown_decision_fields() {
    let decision = serde_json::from_str::<PolicyDecision>(
        r#"{"version":1,"hasAccess":true,"modelAllowed":true,"qosClass":"main","extra":1}"#,
    );
    assert!(decision.is_err());
}

#[test]
fn defaults_are_stable() {
    let config: EnmaasPolicyConfig =
        serde_yaml::from_str("policy_url: http://127.0.0.1:1\nallow_private_endpoint: true\n").expect("config parses");
    assert!(config.enabled);
    assert!(config.fail_open);
    assert_eq!(config.default_class, "main");
    assert!(matches!(config.qos_enforcement, crate::config::QosEnforcement::Soft));
    assert_eq!(config.username_metadata, "x-tenant-username");
    assert_eq!(config.model_metadata, "model_to_provider.client_model");
}

use std::{
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::{Arc, LazyLock, Mutex},
    thread,
};

use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use praxis_core::{id::IdGenerator, time::SystemTimeSource};
use praxis_filter::{
    BodyMode, FilterAction, HttpFilter, HttpFilterContext, Request, RequestExtensions, Response, SubRequestResponseMode,
};

static TEST_ID_GENERATOR: LazyLock<IdGenerator> = LazyLock::new(|| IdGenerator::with_seed(0));

fn make_request(method: Method, path: &str) -> Request {
    Request {
        method,
        uri: path.parse::<Uri>().expect("test uri"),
        headers: HeaderMap::new(),
    }
}

fn make_filter_context(req: &Request) -> HttpFilterContext<'_> {
    HttpFilterContext {
        buffered_request_body: None,
        body_done_indices: Vec::new(),
        branch_iterations: std::collections::HashMap::new(),
        grpc_completion: None,
        client_addr: None,
        cluster: None,
        current_filter_id: None,
        downstream_tls: false,
        extensions: RequestExtensions::default(),
        executed_filter_indices: Vec::new(),
        executed_branch_filters: Vec::new(),
        extra_request_headers: Vec::new(),
        request_headers_to_remove: Vec::new(),
        request_headers_to_set: Vec::new(),
        filter_metadata: std::collections::HashMap::new(),
        pre_read_mutations: Vec::new(),
        prior_pre_read_mutations: Vec::new(),
        structured_metadata: std::collections::HashMap::new(),
        filter_results: std::collections::HashMap::new(),
        filter_state: std::collections::HashMap::new(),
        health_registry: None,
        id_generator: &TEST_ID_GENERATOR,
        kv_stores: None,
        session_stores: None,
        metrics_route: None,
        peer_identity: None,
        request: req,
        request_body_bytes: 0,
        request_body_mode: BodyMode::Stream,
        request_start: std::time::Instant::now(),
        response_body_bytes: 0,
        response_body_mode: BodyMode::Stream,
        response_header: None,
        response_headers_modified: false,
        subrequest_client: None,
        subrequest_response_mode: SubRequestResponseMode::Buffered,
        attempted_endpoints: Vec::new(),
        retry_policy: None,
        route_retry_policy: None,
        cluster_retry_state: None,
        cluster_retry_state_released: false,
        endpoint_reselector: None,
        pinned_endpoint_address: None,
        rewritten_path: None,
        selected_endpoint_index: None,
        time_source: &SystemTimeSource,
        upstream: None,
        upstream_reached: false,
    }
}

fn mock_policy(body: &'static str) -> (String, Arc<Mutex<String>>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock policy");
    let address = listener.local_addr().expect("mock policy address");
    let captured = Arc::new(Mutex::new(String::new()));
    let captured_for_thread = Arc::clone(&captured);
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept policy request");
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            let length = stream.read(&mut chunk).expect("read policy request");
            if length == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..length]);
            if let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                let header_end = header_end + 4;
                let headers = String::from_utf8_lossy(&buffer[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length:")
                            .or_else(|| line.strip_prefix("Content-Length:"))
                    })
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if buffer.len() >= header_end + content_length {
                    break;
                }
            }
        }
        *captured_for_thread.lock().expect("capture lock") = String::from_utf8_lossy(&buffer).to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).expect("write policy response");
    });
    (format!("http://{address}/v1/policy/decide"), captured, handle)
}

fn build_filter(policy_url: &str) -> Box<dyn HttpFilter> {
    let yaml = serde_yaml::from_str(&format!("policy_url: {policy_url}\nallow_private_endpoint: true\n"))
        .expect("filter yaml");
    EnmaasPolicyFilter::from_config(&yaml).expect("filter builds")
}

#[tokio::test]
async fn callout_uses_body_model_and_publishes_class() {
    let (policy_url, captured, handle) = mock_policy(
        r#"{"version":1,"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1","cacheTtlSeconds":5}"#,
    );
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");
    ctx.set_metadata("x-tenant-model", "model-b");
    ctx.request_headers_to_set
        .push((HeaderName::from_static("x-model"), HeaderValue::from_static("model-c")));

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.get_metadata("qos.class"), Some("main"));
    assert_eq!(ctx.get_metadata("qos.subject"), Some("subject-1"));
    assert!(captured.lock().expect("capture lock").contains("\"model\":\"model-a\""));
    handle.join().expect("mock policy thread");

    let mut response = Response {
        headers: HeaderMap::new(),
        status: StatusCode::OK,
    };
    ctx.response_header = Some(&mut response);
    drop(filter.on_response(&mut ctx).await.expect("policy response"));
    ctx.response_header = None;
    assert_eq!(
        response
            .headers
            .get("x-enmaas-qos-class")
            .and_then(|value| value.to_str().ok()),
        Some("main")
    );
}

#[tokio::test]
async fn model_denial_rejects_with_403() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"version":1,"hasAccess":true,"modelAllowed":false,"qosClass":"main"}"#);
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 403);
            assert_eq!(
                rejection
                    .headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("x-enmaas-qos-class"))
                    .map(|(_, value)| value.as_str()),
                Some("main")
            );
        },
        other => panic!("expected 403 rejection, got {other:?}"),
    }
    handle.join().expect("mock policy thread");
}
