#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]

use praxis_filter::FilterRegistry;

use crate::{
    config::EnmaasPolicyConfig,
    filter::EnmaasPolicyFilter,
    policy::{DenyReason, MeteringEntitlement},
};

#[test]
fn exports_enmaas_policy() {
    let mut registry = FilterRegistry::with_builtins();
    crate::register_filters(&mut registry);
    assert!(registry.available_filters().contains(&"enmaas_policy"));
}

#[test]
fn parses_minimal_config() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("metering_url: http://127.0.0.1:1\nallow_private_endpoint: true\n").expect("config yaml");
    let filter = EnmaasPolicyFilter::from_config(&yaml).expect("filter builds");
    assert_eq!(filter.name(), "enmaas_policy");
}

#[test]
fn parses_metering_entitlement() {
    let entitlement: MeteringEntitlement = serde_json::from_str(
        r#"{
            "hasAccess": false,
            "modelAllowed": false,
            "denyReason": "model_not_allowed",
            "quotaRetryAfter": 12,
            "qosClass": "ratelimit",
            "qosSubject": "subject-1",
            "qosRetryAfter": 17,
            "qosKeyAlias": "main"
        }"#,
    )
    .expect("entitlement parses");
    let decision = entitlement.into_decision("main", "premium", "main", "main");
    assert!(!decision.allowed);
    assert_eq!(decision.deny_reason, Some(DenyReason::ModelNotAllowed));
    assert_eq!(decision.normalized_class("main"), "ratelimit");
    assert_eq!(decision.retry_after_seconds, Some(17));
    assert_eq!(decision.qos_key_alias.as_deref(), Some("main"));
}

#[test]
fn defaults_are_stable() {
    let config: EnmaasPolicyConfig =
        serde_yaml::from_str("metering_url: http://127.0.0.1:1\nallow_private_endpoint: true\n")
            .expect("config parses");
    assert!(config.enabled);
    assert!(config.fail_open);
    assert_eq!(config.default_class, "main");
    assert!(matches!(config.qos_enforcement, crate::config::QosEnforcement::Soft));
    assert_eq!(config.username_metadata, "x-tenant-username");
    assert_eq!(config.model_metadata, "model_to_provider.client_model");
    assert_eq!(config.timeout_seconds, 5);
    assert_eq!(config.response_class_header, "x-enmaas-qos-class");
    assert_eq!(config.metering_feature_key, "inference-tokens");
    assert!(matches!(config.credential_mode, crate::config::CredentialMode::Off));
    assert_eq!(config.qos_clusters, vec!["cb-glm".to_owned()]);
    assert_eq!(config.key_id_clusters, vec!["cb-glm".to_owned()]);
}

use std::{
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::{Arc, LazyLock, Mutex},
    thread,
};

use bytes::Bytes;
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
        cluster: Some(Arc::from("cb-glm")),
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
    (format!("http://{address}"), captured, handle)
}

fn mock_metering(status: &'static str, body: &'static str) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind mock metering");
    let address = listener.local_addr().expect("mock metering address");
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept metering request");
        let mut buffer = [0_u8; 4096];
        let _ = stream.read(&mut buffer).expect("read metering request");
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).expect("write metering response");
    });
    (format!("http://{address}"), handle)
}

fn mock_metering_counted() -> (String, Arc<Mutex<usize>>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind counted metering");
    let address = listener.local_addr().expect("counted metering address");
    let calls = Arc::new(Mutex::new(0_usize));
    let calls_for_thread = Arc::clone(&calls);
    let handle = thread::spawn(move || {
        listener.set_nonblocking(true).expect("nonblocking listener");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    *calls_for_thread.lock().expect("call count") += 1;
                    let mut buffer = [0_u8; 4096];
                    let _ = stream.read(&mut buffer).expect("read metering request");
                    let response = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 13\r\nConnection: close\r\n\r\nupstream down";
                    if stream.write_all(response.as_bytes()).is_err() {
                        break;
                    }
                },
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::park_timeout(std::time::Duration::from_millis(5));
                },
                Err(_) => break,
            }
        }
    });
    (format!("http://{address}"), calls, handle)
}

fn build_filter(policy_url: &str) -> Box<dyn HttpFilter> {
    build_filter_with(policy_url, "")
}

fn build_filter_with(policy_url: &str, extra: &str) -> Box<dyn HttpFilter> {
    let yaml = serde_yaml::from_str(&format!(
        "metering_url: {policy_url}\nallow_private_endpoint: true\n{extra}"
    ))
    .expect("filter yaml");
    EnmaasPolicyFilter::from_config(&yaml).expect("filter builds")
}

#[tokio::test]
async fn callout_uses_body_model_and_publishes_class() {
    let (policy_url, captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1","qosKeyAlias":"main"}"#,
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
    assert!(captured.lock().expect("capture lock").contains("model=model-a"));
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
        mock_policy(r#"{"hasAccess":true,"modelAllowed":false,"denyReason":"model_not_allowed","qosClass":"main"}"#);
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

#[tokio::test]
async fn quota_denial_uses_retry_after() {
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":false,"modelAllowed":true,"denyReason":"quota_exhausted","qosClass":"main","qosSubject":"subject-1","quotaRetryAfter":42}"#,
    );
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 429);
            assert_eq!(
                rejection
                    .headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
                    .map(|(_, value)| value.as_str()),
                Some("42")
            );
            assert_eq!(
                rejection
                    .headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("x-enmaas-qos-class"))
                    .map(|(_, value)| value.as_str()),
                Some("main")
            );
        },
        other => panic!("expected quota rejection, got {other:?}"),
    }
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn hard_ratelimit_rejects_with_policy_retry() {
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"ratelimit","qosSubject":"subject-1","qosRetryAfter":17}"#,
    );
    let filter = build_filter_with(&policy_url, "qos_enforcement: hard\n");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 429);
            assert_eq!(
                rejection
                    .headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
                    .map(|(_, value)| value.as_str()),
                Some("17")
            );
        },
        other => panic!("expected ratelimit rejection, got {other:?}"),
    }
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn soft_ratelimit_passes_and_publishes_class() {
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"ratelimit","qosSubject":"subject-1","qosRetryAfter":17}"#,
    );
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.get_metadata("qos.class"), Some("ratelimit"));
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn local_bucket_rejects_after_burst() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter_with(
        &policy_url,
        "qos_enforcement: hard\nmain_burst: 1\nmain_refill_per_second: 0.001\n",
    );
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut first = make_filter_context(&request);
    first.set_metadata("x-tenant-username", "alice");
    first.set_metadata("model_to_provider.client_model", "model-a");
    let first_action = filter.on_request(&mut first).await.expect("first policy request");
    assert!(matches!(first_action, FilterAction::Continue));

    let mut second = make_filter_context(&request);
    second.set_metadata("x-tenant-username", "alice");
    second.set_metadata("model_to_provider.client_model", "model-a");
    let second_action = filter.on_request(&mut second).await.expect("second policy request");
    match second_action {
        FilterAction::Reject(rejection) => assert_eq!(rejection.status, 429),
        other => panic!("expected local bucket rejection, got {other:?}"),
    }
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn strips_client_headers_before_forwarding() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request.headers.insert(
        http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer client-token"),
    );
    request.headers.insert(
        http::header::ACCEPT_ENCODING,
        HeaderValue::from_static("gzip, deflate, br"),
    );
    request
        .headers
        .insert("x-api-key", HeaderValue::from_static("client-api-key"));
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));

    let removed: Vec<&str> = ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect();
    assert!(removed.contains(&"authorization"));
    assert!(removed.contains(&"accept-encoding"));
    assert!(removed.contains(&"x-api-key"));
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn forwards_validated_key_id_only_for_cb_glm() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");
    ctx.set_metadata("api_key.id", "key-uuid-1");
    ctx.request_headers_to_remove.clear();

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.request_headers_to_remove
            .contains(&HeaderName::from_static("x-enmaas-key-id"))
    );
    assert!(
        ctx.extra_request_headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-enmaas-key-id") && value == "key-uuid-1")
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn does_not_forward_key_id_to_external_cluster_but_strips_spoof() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request
        .headers
        .insert("x-enmaas-key-id", HeaderValue::from_static("spoofed"));
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("openai"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");
    ctx.set_metadata("api_key.id", "key-uuid-1");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        !ctx.extra_request_headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("x-enmaas-key-id"))
    );
    assert!(
        ctx.request_headers_to_remove
            .contains(&HeaderName::from_static("x-enmaas-key-id"))
    );
    let effective = praxis_ai_apis::callout_headers::effective_request_callout_headers(
        &ctx,
        std::borrow::Cow::Borrowed(&ctx.request.headers),
    );
    assert!(effective.get("x-enmaas-key-id").is_none());
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn stripped_key_id_is_replaced_with_validated_value() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request
        .headers
        .insert("x-enmaas-key-id", HeaderValue::from_static("spoofed"));
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");
    ctx.set_metadata("api_key.id", "validated-key-id");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.request_headers_to_remove
            .contains(&HeaderName::from_static("x-enmaas-key-id"))
    );
    assert!(
        ctx.extra_request_headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-enmaas-key-id") && value == "validated-key-id")
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn final_effective_headers_contain_one_validated_key_id() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request
        .headers
        .insert("x-enmaas-key-id", HeaderValue::from_static("spoofed"));
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");
    ctx.set_metadata("api_key.id", "validated-key-id");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    let effective = praxis_ai_apis::callout_headers::effective_request_callout_headers(
        &ctx,
        std::borrow::Cow::Borrowed(&ctx.request.headers),
    );
    let values: Vec<&str> = effective
        .get_all("x-enmaas-key-id")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    assert_eq!(values, vec!["validated-key-id"]);
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn missing_key_id_strips_spoofed_header_without_injecting() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request
        .headers
        .insert("x-enmaas-key-id", HeaderValue::from_static("spoofed"));
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.request_headers_to_remove
            .contains(&HeaderName::from_static("x-enmaas-key-id"))
    );
    assert!(
        !ctx.extra_request_headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("x-enmaas-key-id"))
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn fail_open_without_identity_still_strips_client_headers() {
    let filter = build_filter("http://127.0.0.1:1");
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request.headers.insert(
        http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer client-token"),
    );
    request.headers.insert(
        http::header::ACCEPT_ENCODING,
        HeaderValue::from_static("gzip, deflate, br"),
    );
    let mut ctx = make_filter_context(&request);

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.request_headers_to_remove.contains(&http::header::AUTHORIZATION));
    assert!(ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING));
}

#[tokio::test]
async fn spoofed_x_api_key_is_stripped_before_class_key_injection() {
    let key_file = tempfile::NamedTempFile::new().expect("temp key file");
    std::fs::write(key_file.path(), r#"{"main":[{"id":"main-0","key":"secret-main"}]}"#).expect("write key file");
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1","qosKeyAlias":"main"}"#,
    );
    let extra = format!(
        "credential_mode: active\ncredential_header: x-api-key\nkey_file: {}\n",
        key_file.path().display()
    );
    let filter = build_filter_with(&policy_url, &extra);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request
        .headers
        .insert("x-api-key", HeaderValue::from_static("spoofed-client-key"));
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.request_headers_to_remove
            .contains(&HeaderName::from_static("x-api-key"))
    );
    assert!(
        ctx.extra_request_headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("x-api-key") && value == "Bearer secret-main")
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn active_key_is_added_after_client_credential_strip() {
    let key_file = tempfile::NamedTempFile::new().expect("temp key file");
    std::fs::write(key_file.path(), r#"{"main":[{"id":"main-0","key":"secret-main"}]}"#).expect("write key file");
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1","qosKeyAlias":"main"}"#,
    );
    let extra = format!("credential_mode: active\nkey_file: {}\n", key_file.path().display());
    let filter = build_filter_with(&policy_url, &extra);
    let mut request = make_request(Method::POST, "/v1/chat/completions");
    request.headers.insert(
        http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer client-token"),
    );
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.request_headers_to_remove.contains(&http::header::AUTHORIZATION));
    assert!(
        ctx.extra_request_headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization") && value == "Bearer secret-main")
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn class_key_alias_is_injected() {
    let key_file = tempfile::NamedTempFile::new().expect("temp key file");
    std::fs::write(key_file.path(), r#"{"main":[{"id":"main-0","key":"secret-main"}]}"#).expect("write key file");
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1","qosKeyAlias":"main"}"#,
    );
    let extra = format!("credential_mode: active\nkey_file: {}\n", key_file.path().display());
    let filter = build_filter_with(&policy_url, &extra);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.extra_request_headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization") && value == "Bearer secret-main")
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn shadow_injects_main_key_for_premium_request() {
    let key_file = tempfile::NamedTempFile::new().expect("temp key file");
    std::fs::write(
        key_file.path(),
        r#"{"premium":[{"id":"premium-0","key":"secret-premium"}],"main":[{"id":"main-0","key":"secret-main"}]}"#,
    )
    .expect("write key file");
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"premium","qosSubject":"subject-1","qosKeyAlias":"premium"}"#,
    );
    let extra = format!("credential_mode: shadow\nkey_file: {}\n", key_file.path().display());
    let filter = build_filter_with(&policy_url, &extra);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.extra_request_headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization") && value == "Bearer secret-main")
    );
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn off_does_not_inject_credentials() {
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"premium","qosSubject":"subject-1","qosKeyAlias":"premium"}"#,
    );
    let filter = build_filter_with(&policy_url, "credential_mode: off\n");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("cb-glm"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.extra_request_headers.is_empty());
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn buffered_body_model_is_authoritative_when_metadata_is_absent() {
    let (policy_url, captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("x-tenant-model", "header-model");
    ctx.buffered_request_body = Some(Bytes::from_static(br#"{"model":"body-model","messages":[]}"#));

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert!(captured.lock().expect("capture lock").contains("model=body-model"));
    handle.join().expect("mock policy thread");
}


#[tokio::test]
async fn qos_scope_does_not_enforce_ratelimit_on_external_cluster() {
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"ratelimit","qosSubject":"subject-1","qosRetryAfter":17}"#,
    );
    let filter = build_filter_with(&policy_url, "qos_enforcement: hard\n");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("openai"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.get_metadata("qos.class"), None);
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn qos_scope_does_not_bucket_external_cluster() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":true,"qosClass":"main","qosSubject":"subject-1"}"#);
    let filter = build_filter_with(
        &policy_url,
        "qos_enforcement: hard\nmain_burst: 1\nmain_refill_per_second: 0.001\n",
    );
    let request = make_request(Method::POST, "/v1/chat/completions");
    for _ in 0..2 {
        let mut ctx = make_filter_context(&request);
        ctx.cluster = Some(Arc::from("openai"));
        ctx.set_metadata("x-tenant-username", "alice");
        ctx.set_metadata("model_to_provider.client_model", "model-a");
        let action = filter.on_request(&mut ctx).await.expect("policy request");
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.get_metadata("qos.class"), None);
    }
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn quota_denial_remains_global_for_external_cluster() {
    let (policy_url, _captured, handle) = mock_policy(
        r#"{"hasAccess":false,"modelAllowed":true,"denyReason":"quota_exhausted","qosClass":"main","qosSubject":"subject-1","quotaRetryAfter":42}"#,
    );
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("openai"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 429);
            assert!(
                rejection
                    .headers
                    .iter()
                    .all(|(name, _)| !name.eq_ignore_ascii_case("x-enmaas-qos-class"))
            );
        },
        other => panic!("expected quota rejection, got {other:?}"),
    }
    handle.join().expect("mock policy thread");
}

#[tokio::test]
async fn model_denial_remains_global_without_class_header_on_external_cluster() {
    let (policy_url, _captured, handle) =
        mock_policy(r#"{"hasAccess":true,"modelAllowed":false,"denyReason":"model_not_allowed","qosClass":"main"}"#);
    let filter = build_filter(&policy_url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.cluster = Some(Arc::from("openai"));
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => {
            assert_eq!(rejection.status, 403);
            assert!(
                rejection
                    .headers
                    .iter()
                    .all(|(name, _)| !name.eq_ignore_ascii_case("x-enmaas-qos-class"))
            );
        },
        other => panic!("expected model rejection, got {other:?}"),
    }
    handle.join().expect("mock policy thread");
}


#[tokio::test]
async fn metering_5xx_fails_open() {
    let (url, handle) = mock_metering("503 Service Unavailable", "upstream down");
    let filter = build_filter(&url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.get_metadata("qos.class"), Some("main"));
    handle.join().expect("mock metering thread");
}

#[tokio::test]
async fn fail_open_decision_is_short_cached() {
    let (url, calls, handle) = mock_metering_counted();
    let filter = build_filter(&url);
    let request = make_request(Method::POST, "/v1/chat/completions");

    for _ in 0..2 {
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("x-tenant-username", "alice");
        ctx.set_metadata("model_to_provider.client_model", "model-a");
        let action = filter.on_request(&mut ctx).await.expect("policy request");
        assert!(matches!(action, FilterAction::Continue));
    }

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(
        *calls.lock().expect("call count"),
        1,
        "fail-open decision should suppress the second callout"
    );
    handle.join().expect("counted metering thread");
}

#[tokio::test]
async fn metering_malformed_json_fails_open() {
    let (url, handle) = mock_metering("200 OK", "not-json");
    let filter = build_filter(&url);
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    handle.join().expect("mock metering thread");
}

#[tokio::test]
async fn metering_5xx_fail_closed_rejects() {
    let (url, handle) = mock_metering("503 Service Unavailable", "upstream down");
    let filter = build_filter_with(&url, "fail_open: false\n");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => assert_eq!(rejection.status, 503),
        other => panic!("expected fail-closed rejection, got {other:?}"),
    }
    handle.join().expect("mock metering thread");
}

#[tokio::test]
async fn missing_identity_fails_open_by_default() {
    let filter = build_filter("http://127.0.0.1:1");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.get_metadata("qos.class"), None);
}

#[tokio::test]
async fn missing_identity_fail_closed_rejects() {
    let filter = build_filter_with("http://127.0.0.1:1", "fail_open: false\n");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("model_to_provider.client_model", "model-a");

    let action = filter.on_request(&mut ctx).await.expect("policy request");
    match action {
        FilterAction::Reject(rejection) => assert_eq!(rejection.status, 503),
        other => panic!("expected missing-identity rejection, got {other:?}"),
    }
}

#[tokio::test]
async fn missing_model_rejected_when_disallowed() {
    let filter = build_filter_with("http://127.0.0.1:1", "allow_missing_model: false\n");
    let request = make_request(Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&request);
    ctx.set_metadata("x-tenant-username", "alice");

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
        other => panic!("expected missing-model rejection, got {other:?}"),
    }
}
