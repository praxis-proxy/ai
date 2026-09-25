// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `openai_responses_format` classifier filter.

use praxis_core::config::Config;
#[cfg(feature = "store-sqlite")]
use praxis_test_utils::StatefulCapturingBackend;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_header, parse_status, start_backend_with_shutdown,
    start_echo_backend, start_header_echo_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Classification and Routing Tests
// -----------------------------------------------------------------------------

#[test]
fn responses_request_routes_to_responses_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "responses request should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "responses request should route to responses cluster"
    );
}

#[test]
fn responses_array_input_routes_to_responses_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "array input should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "array input request should route to responses cluster"
    );
}

#[test]
fn chat_completions_routes_to_chat_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "chat completions should return 200");
    assert_eq!(
        parse_body(&raw),
        "chat-backend",
        "chat completions should route to chat cluster"
    );
}

#[test]
fn unknown_json_routes_to_default_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","prompt":"hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "unknown JSON should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "unknown JSON on a non-create path should route to the default cluster; the \
         endpoint is not authoritative there (on POST /v1/responses the same body is a \
         Responses create request -- see responses_create_without_discriminator_*)"
    );
}

#[test]
fn responses_create_without_discriminator_routes_to_responses_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-5"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "responses create should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "a create body omitting the discriminator fields must still route to the \
         responses cluster because POST /v1/responses is authoritative, not fall \
         through to default"
    );
}

#[test]
fn non_json_continues_by_default() {
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = continue_yaml(proxy_port, default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "POST /v1/responses HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: 11\r\n\
         \r\n\
         hello world"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "non-JSON should continue to default cluster");
    assert_eq!(parse_body(&raw), "default-backend", "non-JSON should reach backend");
}

#[test]
fn unknown_json_rejected_when_configured() {
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = reject_yaml(proxy_port, default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","prompt":"hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(
        parse_status(&raw),
        400,
        "unknown JSON should be rejected when on_invalid: reject; this uses a \
         non-create path so the body is not promoted to responses by endpoint authority"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("unrecognized AI API format"),
        "rejection should mention unrecognized format, got: {response_body}"
    );
}

#[test]
fn invalid_json_rejected_when_configured() {
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = reject_yaml(proxy_port, default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", "not valid json {{{"));

    assert_eq!(parse_status(&raw), 400, "invalid JSON should be rejected");
    let body = parse_body(&raw);
    assert!(
        body.contains("invalid JSON body") || body.contains("invalid_request_error"),
        "rejection should mention invalid JSON, got: {body}"
    );
}

#[test]
fn non_json_rejected_when_configured() {
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = reject_yaml(proxy_port, default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "POST /v1/responses HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: 11\r\n\
         \r\n\
         hello world"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        400,
        "non-JSON should be rejected when on_invalid: reject"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("not JSON") || body.contains("invalid_request_error"),
        "rejection should mention non-JSON, got: {body}"
    );
}

// -----------------------------------------------------------------------------
// Header Promotion for Routing
// -----------------------------------------------------------------------------

#[test]
fn model_header_routes_to_specific_backend() {
    let model_a_guard = start_backend_with_shutdown("model-a-backend");
    let model_b_guard = start_backend_with_shutdown("model-b-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = model_routing_yaml(
        proxy_port,
        model_a_guard.port(),
        model_b_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"model-a","input":"test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "model-a should return 200");
    assert_eq!(
        parse_body(&raw),
        "model-a-backend",
        "model-a should route to model-a-backend via promoted x-praxis-ai-model header"
    );

    let body_b = r#"{"model":"model-b","input":"test"}"#;
    let raw_b = http_send(proxy.addr(), &json_post("/v1/responses", body_b));

    assert_eq!(parse_status(&raw_b), 200, "model-b should return 200");
    assert_eq!(
        parse_body(&raw_b),
        "model-b-backend",
        "model-b should route to model-b-backend via promoted x-praxis-ai-model header"
    );
}

#[test]
fn stream_header_routes_streaming_traffic() {
    let stream_guard = start_backend_with_shutdown("stream-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = stream_routing_yaml(proxy_port, stream_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","stream":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "streaming request should return 200");
    assert_eq!(
        parse_body(&raw),
        "stream-backend",
        "stream:true should route to stream-backend via promoted x-praxis-ai-stream header"
    );

    let body_no_stream = r#"{"model":"gpt-4.1","input":"test"}"#;
    let raw_no_stream = http_send(proxy.addr(), &json_post("/v1/responses", body_no_stream));

    assert_eq!(parse_status(&raw_no_stream), 200, "non-streaming should return 200");
    assert_eq!(
        parse_body(&raw_no_stream),
        "default-backend",
        "request without stream should route to default-backend"
    );
}

#[test]
fn reserved_headers_stripped_before_upstream() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();

    let yaml = header_echo_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","stream":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw).to_lowercase();
    assert!(
        !echoed.contains("x-praxis-ai-format"),
        "x-praxis-ai-format should be stripped before upstream (reserved header)"
    );
    assert!(
        !echoed.contains("x-praxis-ai-model"),
        "x-praxis-ai-model should be stripped before upstream (reserved header)"
    );
    assert!(
        !echoed.contains("x-praxis-ai-stream"),
        "x-praxis-ai-stream should be stripped before upstream (reserved header)"
    );
}

// -----------------------------------------------------------------------------
// Bounded Value Promotion Tests
// -----------------------------------------------------------------------------

#[test]
fn oversized_model_not_promoted_to_header() {
    let model_guard = start_backend_with_shutdown("model-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let long_model = "x".repeat(300);
    let yaml = model_routing_yaml(proxy_port, model_guard.port(), model_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(r#"{{"model":"{long_model}","input":"test"}}"#);
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "oversized model should still return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "oversized model (>256 bytes) should not match a route, falling to default"
    );
}

#[test]
fn control_char_model_not_promoted_to_header() {
    let model_guard = start_backend_with_shutdown("model-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = model_routing_yaml(proxy_port, model_guard.port(), model_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"bad\nmodel","input":"test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "control-char model should still return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "model with control chars should not be promoted, falling to default"
    );
}

// -----------------------------------------------------------------------------
// Filter Results Branch Tests
// -----------------------------------------------------------------------------

#[test]
fn filter_results_enable_branch_routing() {
    let responses_guard = start_backend_with_shutdown("responses-branch-hit");
    let default_guard = start_backend_with_shutdown("default-branch-miss");
    let proxy_port = free_port();

    let yaml = branch_yaml(proxy_port, responses_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "responses request should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-branch-hit",
        "branch on_result should fire for responses format and route to branch cluster"
    );

    let chat_body = r#"{"model":"gpt-4","messages":[]}"#;
    let raw_chat = http_send(proxy.addr(), &json_post("/v1/chat/completions", chat_body));

    assert_eq!(parse_status(&raw_chat), 200, "chat request should return 200");
    assert_eq!(
        parse_body(&raw_chat),
        "default-branch-miss",
        "branch should not fire for openai_chat_completions, falling through to default"
    );
}

#[test]
fn background_true_is_rejected_by_unconditioned_validator() {
    let backend_guard = start_backend_with_shutdown("foreground-forwarded");
    let proxy_port = free_port();

    let yaml = unconditioned_background_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","background":true,"store":false,"stream":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 400, "background request should fail before routing");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(response["error"]["message"], "background mode is not supported");

    let foreground_body = r#"{"model":"gpt-4.1","input":"test","background":false}"#;
    let foreground_raw = http_send(proxy.addr(), &json_post("/v1/responses", foreground_body));

    assert_eq!(
        parse_status(&foreground_raw),
        200,
        "foreground request should return 200"
    );
    assert_eq!(
        parse_body(&foreground_raw),
        "foreground-forwarded",
        "background:false should reach the backend"
    );
}

#[test]
fn api_openai_hostname_does_not_override_non_openai_provider() {
    let proxy_port = free_port();
    let yaml = non_openai_api_hostname_background_yaml(proxy_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","background":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        400,
        "endpoint hostname must not grant background lifecycle ownership"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(response["error"]["message"], "background mode is not supported");
}

#[test]
#[cfg(feature = "store-sqlite")]
fn background_true_and_polling_are_forwarded_to_openai_provider() {
    let backend = StatefulCapturingBackend::new(vec![
        (200, r#"{"id":"resp_background","status":"queued"}"#.to_owned()),
        (200, r#"{"id":"resp_background","status":"completed"}"#.to_owned()),
        (200, r#"{"id":"resp_background","status":"queued"}"#.to_owned()),
        (200, r#"{"id":"resp_background","status":"completed"}"#.to_owned()),
    ])
    .start_with_shutdown();
    let temp = tempfile::tempdir().unwrap();
    let database_path = temp.path().join("openai-skip-store.db");
    let database_url = format!("sqlite://{}?mode=rwc", database_path.display());
    let proxy_port = free_port();
    let yaml = openai_background_lifecycle_yaml(proxy_port, backend.port(), &database_url);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let bodies = [
        r#"{"model":"gpt-4.1","input":"test","background":true,"store":true,"stream":false}"#,
        r#"{"model":"gpt-4.1","input":"test","background":true,"store":true,"stream":true}"#,
    ];
    for body in bodies {
        let create = http_send(proxy.addr(), &json_post("/v1/responses", body));
        assert_eq!(parse_status(&create), 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parse_body(&create)).unwrap()["status"],
            "queued"
        );

        let get = format!(
            "GET /v1/responses/resp_background HTTP/1.1\r\n\
             Host: localhost:{proxy_port}\r\n\
             Connection: close\r\n\
             \r\n"
        );
        let poll = http_send(proxy.addr(), &get);
        assert_eq!(parse_status(&poll), 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&parse_body(&poll)).unwrap()["status"],
            "completed"
        );
    }

    let captured = backend.requests();
    assert!(
        bodies.iter().all(|body| {
            captured
                .iter()
                .any(|request| request.method == "POST" && request.uri == "/v1/responses" && request.body == *body)
        }),
        "OpenAI should receive unchanged finite and streaming background creates"
    );
    assert!(
        captured
            .iter()
            .any(|request| request.method == "GET" && request.uri == "/v1/responses/resp_background"),
        "polling must reach the same OpenAI lifecycle owner"
    );
    assert!(
        !database_path.exists(),
        "the provider-gated local store must not execute for OpenAI-owned passthrough"
    );
}

#[test]
#[cfg(feature = "store-sqlite")]
fn non_openai_bound_background_rejects_before_store_irr_or_backend() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"id":"unexpected"}"#.to_owned())]).start_with_shutdown();
    let temp = tempfile::tempdir().unwrap();
    let database_path = temp.path().join("managed-store.db");
    let database_url = format!("sqlite://{}?mode=rwc", database_path.display());
    let proxy_port = free_port();
    let yaml = managed_background_yaml(proxy_port, backend.port(), &database_url);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","background":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 400, "IRR background request should be rejected");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(response["error"]["type"], "invalid_request_error");
    assert_eq!(response["error"]["code"], "invalid_request_error");
    assert!(response["error"]["param"].is_null());
    assert_eq!(response["error"]["message"], "background mode is not supported");
    let captured = backend.requests();
    let captured_summary: Vec<_> = captured
        .iter()
        .map(|request| format!("{} {} {}", request.method, request.uri, request.body))
        .collect();
    assert!(
        captured
            .iter()
            .all(|request| request.method != "POST" || request.uri != "/v1/responses"),
        "bound policy must reject before IRR forwards the client request; captured: {captured_summary:?}"
    );
    assert!(
        !database_path.exists(),
        "validator rejection must happen before the conditioned store executes"
    );
}

#[cfg(feature = "store-sqlite")]
#[test]
fn executing_local_store_rejects_background() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"id":"unexpected"}"#.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let yaml = stored_background_yaml(proxy_port, backend.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","background":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        400,
        "an executing local store must reject background"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(response["error"]["type"], "invalid_request_error");
    assert_eq!(response["error"]["code"], "invalid_request_error");
    assert!(response["error"]["param"].is_null());
    assert_eq!(response["error"]["message"], "background mode is not supported");
    let captured = backend.requests();
    let captured_summary: Vec<_> = captured
        .iter()
        .map(|request| format!("{} {} {}", request.method, request.uri, request.body))
        .collect();
    assert!(
        captured
            .iter()
            .all(|request| request.method != "POST" || request.uri != "/v1/responses"),
        "local store rejection must happen before forwarding the client request; captured: {captured_summary:?}"
    );
}

// -----------------------------------------------------------------------------
// Path-Based Classification (GET / DELETE)
// -----------------------------------------------------------------------------

#[test]
fn get_v1_responses_with_id_routes_to_responses_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "GET /v1/responses/resp_abc123 HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "GET /v1/responses/{{id}} should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "GET /v1/responses/{{id}} should route to responses cluster"
    );
}

#[test]
fn get_v1_responses_input_items_routes_to_responses_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "GET /v1/responses/resp_abc123/input_items HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "GET /v1/responses/{{id}}/input_items should return 200"
    );
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "GET /v1/responses/{{id}}/input_items should route to responses cluster"
    );
}

#[test]
fn delete_v1_responses_with_id_routes_to_responses_cluster() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "DELETE /v1/responses/resp_abc123 HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "DELETE /v1/responses/{{id}} should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "DELETE /v1/responses/{{id}} should route to responses cluster"
    );
}

#[test]
fn get_v1_responses_branch_routes_correctly() {
    let responses_guard = start_backend_with_shutdown("responses-branch-hit");
    let default_guard = start_backend_with_shutdown("default-branch-miss");
    let proxy_port = free_port();

    let yaml = branch_yaml(proxy_port, responses_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "GET /v1/responses/resp_abc123 HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "GET branch should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-branch-hit",
        "GET path-classified request should trigger branch on format=responses"
    );
}

#[test]
fn get_unrelated_path_routes_to_default() {
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let chat_guard = start_backend_with_shutdown("chat-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(
        proxy_port,
        responses_guard.port(),
        chat_guard.port(),
        default_guard.port(),
    );
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "GET /v1/models HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "unrelated GET should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "GET /v1/models should route to default cluster"
    );
}

// -----------------------------------------------------------------------------
// Body Preservation Tests
// -----------------------------------------------------------------------------

#[test]
fn body_unchanged_after_classification() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = echo_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello, world!","stream":false,"store":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    assert_eq!(
        echoed, body,
        "body should be byte-for-byte unchanged after classification"
    );
}

#[test]
fn large_body_over_64k_classified_and_forwarded() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = echo_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let padding = "x".repeat(65_536); // 64 KiB of padding
    let body = format!(r#"{{"model":"gpt-4.1","input":"{padding}"}}"#);
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "large body should return 200");
    let echoed = parse_body(&raw);
    assert_eq!(echoed, body, "large body should be byte-for-byte unchanged");
}

// -----------------------------------------------------------------------------
// Error Formatter Integration Tests
// -----------------------------------------------------------------------------

#[test]
fn proxy_failure_formats_openai_error_for_responses() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = echo_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "Content-Type should be application/json"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("response body should be valid JSON");
    assert_eq!(
        parsed["error"]["type"], "server_error",
        "OpenAI error type should be server_error for 502"
    );
    assert!(parsed["error"]["param"].is_null(), "param should be null");
    assert!(
        parsed["error"]["message"].is_string(),
        "error message should be a string"
    );
    assert!(parsed["error"]["code"].is_string(), "error code should be a string");
}

#[test]
fn proxy_failure_formats_openai_error_for_chat_completions() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = echo_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "Content-Type should be application/json"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("response body should be valid JSON");
    assert_eq!(
        parsed["error"]["type"], "server_error",
        "OpenAI error type should be server_error for 502"
    );
    assert!(parsed["error"]["param"].is_null(), "param should be null");
    assert!(
        parsed["error"]["message"].is_string(),
        "error message should be a string"
    );
    assert!(parsed["error"]["code"].is_string(), "error code should be a string");
}

#[test]
fn proxy_failure_formats_openai_error_for_responses_subresource() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = echo_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let request = format!(
        "GET /v1/responses/resp_123 HTTP/1.1\r\n\
         Host: localhost:{proxy_port}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "Content-Type should be application/json"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("response body should be valid JSON");
    assert_eq!(
        parsed["error"]["type"], "server_error",
        "OpenAI error type should be server_error for 502"
    );
    assert!(parsed["error"]["param"].is_null(), "param should be null");
}

#[test]
fn proxy_failure_does_not_format_openai_error_for_unclassified_request() {
    let dead_port = free_port();
    let proxy_port = free_port();

    let yaml = continue_yaml(proxy_port, dead_port);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"unrelated_api":"data"}"#;
    let raw = http_send(proxy.addr(), &json_post("/other/endpoint", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "proxy failure on unreachable upstream should return 502"
    );

    let body_str = parse_body(&raw);
    let parsed: serde_json::Value =
        serde_json::from_str(&body_str).expect("proxy error should be valid JSON (RFC 9457)");
    assert!(
        parsed.get("error").and_then(|e| e.get("type")).is_none(),
        "unclassified request should not receive OpenAI formatted error envelope"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// YAML config for routing by classified format using header matching.
fn routing_yaml(proxy_port: u16, responses_port: u16, chat_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            headers:
              x-praxis-ai-format: "openai_responses"
            cluster: "responses"
          - path_prefix: "/"
            headers:
              x-praxis-ai-format: "openai_chat_completions"
            cluster: "chat"
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "responses"
            endpoints:
              - "127.0.0.1:{responses_port}"
          - name: "chat"
            endpoints:
              - "127.0.0.1:{chat_port}"
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config with on_invalid: continue and a catch-all default backend.
fn continue_yaml(proxy_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config with on_invalid: reject.
fn reject_yaml(proxy_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
        on_invalid: reject
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config that routes all traffic to a header-echo backend.
fn header_echo_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config that routes all traffic to an echo (body) backend.
fn echo_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config for branch-based routing using filter results.
fn branch_yaml(proxy_port: u16, responses_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
        branch_chains:
          - name: responses_branch
            on_result:
              filter: openai_responses_format
              key: format
              result: openai_responses
            rejoin: shared_load_balancer
            chains:
              - name: responses_chain
                filters:
                  - filter: router
                    routes:
                      - path_prefix: "/"
                        cluster: "responses"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - name: shared_load_balancer
        filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
          - name: "responses"
            endpoints:
              - "127.0.0.1:{responses_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config using the validator's ordinary fail-closed policy.
fn unconditioned_background_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_request
      - filter: openai_responses_validate
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config proving that only the logical provider declaration grants the
/// background capability, even when the endpoint uses OpenAI's hostname.
fn non_openai_api_hostname_background_yaml(proxy_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_request
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "not-openai"
      - filter: openai_responses_validate
        conditions:
          - unless:
              bound_upstream:
                application_provider: openai
      - filter: load_balancer
        cluster_source: bound_upstream
        clusters:
          - name: "not-openai"
            http:
              application_provider: "foo"
            endpoints:
              - "api.openai.com:443"
          - name: "openai-capable"
            http:
              application_provider: "openai"
            endpoints:
              - "api.openai.com:443"
"#
    )
}

/// YAML config routing create and lifecycle operations to one OpenAI owner.
#[cfg(feature = "store-sqlite")]
fn openai_background_lifecycle_yaml(proxy_port: u16, backend_port: u16, database_url: &str) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_request
      - filter: state_owner
        mode: single_tenant
        tenant_id: default
      - filter: router
        routes:
          - path_prefix: "/v1/responses"
            cluster: "openai"
      - filter: openai_responses_validate
        conditions:
          - unless:
              bound_upstream:
                application_provider: openai
      - filter: openai_response_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        conditions:
          - unless:
              bound_upstream:
                application_provider: openai
      - filter: load_balancer
        cluster_source: bound_upstream
        clusters:
          - name: "openai"
            http:
              application_provider: "openai"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config proving bound policy rejects before store and IRR execution.
#[cfg(feature = "store-sqlite")]
fn managed_background_yaml(proxy_port: u16, backend_port: u16, database_url: &str) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_request
      - filter: state_owner
        mode: single_tenant
        tenant_id: default
      - filter: router
        routes:
          - path_prefix: "/v1/responses"
            cluster: "inference"
      - filter: openai_responses_validate
        conditions:
          - unless:
              bound_upstream:
                application_provider: openai
      - filter: openai_response_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        conditions:
          - unless:
              bound_upstream:
                application_provider: openai
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 1
        steps:
          - name: inference
            filters:
              - filter: load_balancer
                cluster_source: bound_upstream
                clusters:
                  - name: "inference"
                    http:
                      application_provider: "vllm"
                    endpoints:
                      - "127.0.0.1:{backend_port}"
                  - name: "openai"
                    http:
                      application_provider: "openai"
                    endpoints:
                      - "127.0.0.1:{backend_port}"
            on_result:
              - default: true
                done: true
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config proving that local retrieval ownership blocks background mode.
#[cfg(feature = "store-sqlite")]
fn stored_background_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_request
      - filter: state_owner
        mode: single_tenant
        tenant_id: default
      - filter: openai_response_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "openai"
      - filter: load_balancer
        clusters:
          - name: "openai"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config for routing by promoted model header.
fn model_routing_yaml(proxy_port: u16, model_a_port: u16, model_b_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
      - filter: router
        routes:
          - path_prefix: "/"
            headers:
              x-praxis-ai-model: "model-a"
            cluster: "model-a"
          - path_prefix: "/"
            headers:
              x-praxis-ai-model: "model-b"
            cluster: "model-b"
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "model-a"
            endpoints:
              - "127.0.0.1:{model_a_port}"
          - name: "model-b"
            endpoints:
              - "127.0.0.1:{model_b_port}"
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config for routing by promoted stream header.
fn stream_routing_yaml(proxy_port: u16, stream_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
      - filter: router
        routes:
          - path_prefix: "/"
            headers:
              x-praxis-ai-stream: "true"
            cluster: "stream"
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "stream"
            endpoints:
              - "127.0.0.1:{stream_port}"
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}
