// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `openai_mcp_tool_resolve` filter.

use praxis_core::config::Config;
use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, TempSqlite, free_port, http_get, http_send, json_post,
    parse_body, parse_status, start_backend_with_shutdown, start_echo_backend, start_mcp_mock_server_with_config,
    start_proxy,
};

// =============================================================================
// Pass-Through (no MCP tools)
// =============================================================================

#[test]
fn request_without_mcp_tools_passes_through() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "non-MCP request should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "non-MCP request should reach inference backend"
    );
}

#[test]
fn request_without_tools_passes_through() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request without tools should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "request without tools should reach inference backend"
    );
}

// =============================================================================
// SSRF Rejection
// =============================================================================

#[test]
fn mcp_loopback_url_rejected_as_ssrf() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"evil","server_url":"http://127.0.0.1/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "loopback MCP URL should be rejected");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF: {response_body}"
    );
}

#[test]
fn mcp_metadata_url_rejected_as_ssrf() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"meta","server_url":"http://169.254.169.254/latest/meta-data/","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "metadata URL should be rejected");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF: {response_body}"
    );
}

#[test]
fn mcp_localhost_url_rejected_as_ssrf() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"local","server_url":"http://localhost/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "localhost MCP URL should be rejected");
}

// =============================================================================
// Connection Failure
// =============================================================================

#[test]
fn mcp_unreachable_server_returns_502() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"dead","server_url":"http://192.0.2.1:{dead_port}/mcp","allowed_tools":["x"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 502, "unreachable MCP server should produce 502");
}

#[test]
fn streaming_mcp_unreachable_server_emits_failed_event() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","stream":true,"tools":[{{"type":"mcp","server_label":"dead","server_url":"http://192.0.2.1:{dead_port}/mcp","allowed_tools":["x"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming failure should be delivered through the Responses event lifecycle"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("event: response.mcp_list_tools.failed"),
        "stream should expose the MCP listing failure: {response_body}"
    );
    assert!(
        response_body.contains(r#""type":"response.failed""#),
        "stream should terminate with response.failed: {response_body}"
    );
    assert!(
        response_body.contains(r#""server_label":"dead""#),
        "failed MCP output item should identify the server: {response_body}"
    );
    // The lifecycle snapshots echo the requested model. In the shipped pipeline
    // the format filter promotes it to `openai_responses_format.model` metadata,
    // so this exercises the metadata branch (not the ResponsesState fallback the
    // unit tests cover).
    assert!(
        response_body.contains(r#""model":"gpt-4.1""#),
        "failure lifecycle should echo the requested model: {response_body}"
    );
}

// =============================================================================
// Full-flow: streaming discovery failure persists to the store
//
// These exercise the header-phase `TerminalResponse` end to end through the
// store pipeline (format -> validate -> tool_parse -> store -> stream_events ->
// mcp_tool_resolve). A streaming `tools/list` runtime failure is emitted as a
// 200 SSE lifecycle; because it is a `TerminalResponse` (not a `Reject`), the
// response phase runs `openai_stream_events` (which accumulates the terminal
// `response.failed`) and `openai_response_store` (which persists it) so a caller
// can later retrieve the failed response -- all without contacting the backend.
// =============================================================================

/// Outcome of driving a streaming MCP discovery failure through the store
/// pipeline: the SSE lifecycle, the failed response id, the follow-up GET
/// result, and how many times the inference backend was contacted.
struct FailureRetrieval {
    /// The SSE body returned to the client for the POST.
    sse: String,
    /// The response resource carried by the terminal `response.failed` event.
    failed_response: serde_json::Value,
    /// The response id extracted from the terminal `response.failed` event.
    resp_id: String,
    /// HTTP status of the follow-up `GET /v1/responses/{id}`.
    get_status: u16,
    /// Body of the follow-up GET.
    get_body: String,
    /// Number of requests the inference backend received.
    backend_hits: usize,
}

/// Extract the response resource carried by the terminal `response.failed`
/// event.
fn extract_failed_response(sse: &str) -> serde_json::Value {
    let mut current_event = None;
    for line in sse.lines() {
        if let Some(event) = line.strip_prefix("event: ") {
            current_event = Some(event.to_owned());
        } else if let Some(data) = line.strip_prefix("data: ") {
            if current_event.as_deref() == Some("response.failed") {
                let value: serde_json::Value =
                    serde_json::from_str(data).expect("response.failed data should be valid JSON");
                return value["response"].clone();
            }
            current_event = None;
        }
    }
    panic!("no response.failed event found in SSE stream:\n{sse}");
}

/// Assert a synthesized/persisted failure resource carries the full failure
/// detail: `status: "failed"`, an error object, and the failed `mcp_list_tools`
/// output item. `expect_store` pins the effective storage flag on the snapshot.
fn assert_failed_resource_fidelity(resource: &serde_json::Value, expect_store: bool) {
    assert_eq!(resource["status"], "failed", "resource records the discovery failure");
    assert_eq!(
        resource["store"], expect_store,
        "snapshot echoes the effective store flag"
    );
    assert_eq!(
        resource["error"]["code"], "server_error",
        "failed resource carries the runtime error code"
    );
    assert!(
        resource["error"]["message"].is_string(),
        "failed resource carries an error message: {resource}"
    );
    let item = &resource["output"][0];
    assert_eq!(item["type"], "mcp_list_tools", "output item is the failed listing");
    assert_eq!(item["server_label"], "weather", "output item identifies the server");
    assert!(
        item["tools"].as_array().is_some_and(|tools| tools.is_empty()),
        "no tools resolved on a failed listing: {item}"
    );
    assert!(
        item["error"].is_string(),
        "failed listing item carries an error: {item}"
    );
}

/// POST a streaming request whose MCP `tools/list` fails against an unreachable
/// server, then GET the resulting response id. `extra_fields` is spliced into
/// the request JSON right after `"stream":true` (e.g. `,"store":false` or
/// `,"temperature":0.2`); pass `""` to add nothing.
fn run_streaming_mcp_failure(test_name: &str, extra_fields: &str) -> FailureRetrieval {
    run_streaming_mcp_failure_with_yaml(test_name, extra_fields, resolve_yaml_full_flow_store)
}

/// Same as [`run_streaming_mcp_failure`] but with the pipeline YAML supplied by
/// `build_yaml`, so tests can exercise different filter orderings (e.g.
/// `openai_stream_events` before vs. after `openai_mcp_tool_resolve`).
fn run_streaming_mcp_failure_with_yaml(
    test_name: &str,
    extra_fields: &str,
    build_yaml: fn(u16, u16, &str, u64) -> String,
) -> FailureRetrieval {
    // A capturing backend lets us prove the inference cluster is never contacted:
    // the failure terminates in the request phase before the router runs.
    let backend = StatefulCapturingBackend::new(vec![(200, "inference".to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let dead_port = free_port();
    let db = TempSqlite::new(test_name);

    let yaml = build_yaml(proxy_port, backend.port(), db.url(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    // 192.0.2.0/24 is TEST-NET-1 (RFC 5737): guaranteed unreachable, so the
    // discovery attempt reaches runtime I/O and fails (a runtime failure), not a
    // local request-policy failure like SSRF.
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","stream":true{extra_fields},"tools":[{{"type":"mcp","server_label":"weather","server_url":"http://192.0.2.1:{dead_port}/mcp","allowed_tools":["x"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));
    assert_eq!(
        parse_status(&raw),
        200,
        "streaming MCP failure must surface as a 200 SSE lifecycle"
    );
    let sse = parse_body(&raw);
    let failed_response = extract_failed_response(&sse);
    let resp_id = failed_response["id"]
        .as_str()
        .expect("response.failed should carry a response id")
        .to_owned();

    let (get_status, get_body) = http_get(proxy.addr(), &format!("/v1/responses/{resp_id}"), None);
    let backend_hits = backend.requests().len();

    drop(proxy);
    FailureRetrieval {
        sse,
        failed_response,
        resp_id,
        get_status,
        get_body,
        backend_hits,
    }
}

/// Assert the SSE lifecycle carries the failure, the terminal snapshot has full
/// fidelity (with the expected store flag), and the backend stayed cold.
fn assert_failed_lifecycle(outcome: &FailureRetrieval, expect_store: bool) {
    assert!(
        outcome.sse.contains("event: response.mcp_list_tools.failed"),
        "stream should expose the MCP listing failure: {}",
        outcome.sse
    );
    assert!(
        outcome.sse.contains(r#""type":"response.failed""#),
        "stream should terminate with response.failed: {}",
        outcome.sse
    );
    assert!(
        outcome.resp_id.starts_with("resp_"),
        "failure lifecycle should carry a response id: {}",
        outcome.resp_id
    );
    // The terminal snapshot delivered over the wire pins the failure detail and
    // the effective store flag, independent of what the store later persists.
    assert_failed_resource_fidelity(&outcome.failed_response, expect_store);
    assert_eq!(
        outcome.backend_hits, 0,
        "the inference backend must never be contacted for a local discovery failure"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_is_retrievable_when_store_omitted() {
    let outcome = run_streaming_mcp_failure("mcp_fail_store_omitted", "");
    assert_failed_lifecycle(&outcome, true);

    assert_eq!(
        outcome.get_status, 200,
        "an omitted store defaults to persisted, so the failed response is retrievable"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");
    assert_eq!(parsed["id"], outcome.resp_id, "retrieved id should match the stream's");
    // The persisted resource must round-trip the full failure detail, not just a
    // `status: "failed"` marker, so a caller can inspect what went wrong.
    assert_failed_resource_fidelity(&parsed, true);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_is_retrievable_when_store_true() {
    let outcome = run_streaming_mcp_failure("mcp_fail_store_true", r#","store":true"#);
    assert_failed_lifecycle(&outcome, true);

    assert_eq!(
        outcome.get_status, 200,
        "an explicit store:true persists the failed response for retrieval"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");
    assert_eq!(parsed["id"], outcome.resp_id, "retrieved id should match the stream's");
    assert_failed_resource_fidelity(&parsed, true);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_not_retrievable_when_store_false() {
    let outcome = run_streaming_mcp_failure("mcp_fail_store_false", r#","store":false"#);
    // The over-the-wire snapshot still carries full failure detail, but with
    // `store: false` so a caller knows it was not persisted.
    assert_failed_lifecycle(&outcome, false);

    assert_eq!(
        outcome.get_status, 404,
        "store:false must not persist the failed response, so retrieval is a miss"
    );
}

/// Assert a resource echoes the non-default request options spliced by
/// [`streaming_list_failure_persists_non_default_request_options`].
fn assert_non_default_options_echoed(resource: &serde_json::Value) {
    assert_eq!(resource["temperature"], 0.2, "temperature echoed: {resource}");
    assert_eq!(resource["top_p"], 0.5, "top_p echoed: {resource}");
    assert_eq!(
        resource["parallel_tool_calls"], false,
        "parallel_tool_calls echoed: {resource}"
    );
    assert_eq!(
        resource["max_output_tokens"], 256,
        "max_output_tokens echoed: {resource}"
    );
    assert_eq!(resource["instructions"], "be terse", "instructions echoed: {resource}");
    assert_eq!(resource["truncation"], "auto", "truncation echoed: {resource}");
    assert_eq!(resource["tool_choice"], "required", "tool_choice echoed: {resource}");
    assert_eq!(
        resource["metadata"],
        serde_json::json!({"trace": "abc123"}),
        "metadata echoed: {resource}"
    );
}

/// Non-default request parameters (temperature, top_p, metadata, tool_choice,
/// ...) must survive into both the streamed failure snapshot and the persisted
/// resource, so a caller who set them does not retrieve a failure that silently
/// claims the API defaults.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_persists_non_default_request_options() {
    let outcome = run_streaming_mcp_failure(
        "mcp_fail_non_default_opts",
        r#","temperature":0.2,"top_p":0.5,"parallel_tool_calls":false,"max_output_tokens":256,"instructions":"be terse","truncation":"auto","tool_choice":"required","metadata":{"trace":"abc123"}"#,
    );
    assert_failed_lifecycle(&outcome, true);

    assert_eq!(
        outcome.get_status, 200,
        "an omitted store defaults to persisted, so the failed response is retrievable"
    );
    let persisted: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");

    // Both the over-the-wire snapshot and the persisted resource must echo the
    // caller's parameters verbatim, not substitute API defaults.
    assert_non_default_options_echoed(&outcome.failed_response);
    assert_non_default_options_echoed(&persisted);
}

/// Regression for the pipeline-ordering persistence gap in `agentic-loop.yaml`,
/// where `openai_stream_events` runs *after* `openai_mcp_tool_resolve` (nested in
/// the iterative_request_router). The resolver's short-circuiting
/// `TerminalResponse` means stream_events never runs on the response phase, so it
/// never accumulates the failure into `ResponsesState.response_object`. Unless the
/// resolver writes that state directly when building the terminal failure, the
/// store sees a null `response_object` and skips persistence -- and this GET would
/// 404 despite `store` defaulting to enabled. Proves the failed response is
/// retrievable regardless of where `openai_stream_events` sits relative to the
/// resolver.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_persists_when_stream_events_after_resolve() {
    let outcome = run_streaming_mcp_failure_with_yaml(
        "mcp_fail_stream_events_after",
        "",
        resolve_yaml_store_stream_events_after_resolve,
    );
    assert_failed_lifecycle(&outcome, true);

    assert_eq!(
        outcome.get_status, 200,
        "the failed response must persist even when openai_stream_events runs after the resolver"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");
    assert_eq!(parsed["id"], outcome.resp_id, "retrieved id should match the stream's");
    assert_failed_resource_fidelity(&parsed, true);
}

/// Regression for a persistence gap in the store-without-validate/rehydrate
/// configuration: when no filter builds a `ResponsesState` before the resolver,
/// the terminal-failure build must CREATE the state (not just mutate an existing
/// one) so `openai_response_store` has a `response_object` to persist. With a
/// `get_mut`-only write this GET would 404 despite `store` defaulting to enabled;
/// `get_or_insert_with` makes the failed response retrievable regardless of which
/// upstream filters ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_persists_without_validate_or_rehydrate() {
    let outcome = run_streaming_mcp_failure_with_yaml(
        "mcp_fail_no_validate_rehydrate",
        "",
        resolve_yaml_store_no_validate_rehydrate,
    );
    assert_failed_lifecycle(&outcome, true);

    assert_eq!(
        outcome.get_status, 200,
        "the failed response must persist even without validate/rehydrate to pre-build state"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");
    assert_eq!(parsed["id"], outcome.resp_id, "retrieved id should match the stream's");
    assert_failed_resource_fidelity(&parsed, true);
}

/// The no-validate/rehydrate configuration must still echo the caller's real,
/// non-default request options into both the streamed and persisted failure
/// resource. Because nothing builds a `ResponsesState.request_body` up front, the
/// options can only come from the size-bounded capture the resolver takes from
/// the request body during the pre-read; without it the failure would silently
/// claim API defaults even though the caller set otherwise.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_persists_non_default_options_without_validate_or_rehydrate() {
    let outcome = run_streaming_mcp_failure_with_yaml(
        "mcp_fail_non_default_no_validate",
        r#","temperature":0.2,"top_p":0.5,"parallel_tool_calls":false,"max_output_tokens":256,"instructions":"be terse","truncation":"auto","tool_choice":"required","metadata":{"trace":"abc123"}"#,
        resolve_yaml_store_no_validate_rehydrate,
    );
    assert_failed_lifecycle(&outcome, true);

    assert_eq!(
        outcome.get_status, 200,
        "the failed response must persist without validate/rehydrate to pre-build state"
    );
    let persisted: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");

    // Both the over-the-wire snapshot and the persisted resource echo the
    // caller's options -- captured from the body, not defaulted -- even though no
    // filter built a `ResponsesState.request_body`.
    assert_non_default_options_echoed(&outcome.failed_response);
    assert_non_default_options_echoed(&persisted);
}

/// An oversized-but-valid request (a multi-hundred-KB `instructions`) must not be
/// amplified across the three SSE snapshots and the persisted resource. The
/// resolver's size-bounded capture drops the oversized `instructions` (echoing
/// the API default) while retaining the small options, so the whole SSE stream
/// stays a small fraction of the request that produced it -- a 64 MiB request
/// cannot be turned into hundreds of MiB of synthesized response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_list_failure_bounds_oversized_request_options() {
    let huge_instructions = "x".repeat(300 * 1024);
    let extra_fields = format!(r#","temperature":0.2,"instructions":"{huge_instructions}""#);
    let outcome = run_streaming_mcp_failure_with_yaml(
        "mcp_fail_oversized_opts",
        &extra_fields,
        resolve_yaml_store_no_validate_rehydrate,
    );
    assert_failed_lifecycle(&outcome, true);

    // The small option still echoes; the oversized one is dropped to its default.
    assert_eq!(outcome.failed_response["temperature"], 0.2, "small option echoes");
    assert_eq!(
        outcome.failed_response["instructions"],
        serde_json::Value::Null,
        "oversized instructions is dropped, not echoed"
    );
    // If the oversized field had been echoed it would appear ~3x in the stream;
    // assert the whole SSE body stays far below the single request's size.
    assert!(
        outcome.sse.len() < 100 * 1024,
        "oversized options must not amplify into the SSE body (got {} bytes)",
        outcome.sse.len()
    );

    assert_eq!(outcome.get_status, 200, "the bounded failure still persists");
    let persisted: serde_json::Value =
        serde_json::from_str(&outcome.get_body).expect("retrieved response should be valid JSON");
    assert_eq!(persisted["temperature"], 0.2, "persisted small option echoes");
    assert_eq!(
        persisted["instructions"],
        serde_json::Value::Null,
        "persisted oversized instructions is dropped, not echoed"
    );
}

/// A streaming request whose MCP `server_url` is blocked by SSRF policy must
/// retain a genuine HTTP error -- never the 200 SSE lifecycle reserved for
/// runtime `tools/list` failures against a validated-safe target. This guards
/// the security-critical direction end to end through the store pipeline: a
/// pipeline-ordering or classification regression that leaked a policy failure
/// as a fake 200 success would be caught here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_local_policy_failure_retains_http_error() {
    let backend = StatefulCapturingBackend::new(vec![(200, "inference".to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("mcp_stream_ssrf");

    // The MCP filter's own `allow_loopback` defaults to false (independent of the
    // global allow_private_endpoints the loopback inference upstream needs), so a
    // loopback MCP URL is blocked as SSRF before any runtime I/O -- a local policy
    // failure, not a runtime one.
    let yaml = resolve_yaml_full_flow_store(proxy_port, backend.port(), db.url(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","stream":true,"tools":[{"type":"mcp","server_label":"evil","server_url":"http://127.0.0.1/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "a streaming SSRF failure must stay a hard HTTP error, not a 200 SSE lifecycle"
    );
    let sse = parse_body(&raw);
    assert!(
        !sse.contains("event: response.mcp_list_tools.failed") && !sse.contains(r#""type":"response.failed""#),
        "SSRF must not emit the discovery lifecycle: {sse}"
    );
    assert!(
        backend.requests().is_empty(),
        "SSRF rejection must not contact the inference backend"
    );
}

// =============================================================================
// MCPToolFilter Object: read_only accepted
// =============================================================================

#[test]
fn mcp_read_only_filter_accepted() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"srv","server_url":"http://10.0.0.1/mcp","allowed_tools":{"read_only":true}}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    let status = parse_status(&raw);
    assert_ne!(status, 400, "read_only filter should be accepted, not rejected as 400");
}

// =============================================================================
// Non-Responses Path
// =============================================================================

#[test]
fn non_responses_path_passes_through() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"s","server_url":"http://127.0.0.1/mcp"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "non-Responses path should not trigger MCP resolution"
    );
    assert_eq!(
        parse_body(&raw),
        "inference",
        "non-Responses path should pass through to backend"
    );
}

// =============================================================================
// Body Preservation with openai_responses_proxy (get_or_insert_with regression)
// =============================================================================

#[test]
fn body_preserved_through_openai_responses_proxy_with_function_tools() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_with_openai_responses_proxy_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    assert_eq!(parsed["model"], "gpt-4.1", "model should be preserved");
    assert_eq!(parsed["input"], "test", "input should be preserved");
    assert!(parsed["tools"].is_array(), "tools array should be preserved");
}

#[test]
fn body_preserved_through_openai_responses_proxy_without_tools() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_with_openai_responses_proxy_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    assert_eq!(parsed["model"], "gpt-4.1", "model should be preserved");
    assert_eq!(parsed["input"], "Hello, world!", "input should be preserved");
}

// =============================================================================
// Authorization + SSRF Interaction
// =============================================================================

#[test]
fn authorization_does_not_bypass_ssrf_check() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"auth","server_url":"http://127.0.0.1/mcp","authorization":"tok_secret","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "SSRF should reject even with authorization");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF, not auth failure: {response_body}"
    );
}

#[test]
fn authorization_with_unreachable_server_returns_502() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"auth","server_url":"http://192.0.2.1:{dead_port}/mcp","authorization":"tok_secret","headers":{{"x-custom":"val"}},"allowed_tools":["x"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        502,
        "unreachable server with auth+headers should produce 502"
    );
}

// =============================================================================
// MCPToolFilter Object Form (tool_names accepted)
// =============================================================================

#[test]
fn mcp_tool_names_filter_object_accepted() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();
    let dead_port = free_port();

    let yaml = resolve_yaml_with_timeout(proxy_port, backend_guard.port(), 500);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"srv","server_url":"http://192.0.2.1:{dead_port}/mcp","allowed_tools":{{"tool_names":["get_weather"]}}}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        502,
        "tool_names filter object should be accepted (502 = attempted resolution, not 400)"
    );
}

// =============================================================================
// Mock MCP Server: tools/list and mcp_tool_map populated
// =============================================================================

#[test]
fn mcp_tools_list_succeeds_against_mock_server() {
    let mcp_config = McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather").with_description("Get weather"),
            McpToolFixture::new("create_event"),
        ],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"weather","server_url":"{mcp_url}","allowed_tools":["get_weather"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "MCP resolution should succeed");

    assert!(
        mcp_server.method_count("initialize") >= 1,
        "should have called initialize on MCP server"
    );
    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "should have called tools/list on MCP server"
    );
}

#[test]
fn mcp_too_many_tools_rejected() {
    let tools: Vec<McpToolFixture> = (0..5).map(|i| McpToolFixture::new(format!("tool_{i}"))).collect();
    let mcp_config = McpMockConfig {
        tools,
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback_with_max_tools(proxy_port, backend_guard.port(), 2);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"many","server_url":"{mcp_url}","allowed_tools":["tool_0"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 502, "too many tools should produce 502");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("too many tools"),
        "rejection should mention too many tools: {response_body}"
    );
}

#[test]
fn mcp_mock_server_echoes_resolved_body() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("calc")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback_with_proxy(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"math","server_url":"{mcp_url}","allowed_tools":["calc"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    assert_eq!(parsed["model"], "gpt-4.1", "model should be preserved");
    let input = &parsed["input"];
    let has_test_content = if input.is_string() {
        input.as_str() == Some("test")
    } else if let Some(arr) = input.as_array() {
        arr.iter()
            .any(|item| item.get("content").and_then(|c| c.as_str()) == Some("test"))
    } else {
        false
    };
    assert!(has_test_content, "input should contain 'test': {input}");

    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "should have called tools/list"
    );
}

/// Verify end-to-end that `type: "mcp"` tools are rewritten to
/// `type: "function"` before being forwarded upstream.
#[test]
fn mcp_tools_rewritten_to_function_type() {
    let mcp_config = McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather").with_description("Get weather"),
            McpToolFixture::new("set_alarm"),
        ],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback_with_proxy(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"function","name":"calc"}},{{"type":"mcp","server_label":"home","server_url":"{mcp_url}","allowed_tools":["get_weather"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();

    let tools = parsed["tools"].as_array().expect("tools should be an array");
    assert_eq!(tools.len(), 2, "should have 2 tools (1 function + 1 rewritten MCP)");

    assert_eq!(tools[0]["type"], "function", "first tool type unchanged");
    assert_eq!(tools[0]["name"], "calc", "first tool name unchanged");

    assert_eq!(tools[1]["type"], "function", "MCP tool rewritten to function");
    assert_eq!(tools[1]["name"], "home__get_weather", "MCP tool name prefixed");
    assert_eq!(tools[1]["description"], "Get weather", "description preserved");
    assert!(tools[1]["parameters"].is_object(), "parameters added");

    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "should have called tools/list"
    );
}

#[test]
fn mcp_invalid_authorization_rejected() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("tool_a")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"auth","server_url":"{mcp_url}","authorization":"tok\nbad","allowed_tools":["tool_a"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        502,
        "invalid authorization chars should produce 502"
    );
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("invalid HTTP header"),
        "rejection should mention invalid header: {response_body}"
    );
}

// =============================================================================
// Duplicate server_label Rejection
// =============================================================================

#[test]
fn duplicate_server_label_rejected_before_callout() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("tool_a")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"dup","server_url":"{mcp_url}","authorization":"tok_a","allowed_tools":["tool_a"]}},{{"type":"mcp","server_label":"dup","server_url":"{mcp_url}","authorization":"tok_b","allowed_tools":["tool_a"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 400, "duplicate server_label must produce 400");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("duplicate server_label"),
        "rejection should mention duplicate server_label: {response_body}"
    );

    assert_eq!(
        mcp_server.method_count("tools/list"),
        0,
        "no tools/list calls should be made for duplicate labels"
    );
    assert_eq!(
        mcp_server.method_count("initialize"),
        0,
        "no initialize calls should be made for duplicate labels"
    );
}

// =============================================================================
// Connector Resolution
// =============================================================================

#[test]
fn connector_resolves_to_mock_mcp_server() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("search").with_description("Search files")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let connectors = format!("        connectors:\n          - id: corp_drive\n            server_url: {mcp_url}");
    let yaml = resolve_yaml_loopback_with_connectors_and_proxy(proxy_port, backend_guard.port(), &connectors);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"drive","connector_id":"corp_drive","allowed_tools":["search"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "connector resolution should succeed");

    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    let tools = parsed["tools"].as_array().expect("tools should be an array");
    assert_eq!(tools.len(), 1, "should have 1 resolved function tool");
    assert_eq!(tools[0]["type"], "function", "MCP tool rewritten to function");
    assert_eq!(tools[0]["name"], "drive__search", "name prefixed with server_label");

    assert!(
        !echoed.contains("connector_id"),
        "connector_id must not appear in rewritten body"
    );
    assert!(
        !echoed.contains(&mcp_url),
        "resolved server_url must not appear in rewritten body"
    );

    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "should have called tools/list on the resolved MCP server"
    );
}

#[test]
fn unknown_connector_rejected_before_callout() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("tool_a")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let yaml = resolve_yaml_loopback(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"s","connector_id":"nonexistent"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 400, "unknown connector should be 400");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("unknown connector_id"),
        "response: {response_body}"
    );

    assert_eq!(
        mcp_server.method_count("initialize"),
        0,
        "no MCP calls should be made for unknown connectors"
    );
}

#[test]
fn connector_with_authorization_forwarded() {
    let mcp_config = McpMockConfig {
        tools: vec![McpToolFixture::new("tool_a")],
        ..McpMockConfig::default()
    };
    let mcp_server = start_mcp_mock_server_with_config(mcp_config);
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp_server.port());
    let connectors = format!("        connectors:\n          - id: authed\n            server_url: {mcp_url}");
    let yaml = resolve_yaml_loopback_with_connectors_and_proxy(proxy_port, backend_guard.port(), &connectors);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"s","connector_id":"authed","authorization":"Bearer tok_test","allowed_tools":["tool_a"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "connector with auth should succeed");

    assert!(
        mcp_server.method_count("tools/list") >= 1,
        "should have called tools/list with forwarded auth"
    );

    let requests = mcp_server.received_requests();
    let has_auth = requests.iter().any(|r| {
        r.headers
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case("authorization") && value.contains("tok_test"))
    });
    assert!(has_auth, "authorization token should be forwarded to MCP server");
}

#[test]
fn direct_url_alongside_connector_both_resolved() {
    let mcp_config_a = McpMockConfig {
        tools: vec![McpToolFixture::new("tool_a")],
        ..McpMockConfig::default()
    };
    let mcp_server_a = start_mcp_mock_server_with_config(mcp_config_a);

    let mcp_config_b = McpMockConfig {
        tools: vec![McpToolFixture::new("tool_b")],
        ..McpMockConfig::default()
    };
    let mcp_server_b = start_mcp_mock_server_with_config(mcp_config_b);

    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let mcp_url_a = format!("http://127.0.0.1:{}/mcp", mcp_server_a.port());
    let connectors = format!("        connectors:\n          - id: c1\n            server_url: {mcp_url_a}");
    let yaml = resolve_yaml_loopback_with_connectors_and_proxy(proxy_port, backend_guard.port(), &connectors);
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let mcp_url_b = format!("http://127.0.0.1:{}/mcp", mcp_server_b.port());
    let body = format!(
        r#"{{"model":"gpt-4.1","input":"test","tools":[{{"type":"mcp","server_label":"conn","connector_id":"c1","allowed_tools":["tool_a"]}},{{"type":"mcp","server_label":"direct","server_url":"{mcp_url_b}","allowed_tools":["tool_b"]}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "mixed connector + direct should succeed");

    let echoed = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&echoed).unwrap();
    let tools = parsed["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 2, "should have 2 resolved function tools");

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"conn__tool_a"), "connector tool resolved");
    assert!(names.contains(&"direct__tool_b"), "direct URL tool resolved");
}

// =============================================================================
// YAML Helpers
// =============================================================================

fn resolve_yaml(proxy_port: u16, backend_port: u16) -> String {
    resolve_yaml_with_timeout(proxy_port, backend_port, 5000)
}

/// Pipeline mirroring the relevant `full-flow.yaml` ordering for store
/// retrieval: `openai_response_store` and `openai_stream_events` run before
/// `openai_mcp_tool_resolve`, so the header-phase `TerminalResponse` triggers
/// their response-phase persistence of the synthesized failure.
fn resolve_yaml_full_flow_store(proxy_port: u16, backend_port: u16, db_url: &str, timeout_ms: u64) -> String {
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
      - filter: openai_responses_validate
      - filter: openai_tool_parse
      - filter: openai_response_store
        backend: sqlite
        database_url: "{db_url}"
        responses_table: openai_responses
        conversations_table: openai_conversations
      - filter: openai_stream_events
      - filter: openai_mcp_tool_resolve
        timeout_ms: {timeout_ms}
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

/// Pipeline mirroring `agentic-loop.yaml`'s ordering, where `openai_stream_events`
/// runs *after* `openai_mcp_tool_resolve` (there, nested inside the
/// iterative_request_router that follows the resolver). The resolver's
/// header-phase `TerminalResponse` short-circuits before `openai_stream_events`
/// ever runs, so stream_events never accumulates the failure into
/// `ResponsesState.response_object`. Persistence must therefore come from the
/// resolver writing that state directly when it builds the terminal failure;
/// otherwise the store sees a null `response_object` and skips persistence.
fn resolve_yaml_store_stream_events_after_resolve(
    proxy_port: u16,
    backend_port: u16,
    db_url: &str,
    timeout_ms: u64,
) -> String {
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
      - filter: openai_responses_validate
      - filter: openai_tool_parse
      - filter: openai_response_store
        backend: sqlite
        database_url: "{db_url}"
        responses_table: openai_responses
        conversations_table: openai_conversations
      - filter: openai_responses_rehydrate
      - filter: openai_mcp_tool_resolve
        timeout_ms: {timeout_ms}
      - filter: openai_stream_events
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

/// Minimal store pipeline with `openai_response_store` + `openai_mcp_tool_resolve`
/// but WITHOUT `openai_responses_validate`/`openai_responses_rehydrate` (and no
/// `openai_stream_events`). Nothing builds a `ResponsesState` before the resolver,
/// so persistence depends entirely on the resolver *creating* the state when it
/// writes the terminal snapshot (`get_or_insert_with`, not `get_mut`). The store
/// still triggers streaming persistence -- its gating reads only
/// `openai_responses_format.*` metadata, which `openai_responses_format`
/// (always first) supplies -- and `build_record_from_state` reads only
/// `response_object`, so a default state carrying just that field suffices.
fn resolve_yaml_store_no_validate_rehydrate(
    proxy_port: u16,
    backend_port: u16,
    db_url: &str,
    timeout_ms: u64,
) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_response_store
        backend: sqlite
        database_url: "{db_url}"
        responses_table: openai_responses
        conversations_table: openai_conversations
      - filter: openai_mcp_tool_resolve
        timeout_ms: {timeout_ms}
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

fn resolve_yaml_with_timeout(proxy_port: u16, backend_port: u16, timeout_ms: u64) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        timeout_ms: {timeout_ms}
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

fn resolve_with_openai_responses_proxy_yaml(proxy_port: u16, backend_port: u16) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
      - filter: openai_responses_proxy
        name: inference
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

fn resolve_yaml_loopback(proxy_port: u16, backend_port: u16) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        allow_loopback: true
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

fn resolve_yaml_loopback_with_max_tools(proxy_port: u16, backend_port: u16, max_tools: usize) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        allow_loopback: true
        max_tools: {max_tools}
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

fn resolve_yaml_loopback_with_proxy(proxy_port: u16, backend_port: u16) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        allow_loopback: true
      - filter: openai_responses_proxy
        name: inference
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

fn resolve_yaml_loopback_with_connectors_and_proxy(
    proxy_port: u16,
    backend_port: u16,
    connectors_yaml: &str,
) -> String {
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
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
        allow_loopback: true
{connectors_yaml}
      - filter: openai_responses_proxy
        name: inference
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
