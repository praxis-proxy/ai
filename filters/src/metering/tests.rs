// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use super::*;
use crate::test_utils::{make_filter_context, make_request};

/// Build the concrete filter with a private test client.
fn build_filter(yaml: &serde_yaml::Value) -> Result<ExternalMeteringFilter, FilterError> {
    ExternalMeteringFilter::build(yaml, SubRequestClient::new(SubRequestConnector::new(1, None)))
}

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn valid_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
"#,
    )
    .unwrap();
    let filter = ExternalMeteringFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "external_metering");
}

#[test]
fn config_with_all_fields_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
timeout_seconds: 10
feature_key: "custom-tokens"
source: "my-gateway"
fail_open: false
identity_header_prefix: "x-custom-"
"#,
    )
    .unwrap();
    let filter = ExternalMeteringFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "external_metering");
}

#[test]
fn config_with_fallbacks_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
default_username: "anonymous"
default_model: "unknown"
"#,
    )
    .unwrap();
    let filter = build_filter(&yaml).unwrap();

    assert_eq!(filter.default_username.as_deref(), Some("anonymous"));
    assert_eq!(filter.default_model.as_deref(), Some("unknown"));
}

#[test]
fn config_without_fallbacks_leaves_them_unset() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
"#,
    )
    .unwrap();
    let filter = build_filter(&yaml).unwrap();

    assert!(filter.default_username.is_none());
    assert!(filter.default_model.is_none());
}

// -----------------------------------------------------------------------------
// Fallback Runtime Behavior
// -----------------------------------------------------------------------------

#[tokio::test]
async fn default_username_meters_anonymous_request() {
    // Unroutable metering URL makes the balance check fail instantly;
    // fail-open (the default) still admits and stores the metering state.
    let filter = filter_from_yaml("metering_url: \"http://127.0.0.1:1\"\ndefault_username: \"anonymous\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(
        !ctx.filter_state.is_empty(),
        "default_username must cause metering state to be stored"
    );
}

#[test]
fn default_model_is_used_when_no_model_was_captured() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\ndefault_model: \"unknown\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let ctx = make_filter_context(&req);
    let state = state_for("alice", "");

    assert_eq!(filter.resolve_report_model(&ctx, &state), "unknown");
}

#[test]
fn captured_model_wins_over_default_model() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\ndefault_model: \"unknown\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata
        .insert(META_METERING_MODEL.to_owned(), "gpt-4".to_owned());
    let state = state_for("alice", "");

    assert_eq!(filter.resolve_report_model(&ctx, &state), "gpt-4");
}

#[test]
fn empty_model_without_default_stays_empty() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let ctx = make_filter_context(&req);
    let state = state_for("alice", "");

    assert_eq!(filter.resolve_report_model(&ctx, &state), "");
}

#[test]
fn config_empty_prefix_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
identity_header_prefix: ""
"#,
    )
    .unwrap();
    let result = ExternalMeteringFilter::from_config(&yaml);
    assert!(result.is_err());
}

#[test]
fn config_missing_url_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: ""
"#,
    )
    .unwrap();
    let result = ExternalMeteringFilter::from_config(&yaml);
    assert!(result.is_err());
}

#[test]
fn config_zero_timeout_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
timeout_seconds: 0
"#,
    )
    .unwrap();
    let result = ExternalMeteringFilter::from_config(&yaml);
    assert!(result.is_err());
}

#[test]
fn config_prefix_with_invalid_header_chars_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
identity_header_prefix: "x tenant "
"#,
    )
    .unwrap();
    let result = ExternalMeteringFilter::from_config(&yaml);
    assert!(result.is_err(), "prefix with spaces must be rejected");
}

#[test]
fn config_empty_namespace_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
identity_metadata_namespace: ""
"#,
    )
    .unwrap();
    let result = ExternalMeteringFilter::from_config(&yaml);
    assert!(result.is_err(), "empty namespace must be rejected");
}

#[test]
fn config_unknown_field_fails() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
metering_url: "http://metering:8080"
unknown_field: true
"#,
    )
    .unwrap();
    let result = ExternalMeteringFilter::from_config(&yaml);
    assert!(result.is_err());
}

// -----------------------------------------------------------------------------
// Balance Response Parsing
// -----------------------------------------------------------------------------

#[test]
fn balance_response_has_access_continues() {
    let body = br#"{"hasAccess": true, "balance": 9000.0, "usage": 1000.0}"#;
    let action = parse_balance_result(body, true);
    assert!(matches!(action, FilterAction::Continue));
}

#[test]
fn balance_response_no_access_rejects() {
    let body = br#"{"hasAccess": false, "balance": 0.0, "usage": 10000.0}"#;
    let action = parse_balance_result(body, true);
    assert!(matches!(action, FilterAction::Reject(_)));
}

#[test]
fn balance_response_invalid_json_fail_open() {
    let body = b"not json";
    let action = parse_balance_result(body, true);
    assert!(matches!(action, FilterAction::Continue));
}

#[test]
fn balance_response_invalid_json_fail_closed() {
    let body = b"not json";
    let action = parse_balance_result(body, false);
    assert!(matches!(action, FilterAction::Reject(_)));
}

#[test]
fn balance_response_empty_body_fail_open() {
    let action = parse_balance_result(b"", true);
    assert!(matches!(action, FilterAction::Continue));
}

#[test]
fn balance_response_empty_body_fail_closed() {
    let action = parse_balance_result(b"", false);
    assert!(matches!(action, FilterAction::Reject(_)));
}

// -----------------------------------------------------------------------------
// URL Construction
// -----------------------------------------------------------------------------

#[test]
fn balance_url_encodes_special_chars() {
    let url = build_balance_url("http://metering:8080", "user@example.com", "inference-tokens", "gpt-4o");
    assert!(url.contains("user%40example.com"), "@ should be encoded: {url}");
    assert!(url.contains("inference-tokens"), "hyphens should not be encoded: {url}");
    assert!(url.contains("gpt-4o"), "hyphens in model should not be encoded: {url}");
}

#[test]
fn balance_url_strips_trailing_slash() {
    let url = build_balance_url("http://metering:8080/", "testuser", "tokens", "llama");
    assert!(url.starts_with("http://metering:8080/api/"));
    assert!(!url.contains("//api/"));
}

// -----------------------------------------------------------------------------
// CloudEvent Construction
// -----------------------------------------------------------------------------

#[test]
fn usage_event_has_correct_structure() {
    let state = state_for("testuser", "gpt-4");
    let tokens = TokenCounts {
        input: 100,
        output: 50,
        total: 150,
        cache_read: 80,
        cache_write: 20,
    };

    let event = build_usage_event(&event_ctx("evt-1", &state), &tokens);

    assert_eq!(event["specversion"], "1.0");
    assert_eq!(event["type"], CE_TYPE_USAGE);
    assert_eq!(event["subject"], "testuser");
    assert_eq!(event["data"]["prompt_tokens"], 100);
    assert_eq!(event["data"]["completion_tokens"], 50);
    assert_eq!(event["data"]["total_tokens"], 150);
    assert_eq!(event["data"]["cached_input_tokens"], 80);
    assert_eq!(event["data"]["cache_creation_tokens"], 20);
    assert_eq!(event["data"]["duration_ms"], 500);
    assert_eq!(event["data"]["model"], "gpt-4");
}

#[test]
fn error_event_has_correct_structure() {
    let mut state = state_for("testuser", "gpt-4");
    state.is_error = true;
    state.response_status = 500;

    let event = build_error_event(&event_ctx("evt-2", &state));

    assert_eq!(event["type"], CE_TYPE_ERROR);
    assert_eq!(event["data"]["status_code"], 500);
    assert_eq!(event["data"]["user"], "testuser");
}

// -----------------------------------------------------------------------------
// Identity Header Capture
// -----------------------------------------------------------------------------

#[test]
fn captures_tenant_headers_with_default_prefix() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers.insert("x-tenant-group", "engineering".parse().unwrap());
    req.headers.insert("x-tenant-subscription", "sub-42".parse().unwrap());
    req.headers.insert("x-tenant-model", "gpt-4".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.username, "alice");
    assert_eq!(state.group, "engineering");
    assert_eq!(state.subscription, "sub-42");
    assert_eq!(state.model, "gpt-4");
}

#[test]
fn strips_identity_and_auth_headers() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers.insert("authorization", "Bearer sk-test".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    let _state = capture_identity(&mut ctx, "x-tenant-", "identity");

    let removed: Vec<&str> = ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect();
    assert!(removed.contains(&"x-tenant-username"));
    assert!(removed.contains(&"authorization"));
    assert!(removed.contains(&"x-api-key"));
}

#[test]
fn custom_prefix_captures_correctly() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-myco-username", "bob".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    let state = capture_identity(&mut ctx, "x-myco-", "identity");

    assert_eq!(state.username, "bob");
}

#[test]
fn missing_username_returns_empty() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert!(state.username.is_empty());
}

#[test]
fn verified_identity_ignores_forged_headers_and_guard_metadata() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    // Client-forged headers alongside a valid JWT.
    req.headers
        .insert("x-tenant-subscription", "sub-forged".parse().unwrap());
    req.headers.insert("x-tenant-model", "model-forged".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    // Verified claims written by an authentication filter (unnamespaced).
    ctx.filter_metadata
        .insert("x-tenant-username".to_owned(), "alice".to_owned());
    ctx.filter_metadata
        .insert("x-tenant-group".to_owned(), "engineering".to_owned());
    // Guard-captured copies of the forged headers (namespaced).
    ctx.filter_metadata
        .insert("identity.x-tenant-subscription".to_owned(), "sub-forged".to_owned());
    ctx.filter_metadata
        .insert("identity.x-tenant-model".to_owned(), "model-forged".to_owned());

    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.username, "alice");
    assert_eq!(state.group, "engineering");
    assert!(
        state.subscription.is_empty(),
        "forged subscription must be ignored when identity is verified: {}",
        state.subscription
    );
    assert!(
        state.model.is_empty(),
        "forged model must be ignored when identity is verified: {}",
        state.model
    );
}

#[test]
fn guard_metadata_supplies_identity_without_verified_claims() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata
        .insert("identity.x-tenant-username".to_owned(), "bob".to_owned());
    ctx.filter_metadata
        .insert("identity.x-tenant-group".to_owned(), "ml".to_owned());
    ctx.filter_metadata
        .insert("identity.x-tenant-subscription".to_owned(), "sub-7".to_owned());
    ctx.filter_metadata
        .insert("identity.x-tenant-model".to_owned(), "claude-3".to_owned());

    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.username, "bob");
    assert_eq!(state.group, "ml");
    assert_eq!(state.subscription, "sub-7");
    assert_eq!(state.model, "claude-3");
}

#[test]
fn partial_verified_identity_blocks_lower_tiers() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers
        .insert("x-tenant-username", "header-mallory".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    // An auth filter that maps only the group claim, no username.
    ctx.filter_metadata
        .insert("x-tenant-group".to_owned(), "engineering".to_owned());
    // Guard-captured copy of a forged subscription.
    ctx.filter_metadata
        .insert("identity.x-tenant-subscription".to_owned(), "sub-forged".to_owned());

    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.group, "engineering", "verified group must survive");
    assert!(
        state.subscription.is_empty(),
        "guard metadata must not extend a partially verified identity: {}",
        state.subscription
    );
    assert!(
        state.username.is_empty(),
        "raw header must not extend a partially verified identity"
    );
}

#[test]
fn custom_namespace_supplies_guard_identity() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata
        .insert("tenant-meta.x-tenant-username".to_owned(), "bob".to_owned());

    let state = capture_identity(&mut ctx, "x-tenant-", "tenant-meta");

    assert_eq!(state.username, "bob", "configured namespace must resolve tier 2");
}

#[test]
fn multi_value_identity_header_is_fully_stripped() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers.append("x-tenant-username", "mallory".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.username, "alice", "first value wins during capture");
    let removed: Vec<&str> = ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect();
    assert!(
        removed.contains(&"x-tenant-username"),
        "the duplicated header name must be marked for removal (removing a name drops every value): {removed:?}"
    );
}

#[test]
fn guard_identity_blocks_raw_header_fallback() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    // Raw header not captured by the guard — anomalous, must not
    // be trusted once any metadata identity exists.
    req.headers.insert("x-tenant-subscription", "raw-sub".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata
        .insert("identity.x-tenant-username".to_owned(), "bob".to_owned());

    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.username, "bob");
    assert!(
        state.subscription.is_empty(),
        "raw header must be ignored when guard metadata identity exists: {}",
        state.subscription
    );
    let removed: Vec<&str> = ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect();
    assert!(
        removed.contains(&"x-tenant-subscription"),
        "unused identity header must still be stripped: {removed:?}"
    );
}

#[test]
fn group_falls_back_to_subscription() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers.insert("x-tenant-subscription", "sub-99".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    let state = capture_identity(&mut ctx, "x-tenant-", "identity");

    assert_eq!(state.group, "sub-99");
}

#[test]
fn strips_accept_encoding_to_keep_response_readable() {
    let mut req = make_request(http::Method::POST, "/v1/chat/completions");
    req.headers.insert("x-tenant-username", "alice".parse().unwrap());
    req.headers
        .insert("accept-encoding", "gzip, deflate, br".parse().unwrap());

    let mut ctx = make_filter_context(&req);
    let _state = capture_identity(&mut ctx, "x-tenant-", "identity");

    let removed: Vec<&str> = ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect();
    assert!(removed.contains(&"accept-encoding"));
}

// -----------------------------------------------------------------------------
// Model Extraction
// -----------------------------------------------------------------------------

#[test]
fn extracts_model_from_compact_json() {
    let body = br#"{"model":"gpt-4","messages":[]}"#;
    assert_eq!(extract_model_from_bytes(body).as_deref(), Some("gpt-4"));
}

#[test]
fn extracts_model_with_whitespace_around_colon() {
    let body = br#"{ "model" : "claude-sonnet-4" , "stream": true }"#;
    assert_eq!(extract_model_from_bytes(body).as_deref(), Some("claude-sonnet-4"));
}

#[test]
fn extracts_model_when_not_first_field() {
    let body = br#"{"stream":true,"max_tokens":100,"model":"gpt-4o-mini"}"#;
    assert_eq!(extract_model_from_bytes(body).as_deref(), Some("gpt-4o-mini"));
}

#[test]
fn extract_model_ignores_nested_decoy() {
    // A "model" key inside a message object must not win over the real
    // top-level field, regardless of field order.
    let body = br#"{"messages":[{"role":"user","content":"hi","model":"decoy"}],"model":"gpt-4"}"#;
    assert_eq!(
        extract_model_from_bytes(body).as_deref(),
        Some("gpt-4"),
        "nested decoy must not shadow the top-level model"
    );
}

#[test]
fn extract_model_ignores_decoy_inside_string_value() {
    let body = br#"{"prompt":"please say \"model\": \"decoy\" back","model":"gpt-4o"}"#;
    assert_eq!(
        extract_model_from_bytes(body).as_deref(),
        Some("gpt-4o"),
        "a quoted decoy inside a string value must be skipped"
    );
}

#[test]
fn extract_model_returns_none_when_only_nested() {
    let body = br#"{"messages":[{"model":"decoy","content":"hi"}]}"#;
    assert!(
        extract_model_from_bytes(body).is_none(),
        "a nested-only model key must not be attributed"
    );
}

#[test]
fn extract_model_returns_none_when_absent() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
    assert!(extract_model_from_bytes(body).is_none());
}

#[test]
fn extract_model_returns_none_on_truncated_chunk() {
    // A streamed first chunk may cut off mid-value.
    let body = br#"{"model":"gpt-4"#;
    assert!(extract_model_from_bytes(body).is_none());
}

#[test]
fn extract_model_returns_none_on_non_utf8() {
    let body = &[0xFF_u8, 0xFE, 0x00, 0x01];
    assert!(extract_model_from_bytes(body).is_none());
}

#[test]
fn extract_model_returns_none_on_non_string_value() {
    let body = br#"{"model":null}"#;
    assert!(extract_model_from_bytes(body).is_none());
}

// -----------------------------------------------------------------------------
// Token Metadata Reading
// -----------------------------------------------------------------------------

#[test]
fn reads_token_metadata() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata.insert("token.input".to_owned(), "150".to_owned());
    ctx.filter_metadata.insert("token.output".to_owned(), "80".to_owned());
    ctx.filter_metadata.insert("token.total".to_owned(), "230".to_owned());

    assert_eq!(read_token_meta(&ctx, META_TOKEN_INPUT), 150);
    assert_eq!(read_token_meta(&ctx, META_TOKEN_OUTPUT), 80);
    assert_eq!(read_token_meta(&ctx, META_TOKEN_TOTAL), 230);
}

#[test]
fn missing_token_metadata_returns_zero() {
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let ctx = make_filter_context(&req);

    assert_eq!(read_token_meta(&ctx, META_TOKEN_INPUT), 0);
}

// -----------------------------------------------------------------------------
// Request Lifecycle
// -----------------------------------------------------------------------------

fn filter_from_yaml(yaml: &str) -> ExternalMeteringFilter {
    let parsed: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    build_filter(&parsed).unwrap()
}

#[tokio::test]
async fn skips_metering_when_no_identity_and_no_fallback() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert!(ctx.filter_state.is_empty());
}

#[tokio::test]
async fn request_body_records_model_in_metadata() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4","stream":true}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

    assert!(matches!(action, FilterAction::Release));
    assert_eq!(
        ctx.filter_metadata.get(META_METERING_MODEL).map(String::as_str),
        Some("gpt-4")
    );
}

#[tokio::test]
async fn request_body_does_not_override_identity_model() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    ctx.filter_metadata
        .insert(META_METERING_MODEL.to_owned(), "from-header".to_owned());
    let mut body = Some(Bytes::from_static(br#"{"model":"from-body"}"#));

    let _action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();

    assert_eq!(
        ctx.filter_metadata.get(META_METERING_MODEL).map(String::as_str),
        Some("from-header")
    );
}

// -----------------------------------------------------------------------------
// Response Lifecycle
// -----------------------------------------------------------------------------

#[test]
fn response_body_is_noop_before_end_of_stream() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    store_state(&mut ctx, state_for("alice", "gpt-4"));
    let mut body = Some(Bytes::from_static(b"chunk"));

    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    // State survives so the terminal chunk can still report usage.
    assert!(!ctx.filter_state.is_empty());
}

#[test]
fn response_body_without_state_is_noop() {
    let filter = filter_from_yaml("metering_url: \"http://metering:8080\"\n");
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(&req);
    let mut body = None;

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
}

fn event_ctx<'a>(event_id: &'a str, state: &'a MeteringState) -> EventContext<'a> {
    EventContext {
        duration_ms: 500,
        event_id,
        provider: "openai",
        source: "gw",
        state,
    }
}

fn state_for(username: &str, model: &str) -> MeteringState {
    MeteringState {
        username: username.into(),
        group: "engineering".into(),
        subscription: "sub-1".into(),
        model: model.into(),
        user_agent: "test/1.0".into(),
        request_start: std::time::Instant::now(),
        is_error: false,
        response_status: 200,
    }
}
