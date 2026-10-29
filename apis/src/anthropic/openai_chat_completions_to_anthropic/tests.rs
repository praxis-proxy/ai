// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use http::{HeaderValue, StatusCode};
use praxis_filter::{FilterAction, Response, SelectedUpstreamBodyOutcome};
use serde_json::{Value, json};

use super::{
    BOUND_PREFIX, CREDENTIAL_KEY, CREDENTIAL_NAME, CREDENTIAL_NAMESPACE, CREDENTIAL_STRATEGY,
    OpenaiChatCompletionsToAnthropicFilter, RESPONSE_ARMED, request::translate_request, response,
};

const INPUT: &str = r#"{
      "model":"client-model",
      "max_completion_tokens":90,
      "stream":true,
      "stream_options":{"include_usage":true},
      "messages":[
        {"role":"system","content":"be useful"},
        {"role":"user","content":"find weather"},
        {"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Oslo\"}"}}]},
        {"role":"tool","tool_call_id":"call-1","content":"snow"}
      ],
      "tools":[{"type":"function","function":{"name":"weather","description":"look up weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
      "tool_choice":{"type":"function","function":{"name":"weather"}}
    }"#;

#[test]
fn converts_legacy_request_direction_and_binds_target_model() {
    let (bytes, profile) = translate_request(INPUT.as_bytes(), "claude-test").unwrap();
    let translated: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(translated["model"], "claude-test");
    assert_eq!(translated["max_tokens"], 90);
    assert_eq!(translated["system"], "be useful");
    assert_eq!(translated["messages"][0]["content"][0]["text"], "find weather");
    assert_eq!(translated["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(translated["messages"][1]["content"][0]["input"]["city"], "Oslo");
    assert_eq!(translated["messages"][2]["content"][0]["tool_use_id"], "call-1");
    assert_eq!(translated["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(translated["tool_choice"], json!({"type":"tool","name":"weather"}));
    assert_eq!(
        profile.client_model, "client-model",
        "the profile must preserve the client-visible model"
    );
    assert!(profile.stream);
    assert!(profile.include_usage);
}

#[test]
fn rejects_unmapped_fields_but_preserves_supported_image_content() {
    let unsupported =
        br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"frequency_penalty":0.3}"#;
    assert!(
        translate_request(unsupported, "provider-model")
            .unwrap_err()
            .contains("frequency_penalty")
    );
}

#[test]
fn invalid_tool_arguments_still_fail() {
    let invalid = br#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"call","type":"function","function":{"name":"f","arguments":"[]"}}]}],"max_tokens":8}"#;
    assert!(
        translate_request(invalid, "provider-model")
            .unwrap_err()
            .contains("JSON object")
    );
}

#[test]
fn defaults_missing_max_tokens_and_preserves_alias_precedence() {
    let base = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]"#;
    let defaulted = format!("{base}}}");
    let (bytes, _) = translate_request(defaulted.as_bytes(), "provider-model").unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap()["max_tokens"],
        4096,
        "omitting both token aliases must use the compatibility default"
    );

    for (field, expected) in [("max_tokens", 12), ("max_completion_tokens", 13)] {
        let input = format!("{base},\"{field}\":{expected}}}");
        let (bytes, _) = translate_request(input.as_bytes(), "provider-model").unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["max_tokens"],
            expected,
            "an explicit {field} must override the compatibility default"
        );
    }
    let equal = format!("{base},\"max_tokens\":14,\"max_completion_tokens\":14}}");
    let (bytes, _) = translate_request(equal.as_bytes(), "provider-model").unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap()["max_tokens"],
        14,
        "equal token aliases must preserve their explicit value"
    );
    let conflict = format!("{base},\"max_tokens\":15,\"max_completion_tokens\":16}}");
    let (bytes, _) = translate_request(conflict.as_bytes(), "provider-model").unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap()["max_tokens"],
        16,
        "max_completion_tokens must take precedence over max_tokens"
    );
}

#[test]
fn rejects_invalid_token_values() {
    for (field, value) in [
        ("max_tokens", "0"),
        ("max_tokens", "-1"),
        ("max_tokens", "1.5"),
        ("max_tokens", "\"many\""),
        ("max_tokens", "null"),
        ("max_completion_tokens", "0"),
        ("max_completion_tokens", "-1"),
        ("max_completion_tokens", "1.5"),
        ("max_completion_tokens", "\"many\""),
        ("max_completion_tokens", "null"),
    ] {
        let input = format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"{field}":{value}}}"#);
        assert!(
            translate_request(input.as_bytes(), "provider-model")
                .unwrap_err()
                .contains("positive integer"),
            "{field} value {value} should be rejected"
        );
    }
}

#[test]
fn accepts_and_drops_only_the_legacy_ignored_options() {
    for (field, value) in [
        ("user", r#""customer-42""#),
        ("n", "2"),
        ("seed", "42"),
        ("parallel_tool_calls", "true"),
    ] {
        let input = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"max_tokens":8,"{field}":{value}}}"#
        );
        let (bytes, _) = translate_request(input.as_bytes(), "provider-model").unwrap();
        let output: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            output.get(field).is_none(),
            "{field} must be intentionally omitted upstream"
        );
    }
    let combined = br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"user":"u","n":3,"seed":-7,"parallel_tool_calls":false}"#;
    let (bytes, _) = translate_request(combined, "provider-model").unwrap();
    let output: Value = serde_json::from_slice(&bytes).unwrap();
    for field in ["user", "n", "seed", "parallel_tool_calls"] {
        assert!(
            output.get(field).is_none(),
            "the combined request must omit {field} upstream"
        );
    }
}

#[test]
fn rejects_malformed_ignored_options() {
    for (field, value) in [
        ("user", "1"),
        ("n", "0"),
        ("n", "1.5"),
        ("seed", "true"),
        ("parallel_tool_calls", "1"),
    ] {
        let input = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"max_tokens":8,"{field}":{value}}}"#
        );
        assert!(
            translate_request(input.as_bytes(), "provider-model").is_err(),
            "invalid {field} must be rejected"
        );
    }
}

#[test]
fn translates_base64_and_remote_image_sources_in_content_order() {
    let input = br#"{"model":"client-model","messages":[{"role":"user","content":[{"type":"text","text":"first"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC","detail":"high"}},{"type":"text","text":"last"},{"type":"image_url","image_url":{"url":"https://images.example.test/photo.png"}}]}]}"#;
    let (bytes, profile) = translate_request(input, "provider-model").unwrap();
    let output: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        output["model"], "provider-model",
        "image requests must use the selected provider model"
    );
    assert_eq!(
        profile.client_model, "client-model",
        "the profile must preserve the client-visible model"
    );
    let content = output["messages"][0]["content"].as_array().unwrap();
    assert_eq!(
        content.len(),
        4,
        "image conversion must retain every input content block"
    );
    assert_eq!(
        content[0],
        json!({"type":"text","text":"first"}),
        "leading text must remain before the first image"
    );
    assert_eq!(
        content[1],
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}}),
        "the inline image must preserve its encoding and declared media type"
    );
    assert_eq!(
        content[2],
        json!({"type":"text","text":"last"}),
        "intervening text must remain between the images"
    );
    assert_eq!(
        content[3],
        json!({"type":"image","source":{"type":"url","url":"https://images.example.test/photo.png"}}),
        "the remote image must remain an unchanged URL reference"
    );
}

#[test]
fn rejects_assistant_images_without_relabeling_or_dropping_them() {
    for image in [
        "https://images.example.test/photo.png",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
    ] {
        let input = json!({"model":"m","messages":[
            {"role":"user","content":"Describe the image"},
            {"role":"assistant","content":[
                {"type":"text","text":"before"},
                {"type":"image_url","image_url":{"url":image}},
                {"type":"text","text":"after"}
            ]}
        ]});
        assert!(
            translate_request(input.to_string().as_bytes(), "provider-model")
                .unwrap_err()
                .contains("supported only in user messages"),
            "assistant image must be rejected for source {image}"
        );
    }
}

#[test]
fn rejects_invalid_preferred_token_alias_even_with_valid_legacy_alias() {
    for invalid in [json!(0), json!(-1), json!(1.5), json!("many"), Value::Null] {
        let input = json!({"model":"m","messages":[{"role":"user","content":"hi"}],
            "max_tokens":8,"max_completion_tokens":invalid});
        assert!(
            translate_request(input.to_string().as_bytes(), "provider-model")
                .unwrap_err()
                .contains("positive integer"),
            "invalid preferred alias must not fall back to max_tokens"
        );
    }
}

#[test]
fn validates_base64_beyond_the_image_signature_and_preserves_encoded_data() {
    let mut decoded = b"\x89PNG\r\n\x1a\n".to_vec();
    decoded.resize(32 * 1024, 0);
    let encoded = STANDARD.encode(&decoded);
    let request = |payload: &str| {
        json!({"model":"m","messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":format!("data:image/png;base64,{payload}")}}
        ]}]})
        .to_string()
    };
    let (translated, _) = translate_request(request(&encoded).as_bytes(), "provider-model").unwrap();
    let translated: Value = serde_json::from_slice(&translated).unwrap();
    assert_eq!(
        translated["messages"][0]["content"][0]["source"]["data"], encoded,
        "incremental validation must forward the original encoding unchanged"
    );
    for invalid in [
        format!("{encoded}%"),
        format!("{encoded}AAAA"),
        encoded.trim_end_matches('=').to_owned(),
    ] {
        assert!(
            translate_request(request(&invalid).as_bytes(), "provider-model")
                .unwrap_err()
                .contains("invalid base64"),
            "invalid trailing data or padding after a valid signature must be rejected"
        );
    }
}

#[test]
fn rejects_malformed_or_unsupported_images() {
    for (url, reason) in [
        ("data:image/png;base64,%%%", "invalid base64"),
        ("data:image/png;base64,aGVsbG8=", "does not match"),
        ("data:image/svg+xml;base64,PHN2Zz4=", "media type"),
        ("data:image/png,abc", "base64 encoding marker"),
        ("http://images.example.test/a.png", "HTTPS URL"),
        (
            "https://user:password@images.example.test/a.png",
            "embedded credentials",
        ),
        ("https://user@images.example.test/a.png", "embedded credentials"),
        ("not a url", "valid HTTPS"),
    ] {
        let input = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{url}"}}}}]}}]}}"#
        );
        assert!(
            translate_request(input.as_bytes(), "provider-model")
                .unwrap_err()
                .contains(reason),
            "{url} should fail with {reason}"
        );
    }
}

#[test]
fn tool_choice_none_preserves_tools_for_existing_tool_history() {
    let input = br#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"done"},{"role":"user","content":"continue"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"none"}"#;
    let (bytes, _) = translate_request(input, "provider-model").unwrap();
    let translated: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(translated["tools"][0]["name"], "f");
    assert_eq!(translated["tool_choice"], json!({"type":"none"}));
    assert_eq!(translated["messages"][0]["content"][0]["type"], "tool_use");
    assert_eq!(translated["messages"][1]["content"][0]["type"], "tool_result");
    assert_eq!(translated["messages"][1]["content"][0]["tool_use_id"], "call-1");
}

#[test]
fn empty_assistant_text_is_omitted_when_tool_calls_are_present() {
    let input = br#"{"model":"m","messages":[{"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"f","arguments":"{}"}}]}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}"#;
    let (bytes, _) = translate_request(input, "provider-model").unwrap();
    let translated: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(translated["messages"][0]["content"].as_array().unwrap().len(), 1);
    assert_eq!(translated["messages"][0]["content"][0]["type"], "tool_use");
}

#[test]
fn converts_anthropic_json_tools_usage_and_stop_reason() {
    let body = br#"{"type":"message","id":"msg_123","role":"assistant","model":"claude","content":[{"type":"text","text":"done"},{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Oslo"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":4,"cache_read_input_tokens":2}}"#;
    let translated: Value =
        serde_json::from_slice(&response::translate_response(body, "client-model").unwrap()).unwrap();
    assert_eq!(translated["id"], "chatcmpl-msg_123");
    assert_eq!(translated["model"], "client-model");
    assert_eq!(translated["choices"][0]["message"]["content"], "done");
    assert_eq!(
        translated["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Oslo\"}"
    );
    assert_eq!(translated["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(translated["usage"]["total_tokens"], 16);
    assert_eq!(translated["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
}

#[test]
fn includes_cached_input_in_client_visible_usage() {
    let body = br#"{"type":"message","id":"msg_cached","role":"assistant","content":[],"stop_reason":"end_turn","usage":{"input_tokens":10,"cache_read_input_tokens":1000,"cache_creation_input_tokens":50,"output_tokens":4}}"#;
    let translated: Value =
        serde_json::from_slice(&response::translate_response(body, "client-model").unwrap()).unwrap();
    assert_eq!(translated["usage"]["prompt_tokens"], 1060);
    assert_eq!(translated["usage"]["completion_tokens"], 4);
    assert_eq!(translated["usage"]["total_tokens"], 1064);
    assert_eq!(translated["usage"]["prompt_tokens_details"]["cached_tokens"], 1000);
}

#[test]
fn rejects_overflowed_cached_input_usage() {
    let body = format!(
        "{{\"type\":\"message\",\"id\":\"msg_overflow\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":\"end_turn\",\"usage\":{{\"input_tokens\":{},\"cache_read_input_tokens\":1,\"output_tokens\":0}}}}",
        u64::MAX
    );
    assert!(
        response::translate_response(body.as_bytes(), "client-model")
            .unwrap_err()
            .contains("overflow")
    );
}

#[test]
fn accepts_zero_cache_counters() {
    let body = br#"{"type":"message","id":"msg_zero","role":"assistant","content":[],"stop_reason":"end_turn","usage":{"input_tokens":7,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":2}}"#;
    let translated: Value =
        serde_json::from_slice(&response::translate_response(body, "client-model").unwrap()).unwrap();
    assert_eq!(translated["usage"]["prompt_tokens"], 7);
    assert_eq!(translated["usage"]["total_tokens"], 9);
    assert_eq!(translated["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
}

#[test]
fn accepts_null_cache_counters() {
    let body = br#"{"type":"message","id":"msg_null","role":"assistant","content":[],"stop_reason":"end_turn","usage":{"input_tokens":7,"cache_read_input_tokens":null,"cache_creation_input_tokens":null,"output_tokens":2}}"#;
    let translated: Value =
        serde_json::from_slice(&response::translate_response(body, "client-model").unwrap()).unwrap();
    assert_eq!(translated["usage"]["prompt_tokens"], 7);
    assert_eq!(translated["usage"]["total_tokens"], 9);
    assert!(translated["usage"]["prompt_tokens_details"].is_null());
}

#[test]
fn maps_refusal_stop_reason_to_content_filter() {
    let body = br#"{"type":"message","id":"msg_refusal","role":"assistant","model":"claude","content":[{"type":"text","text":"I cannot help with that request."}],"stop_reason":"refusal","usage":{"input_tokens":4,"output_tokens":6}}"#;
    let translated: Value =
        serde_json::from_slice(&response::translate_response(body, "client-model").unwrap()).unwrap();
    assert_eq!(translated["choices"][0]["finish_reason"], "content_filter");
    let unsupported = br#"{"type":"message","id":"msg_future","role":"assistant","model":"claude","content":[],"stop_reason":"future_reason","usage":{"input_tokens":4,"output_tokens":0}}"#;
    assert_eq!(
        response::translate_response(unsupported, "client-model").unwrap_err(),
        "unsupported Anthropic stop_reason \"future_reason\""
    );
}

#[test]
fn maps_context_window_exhaustion_to_length() {
    let body = br#"{"type":"message","id":"msg_context_limit","role":"assistant","model":"claude","content":[{"type":"text","text":"partial answer"}],"stop_reason":"model_context_window_exceeded","usage":{"input_tokens":4,"output_tokens":6}}"#;
    let translated: Value =
        serde_json::from_slice(&response::translate_response(body, "client-model").unwrap()).unwrap();
    assert_eq!(translated["choices"][0]["message"]["content"], "partial answer");
    assert_eq!(translated["choices"][0]["finish_reason"], "length");
}

#[test]
fn maps_anthropic_errors_to_openai_error_envelope() {
    let error = response::translate_error(
        br#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
        529,
    );
    let value: Value = serde_json::from_slice(&error).unwrap();
    assert_eq!(value["error"]["message"], "busy");
    assert_eq!(value["error"]["type"], "server_error");
}

#[tokio::test]
async fn preserves_upstream_error_status_across_response_callbacks() {
    let config: serde_yaml::Value = serde_yaml::from_str("api_translation_enabled: true").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    response.status = StatusCode::TOO_MANY_REQUESTS;
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.response_header = None;
    let mut body = Some(Bytes::from_static(br#"{"type":"error","error":{"message":"busy"}}"#));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    let translated: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(translated["error"]["message"], "busy");
    assert_eq!(translated["error"]["type"], "rate_limit_error");
    assert_eq!(translated["error"]["code"], 429);
}

#[tokio::test]
async fn request_profile_stays_pinned_through_response_after_route_metadata_changes() {
    let config: serde_yaml::Value =
        serde_yaml::from_str("api_translation_enabled: true\nmax_body_bytes: 4096").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions?trace=1");
    let mut response = Response {
        status: StatusCode::OK,
        headers: [(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))]
            .into_iter()
            .collect(),
    };
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(ctx.rewritten_path.as_deref(), Some("/v1/messages?trace=1"));
    assert!(
        ctx.request_headers_to_remove
            .iter()
            .any(|header| header == "authorization")
    );
    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(header, value)| header == "anthropic-version" && value == "2023-06-01")
    );

    let mut body = Some(Bytes::from_static(
        br#"{"model":"client-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20}"#,
    ));
    let outcome = filter
        .on_selected_upstream_request_body(&mut ctx, &mut body)
        .await
        .unwrap();
    assert!(matches!(outcome, SelectedUpstreamBodyOutcome::Continue));
    let upstream: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(upstream["model"], "claude-test");

    // Simulate a later overlay selection. The bound profile is the response authority.
    ctx.set_metadata("intelligent_route.api_format", "openai-chat");
    ctx.set_metadata("intelligent_route.overlay_revision", "a".repeat(64));
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(ctx.get_metadata(RESPONSE_ARMED), Some("true"));
    let anthropic = Bytes::from_static(br#"{"type":"message","id":"msg_1","role":"assistant","model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#);
    let mut response_body = Some(anthropic);
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut response_body, true).unwrap(),
        FilterAction::Continue
    ));
    let downstream: Value = serde_json::from_slice(response_body.as_ref().unwrap()).unwrap();
    assert_eq!(downstream["model"], "client-model");
    assert_eq!(downstream["choices"][0]["message"]["content"], "ok");
    assert_eq!(ctx.get_metadata(&format!("{BOUND_PREFIX}revision")).unwrap().len(), 64);
}

#[tokio::test]
async fn stale_anthropic_profile_fails_closed_when_tenant_permission_is_off() {
    let config: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    selected_profile_metadata(&mut ctx);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(_)),
        "disabled tenant permission must reject a stale Anthropic profile"
    );
    let FilterAction::Reject(rejection) = action else {
        return;
    };
    assert_eq!(rejection.status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        String::from_utf8_lossy(rejection.body.as_deref().unwrap())
            .contains("platform administrator has forbidden translation")
    );
    assert!(ctx.request_headers_to_set.is_empty());
}

#[tokio::test]
async fn openai_candidate_rewrites_only_the_selected_target_model() {
    let config: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    selected_profile_metadata_with_format(&mut ctx, "openai-chat");
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let original =
        Bytes::from_static(br#"{"model":"client-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20}"#);
    let mut body = Some(original.clone());
    let outcome = filter
        .on_selected_upstream_request_body(&mut ctx, &mut body)
        .await
        .unwrap();
    assert!(matches!(outcome, SelectedUpstreamBodyOutcome::Continue));
    let translated: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(translated["model"], "claude-test");
    let mut original_value: Value = serde_json::from_slice(&original).unwrap();
    original_value["model"] = Value::String("claude-test".to_owned());
    assert_eq!(translated, original_value);
}

#[tokio::test]
#[expect(clippy::panic, reason = "tests assert that unsupported requests are rejected")]
async fn selected_request_body_limit_rejects_before_json_parsing() {
    let config: serde_yaml::Value =
        serde_yaml::from_str("api_translation_enabled: true\nmax_request_body_bytes: 1").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut body = Some(Bytes::from_static(b"{}"));
    let outcome = filter
        .on_selected_upstream_request_body(&mut ctx, &mut body)
        .await
        .unwrap();
    let SelectedUpstreamBodyOutcome::Reject(rejection) = outcome else {
        panic!("body over the configured limit must be rejected");
    };
    assert_eq!(rejection.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(
        String::from_utf8_lossy(rejection.body.as_deref().unwrap())
            .contains("size limit set by the platform administrator")
    );
}

#[tokio::test]
#[expect(
    clippy::panic,
    reason = "tests assert that request bodies at the limit pass and over-limit bodies fail"
)]
async fn split_request_limit_accepts_boundary_and_rejects_one_byte_over() {
    let valid = br#"{"model":"client-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20}"#;
    let max_request_body_bytes = valid.len() + 2;
    let yaml = format!(
        "api_translation_enabled: true\nmax_request_body_bytes: {max_request_body_bytes}\nmax_response_body_bytes: 1"
    );
    let config: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();

    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let mut at_limit = Vec::from(valid.as_slice());
    at_limit.extend_from_slice(b"  ");
    assert_eq!(at_limit.len(), max_request_body_bytes);
    let mut body = Some(Bytes::from(at_limit));
    let outcome = filter
        .on_selected_upstream_request_body(&mut ctx, &mut body)
        .await
        .unwrap();
    assert!(matches!(outcome, SelectedUpstreamBodyOutcome::Continue));

    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let mut over_limit = Vec::from(valid.as_slice());
    over_limit.extend_from_slice(b"   ");
    assert_eq!(over_limit.len(), max_request_body_bytes + 1);
    let mut body = Some(Bytes::from(over_limit));
    let outcome = filter
        .on_selected_upstream_request_body(&mut ctx, &mut body)
        .await
        .unwrap();
    let SelectedUpstreamBodyOutcome::Reject(rejection) = outcome else {
        panic!("request one byte over the configured limit must be rejected");
    };
    assert_eq!(rejection.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(String::from_utf8_lossy(rejection.body.as_deref().unwrap()).contains("request_too_large"));
}

#[tokio::test]
async fn translates_json_response_split_across_streamed_chunks() {
    let config: serde_yaml::Value =
        serde_yaml::from_str("api_translation_enabled: true\nmax_body_bytes: 4096").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    translate_request_body(filter.as_ref(), &mut ctx).await;
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let upstream = anthropic_json_response();
    let split = upstream.len() / 2;
    let mut first = Some(Bytes::copy_from_slice(&upstream[..split]));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut first, false).unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(
        first,
        Some(Bytes::new()),
        "raw provider JSON must not leak before translation"
    );
    assert!(ctx.get_filter_state::<super::BufferedJsonResponse>().is_some());

    let mut second = Some(Bytes::copy_from_slice(&upstream[split..]));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut second, true).unwrap(),
        FilterAction::Continue
    ));
    let downstream: Value = serde_json::from_slice(second.as_ref().unwrap()).unwrap();
    assert_eq!(downstream["model"], "client-model");
    assert_eq!(downstream["choices"][0]["message"]["content"], "split response");
    assert_eq!(downstream["usage"]["total_tokens"], 5);
    assert!(ctx.get_filter_state::<super::BufferedJsonResponse>().is_none());
}

#[tokio::test]
#[expect(
    clippy::panic,
    reason = "tests assert that over-limit provider responses are rejected"
)]
async fn rejects_json_response_that_exceeds_accumulation_limit() {
    let config: serde_yaml::Value =
        serde_yaml::from_str("api_translation_enabled: true\nmax_response_body_bytes: 128").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    translate_request_body(filter.as_ref(), &mut ctx).await;
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let upstream = anthropic_json_response();
    let split = 96;
    assert!(split < 128 && upstream.len() > 128);
    let mut first = Some(Bytes::copy_from_slice(&upstream[..split]));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut first, false).unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(first, Some(Bytes::new()));
    let mut second = Some(Bytes::copy_from_slice(&upstream[split..]));
    let action = filter.on_response_body(&mut ctx, &mut second, true).unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("over-limit provider JSON must be rejected");
    };
    assert_eq!(rejection.status, StatusCode::BAD_GATEWAY);
    assert!(
        String::from_utf8_lossy(rejection.body.as_deref().unwrap())
            .contains("size limit set by the platform administrator")
    );
    assert!(ctx.get_filter_state::<super::BufferedJsonResponse>().is_none());
}

#[tokio::test]
#[expect(
    clippy::panic,
    reason = "tests assert the response cap remains independent of the larger request cap"
)]
async fn split_response_limit_is_enforced_after_a_larger_request_limit_accepts() {
    let upstream = anthropic_json_response();
    let response_limit = upstream.len();
    let yaml = format!(
        "api_translation_enabled: true\nmax_request_body_bytes: 4096\nmax_response_body_bytes: {response_limit}"
    );
    let config: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();

    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    translate_request_body(filter.as_ref(), &mut ctx).await;
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let mut body = Some(Bytes::copy_from_slice(upstream));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    assert!(serde_json::from_slice::<Value>(body.as_ref().unwrap()).is_ok());

    let yaml = format!(
        "api_translation_enabled: true\nmax_request_body_bytes: 4096\nmax_response_body_bytes: {}",
        response_limit - 1
    );
    let config: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    translate_request_body(filter.as_ref(), &mut ctx).await;
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let mut body = Some(Bytes::copy_from_slice(upstream));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("response one byte over the configured response limit must be rejected");
    };
    assert_eq!(rejection.status, StatusCode::BAD_GATEWAY);
    assert!(String::from_utf8_lossy(rejection.body.as_deref().unwrap()).contains("provider_response_too_large"));
}

#[tokio::test]
#[expect(clippy::panic, reason = "tests assert that invalid provider JSON is rejected")]
async fn rejects_invalid_provider_json_with_a_downstream_502() {
    let config: serde_yaml::Value = serde_yaml::from_str("api_translation_enabled: true").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata(&mut ctx);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    translate_request_body(filter.as_ref(), &mut ctx).await;
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut invalid_json = Some(Bytes::from_static(b"{not-json"));
    let action = filter.on_response_body(&mut ctx, &mut invalid_json, true).unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("invalid provider JSON must produce a downstream 502");
    };
    assert_eq!(rejection.status, StatusCode::BAD_GATEWAY);
    let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid Anthropic response JSON")
    );
    assert_eq!(
        invalid_json,
        Some(Bytes::new()),
        "invalid provider bytes must not reach the client"
    );
}

#[tokio::test]
async fn openai_candidate_json_response_remains_chunked_passthrough() {
    let config: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = OpenaiChatCompletionsToAnthropicFilter::from_config(&config).unwrap();
    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut response = json_response();
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.current_filter_id = Some(7);
    selected_profile_metadata_with_format(&mut ctx, "openai-chat");
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(ctx.get_metadata(RESPONSE_ARMED), None);

    for (chunk, eos) in [(b"{\"choices\":".as_slice(), false), (b"[]}".as_slice(), true)] {
        let original = Bytes::copy_from_slice(chunk);
        let mut body = Some(original.clone());
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut body, eos).unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(
            body,
            Some(original),
            "OpenAI response chunk must pass through unchanged"
        );
    }
    assert!(ctx.get_filter_state::<super::BufferedJsonResponse>().is_none());
}

async fn translate_request_body(
    filter: &dyn praxis_filter::HttpFilter,
    ctx: &mut praxis_filter::HttpFilterContext<'_>,
) {
    let mut body = Some(Bytes::from_static(
        br#"{"model":"client-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20}"#,
    ));
    let outcome = filter.on_selected_upstream_request_body(ctx, &mut body).await.unwrap();
    assert!(matches!(outcome, SelectedUpstreamBodyOutcome::Continue));
}

fn json_response() -> Response {
    Response {
        status: StatusCode::OK,
        headers: [(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))]
            .into_iter()
            .collect(),
    }
}

fn anthropic_json_response() -> &'static [u8] {
    br#"{"type":"message","id":"msg_split","role":"assistant","model":"claude-test","content":[{"type":"text","text":"split response"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#
}

fn selected_profile_metadata(ctx: &mut praxis_filter::HttpFilterContext<'_>) {
    selected_profile_metadata_with_format(ctx, "messages");
    ctx.set_metadata(CREDENTIAL_STRATEGY, "apikey");
    ctx.set_metadata(CREDENTIAL_NAME, "anthropic-key");
    ctx.set_metadata(CREDENTIAL_NAMESPACE, "tenant-a");
    ctx.set_metadata(CREDENTIAL_KEY, "api-key");
}

fn selected_profile_metadata_with_format(ctx: &mut praxis_filter::HttpFilterContext<'_>, format: &str) {
    ctx.set_metadata("intelligent_route.outcome", "routed");
    ctx.set_metadata("intelligent_route.stable_id", "model/provider");
    ctx.set_metadata("intelligent_route.overlay_revision", "b".repeat(64));
    ctx.set_metadata("intelligent_route.api_format", format);
    ctx.set_metadata("intelligent_route.path", "/v1/messages");
    ctx.set_metadata("intelligent_route.target_model", "claude-test");
    ctx.set_metadata("intelligent_route.name", "client-model");
}
