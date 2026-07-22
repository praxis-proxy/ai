// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_file_resolve` filter.

use std::{
    io::{Read as _, Write as _},
    net::TcpListener,
};

use bytes::Bytes;
use serde_json::json;

use super::*;
use crate::openai::{
    api_client::{ApiClient, ApiClientConfig},
    responses::state::ResponsesState,
};

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn from_config_with_valid_url_succeeds() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "files_api_url: \"http://files-api:8321\"\nallow_private_files_api_url: true\nallow_pre_security_callout: true",
    )
    .unwrap();
    let filter = FileResolveFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_file_resolve", "filter name should match");
}

#[test]
fn from_config_missing_url_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let result = FileResolveFilter::from_config(&yaml);
    assert!(result.is_err(), "missing files_api_url should be rejected");
}

#[test]
fn from_config_empty_url_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("files_api_url: ''").unwrap();
    let result = FileResolveFilter::from_config(&yaml);
    assert!(result.is_err(), "empty files_api_url should be rejected");
}

#[test]
fn from_config_unknown_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "files_api_url: \"http://files-api:8321\"\nallow_private_files_api_url: true\non_mising: reject",
    )
    .unwrap();
    let result = FileResolveFilter::from_config(&yaml);
    assert!(result.is_err(), "typo in config field should be rejected");
}

// -----------------------------------------------------------------------------
// Body Access
// -----------------------------------------------------------------------------

#[test]
fn body_access_is_read_write() {
    let filter = make_filter();
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadWrite,
        "file_resolve must have read-write body access"
    );
}

#[test]
fn body_mode_is_stream_buffer() {
    let filter = make_filter();
    match filter.request_body_mode() {
        BodyMode::StreamBuffer { max_bytes } => {
            assert_eq!(
                max_bytes,
                Some(67_108_864),
                "StreamBuffer should default to 64 MiB limit"
            );
        },
        other => panic!("expected StreamBuffer, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Reject Helpers
// -----------------------------------------------------------------------------

#[test]
fn reject_callout_failed_returns_502() {
    let err = ResolveError::CalloutFailed {
        file_id: "file-abc".to_owned(),
        detail: "content download failed for http://files.internal:8321/v1/files/file-abc/content: connection refused"
            .to_owned(),
    };
    let action = reject_resolve_error(&err);
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 502, "callout failure should produce 502");
            let body = std::str::from_utf8(r.body.as_deref().unwrap()).unwrap();
            assert!(
                body.contains("Files API request failed"),
                "client response should describe the failure generically"
            );
            assert!(
                !body.contains("files.internal"),
                "client response must not expose callout URL"
            );
            assert!(
                !body.contains("connection refused"),
                "client response must not expose internal transport details"
            );
        },
        _ => panic!("expected Reject action"),
    }
}

#[test]
fn reject_invalid_file_id_returns_400() {
    let err = ResolveError::InvalidFileId {
        file_id: "..".to_owned(),
        detail: "dot path segments are not valid file IDs".to_owned(),
    };
    let action = reject_resolve_error(&err);
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 400, "invalid file IDs should produce 400");
        },
        _ => panic!("expected Reject action"),
    }
}

#[test]
fn reject_too_many_references_returns_413() {
    let err = ResolveError::TooManyReferences { limit: 32 };
    let action = reject_resolve_error(&err);

    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "too many file references should produce 413");
        },
        _ => panic!("expected Reject action"),
    }
}

#[test]
fn reject_too_large_returns_413() {
    let err = ResolveError::TooLarge {
        file_id: "file-abc".to_owned(),
        limit: 1024,
    };
    let action = reject_resolve_error(&err);
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "oversized resolved content should produce 413");
        },
        _ => panic!("expected Reject action"),
    }
}

#[test]
fn reject_rewritten_body_too_large_returns_413() {
    let action = reject_rewritten_body_too_large(2048, 1024);
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "oversized rewritten body should produce 413");
        },
        _ => panic!("expected Reject action"),
    }
}

// -----------------------------------------------------------------------------
// on_request_body
// -----------------------------------------------------------------------------

#[tokio::test]
async fn rejects_oversized_raw_body_before_resolution() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "files_api_url: \"http://127.0.0.1:9\"\nallow_private_files_api_url: true\nallow_pre_security_callout: true\non_missing: reject\nmax_body_bytes: 64",
    )
    .unwrap();
    let filter = FileResolveFilter::from_config(&yaml).unwrap();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"file-never-fetched"}]}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 413),
        "the resolver's own body limit must be enforced before parsing or callouts"
    );
}

#[tokio::test]
async fn skips_non_responses_request() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/chat/completions",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_chat_completions");
    let mut body = Some(Bytes::from(r#"{"messages":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "non-responses request should be released"
    );
}

#[tokio::test]
async fn skips_non_create_responses_endpoint() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses/compact",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"file-abc"}]}]}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "non-create responses endpoint should be released"
    );
}

#[tokio::test]
async fn skips_missing_format_metadata() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(r#"{"input":"test"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "request without format metadata should be released"
    );
}

#[tokio::test]
async fn releases_missing_body() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body: Option<Bytes> = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "missing body should be released"
    );
}

#[tokio::test]
async fn releases_invalid_json() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from("not json"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "invalid JSON should be released"
    );
}

#[tokio::test]
async fn continues_on_no_file_id() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let original = r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}"#;
    let mut body = Some(Bytes::from(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "request with no file_id should continue"
    );
    assert_eq!(body.as_deref(), Some(original.as_bytes()), "body should be unchanged");
}

#[tokio::test]
async fn not_end_of_stream_continues() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(Bytes::from(r#"{"input":"partial"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-end-of-stream should continue"
    );
}

#[tokio::test]
async fn string_input_passes_through() {
    let filter = make_filter();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let original = r#"{"input":"Hello, world!"}"#;
    let mut body = Some(Bytes::from(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "string input should continue");
}

// -----------------------------------------------------------------------------
// sync_state
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sync_state_updates_responses_state() {
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_file", "file_id": "file-abc"}]
            }
        ]
    });
    let mut state = ResponsesState::from_request_body(request_body);

    let history = vec![json!({"role": "user", "content": "earlier turn"})];
    state.messages.splice(0..0, history.clone());
    state.persisted_messages.splice(0..0, history);
    ctx.extensions.insert(state);

    let resolved_body = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_file", "file_data": "SGVsbG8="}]
            }
        ]
    });

    let client = make_client();
    sync_state(&mut ctx, &resolved_body, &client, OnMissing::Continue)
        .await
        .unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.request_body, resolved_body,
        "request_body should be updated with resolved content"
    );

    let tail = &state.messages[1];
    assert!(
        tail["content"][0].get("file_id").is_none(),
        "file_id should be removed from messages tail"
    );
    assert_eq!(
        tail["content"][0]["file_data"], "SGVsbG8=",
        "resolved file_data should appear in messages"
    );

    let persisted_tail = &state.persisted_messages[1];
    assert_eq!(
        persisted_tail["content"][0]["file_data"], "SGVsbG8=",
        "resolved file_data should appear in persisted_messages"
    );
}

#[tokio::test]
async fn sync_state_uses_independent_history_offsets() {
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);

    let request_body = json!({
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file-abc"}]
        }]
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state
        .messages
        .insert(0, json!({"role": "user", "content": "replay history"}));
    state.persisted_messages.splice(
        0..0,
        [
            json!({"role": "user", "content": "persisted history"}),
            json!({"type": "mcp_list_tools", "tools": []}),
        ],
    );
    ctx.extensions.insert(state);

    let resolved_body = json!({
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_data": "SGVsbG8="}]
        }]
    });
    sync_state(&mut ctx, &resolved_body, &make_client(), OnMissing::Continue)
        .await
        .unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.messages[1]["content"][0]["file_data"], "SGVsbG8=");
    assert_eq!(state.persisted_messages[1]["type"], "mcp_list_tools");
    assert_eq!(
        state.persisted_messages[2]["content"][0]["file_data"], "SGVsbG8=",
        "persisted input tail should use its own history length"
    );
}

#[tokio::test]
async fn resolves_history_when_current_input_has_no_file_id() {
    let files_api_url = start_files_api_stub();
    let filter = make_filter_for_url(&files_api_url);
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let request_body = json!({
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "summarize the prior file"}]
        }]
    });
    let history = json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_file", "file_id": "file-history"}]
    });
    let mut state = ResponsesState::from_request_body(request_body.clone());
    state.messages.insert(0, history.clone());
    state.persisted_messages.insert(0, history);
    ctx.extensions.insert(state);
    let original = Bytes::from(serde_json::to_vec(&request_body).unwrap());
    let mut body = Some(original.clone());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "history-only resolution should continue the request"
    );
    assert_eq!(body, Some(original), "current request body should remain unchanged");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    for resolved_history in [&state.messages[0], &state.persisted_messages[0]] {
        let part = &resolved_history["content"][0];
        assert!(
            part.get("file_id").is_none(),
            "history file_id should be removed after resolution"
        );
        assert_eq!(
            part["file_data"], "aGlzdG9yeQ==",
            "resolved history should contain inline base64"
        );
        assert_eq!(
            part["filename"], "history.txt",
            "resolved history should preserve metadata filename"
        );
    }
}

#[tokio::test]
async fn mirrored_history_has_independent_inline_budget() {
    let files_api_url = start_files_api_stub();
    let client = make_client_for_url_with_max(&files_api_url, 16);
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let request_body = json!({"input": []});
    let history = json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_file", "file_id": "file-history"}]
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.messages.push(history.clone());
    state.persisted_messages.push(history);
    ctx.extensions.insert(state);
    let mut budget = client.resolution_budget();

    resolve_state_history(&mut ctx, &client, OnMissing::Reject, &mut budget)
        .await
        .unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.messages[0]["content"][0]["file_data"], "aGlzdG9yeQ==");
    assert_eq!(
        state.persisted_messages[0]["content"][0]["file_data"], "aGlzdG9yeQ==",
        "the persistence mirror should not consume the outbound representation's byte budget"
    );
}

#[tokio::test]
async fn rejects_resolved_history_when_rebuilt_body_exceeds_limit() {
    let files_api_url = start_files_api_stub();
    let request_body = json!({"input": "continue"});
    let history = json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_file", "file_id": "file-history"}]
    });
    let mut state = ResponsesState::from_request_body(request_body.clone());
    state.messages.insert(0, history.clone());
    state.persisted_messages.insert(0, history);
    let unresolved_len = serialized_outbound_body_len(&state).unwrap();

    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        "files_api_url: \"{files_api_url}\"\nallow_private_files_api_url: true\nallow_pre_security_callout: true\nmax_body_bytes: {unresolved_len}"
    ))
    .unwrap();
    let filter = FileResolveFilter::from_config(&yaml).unwrap();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(serde_json::to_vec(&request_body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 413),
        "resolved rehydrated history should respect the resolver's final body limit"
    );
}

#[tokio::test]
async fn rejects_unresolvable_history_when_configured_to_reject() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        "files_api_url: \"http://files-api:8321\"\nallow_private_files_api_url: true\nallow_pre_security_callout: true\non_missing: reject",
    )
    .unwrap();
    let filter = FileResolveFilter::from_config(&yaml).unwrap();
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let request_body = json!({"input": "continue"});
    let history = json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_file", "file_id": ".."}]
    });
    let mut state = ResponsesState::from_request_body(request_body.clone());
    state.messages.insert(0, history.clone());
    state.persisted_messages.insert(0, history);
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(serde_json::to_vec(&request_body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 400),
        "on_missing: reject should also reject unresolved rehydrated history"
    );
}

#[tokio::test]
async fn sync_state_skipped_without_responses_state() {
    let req = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(req);

    let resolved_body = json!({"model": "gpt-4o", "input": []});
    let client = make_client();

    sync_state(&mut ctx, &resolved_body, &client, OnMissing::Continue)
        .await
        .unwrap();

    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "should not create state when none exists"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> Box<dyn HttpFilter> {
    make_filter_for_url("http://files-api:8321")
}

fn make_filter_for_url(files_api_url: &str) -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        "files_api_url: \"{files_api_url}\"\nallow_private_files_api_url: true\nallow_pre_security_callout: true"
    ))
    .unwrap();
    FileResolveFilter::from_config(&yaml).unwrap()
}

fn make_client() -> FilesApiClient {
    make_client_for_url_with_max("http://test:9999", 64 * 1024 * 1024)
}

fn make_client_for_url_with_max(files_api_url: &str, max_resolved_bytes: usize) -> FilesApiClient {
    let api = ApiClient::new(ApiClientConfig {
        api_base_url: files_api_url.to_owned(),
        callout_config: CalloutConfig::default(),
        forward_header_names: vec![],
    })
    .unwrap();
    FilesApiClient::new(
        api,
        FilesApiClientOptions {
            max_file_references: 32,
            max_resolved_bytes,
        },
    )
}

fn start_files_api_stub() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || serve_file_request(stream));
        }
    });

    format!("http://{address}")
}

fn serve_file_request(mut stream: std::net::TcpStream) {
    let mut request = [0_u8; 4096];
    let read = stream.read(&mut request).unwrap();
    let request = String::from_utf8_lossy(&request[..read]);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap();

    let (content_type, body): (&str, &[u8]) = if path.ends_with("/content") {
        ("text/plain", b"history")
    } else {
        (
            "application/json",
            br#"{"id":"file-history","filename":"history.txt","content_type":"text/plain","bytes":7}"#,
        )
    };
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
}
