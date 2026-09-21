// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for Responses web search with per-user callout credentials
//! and owner attribution.
//!
//! The `web-search-scoped-credentials.yaml` example demonstrates PR1 of issue
//! #880: `state_owner` maps trusted ingress identity into a normalized owner,
//! and `callout_credentials` captures a per-user provider key from the ingress
//! header `x-user-brave-key` into a slot, strips the header, and stages the
//! per-user secret into the Brave provider callout instead of the shared
//! api_key. A request missing the configured credential fails closed with a 401
//! before any provider callout runs.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send, parse_body,
    parse_status, patch_yaml, start_proxy,
};
use serde_json::json;

const EXAMPLE: &str = "openai/responses/web-search-scoped-credentials.yaml";

/// Read the example config and point its model backend and search provider at
/// the given local mock ports.
fn patched_yaml(listener_port: u16, model_port: u16, search_port: u16) -> String {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let yaml = patch_yaml(&yaml, listener_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!("api_key: test-key\n                base_url: http://127.0.0.1:{search_port}"),
    );
    // The provider callout targets a loopback mock, so the executor's SSRF check
    // requires the operator opt-in on the outbound pipeline.
    yaml.replace(
        "allow_private_endpoints: true",
        "allow_private_endpoints: true\n  allow_private_upstreams: true",
    )
}

fn load_test_config(listener_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    praxis_core::config::Config::from_yaml(&patched_yaml(listener_port, model_port, search_port))
        .expect("patched config should parse")
}

/// Build a POST with a JSON body and extra ingress headers.
fn json_post_with_headers(path: &str, body: &str, headers: &[(&str, &str)]) -> String {
    let mut extra = String::new();
    for (name, value) in headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         {extra}\
         \r\n\
         {body}",
        body.len(),
    )
}

#[test]
fn web_search_scoped_credentials_example_builds() {
    let config = load_test_config(free_port(), 19_401, 19_402);
    let _pipeline = build_pipeline(&config);
}

#[test]
fn per_user_credential_arrives_at_provider_and_ingress_header_is_stripped() {
    // The core proof: a per-user credential captured in the outer chain is staged
    // into the in-IRR-step web-search callout, proving the ordering is correct
    // (outer callout_credentials runs before the in-IRR-step openai_web_search
    // reads the slot). The test passes at all only if ordering is correct; wrong
    // ordering would 401.
    let first_response = json!({
        "id": "chatcmpl_search",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"Praxis Proxy latest release\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    });
    let second_response = json!({
        "id": "chatcmpl_answer",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Praxis Proxy has a current release."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
    });
    let model = StatefulCapturingBackend::new(vec![
        (200, first_response.to_string()),
        (200, second_response.to_string()),
    ])
    .start_with_shutdown();
    let brave_body = json!({
        "web": {"results": [{
            "title": "Praxis Proxy releases",
            "url": "https://github.com/praxis-proxy/praxis/releases",
            "description": "Current Praxis Proxy releases."
        }]}
    });
    let search = StatefulCapturingBackend::new(vec![(200, brave_body.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "tools": [{
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "FR"}
        }],
        "tool_choice": {"type": "web_search"},
        "include": ["web_search_call.action.sources"],
        "store": false
    });
    let headers = [
        ("x-auth-tenant", "acme"),
        ("x-auth-user", "alice"),
        ("x-user-brave-key", "alice-brave-secret"),
    ];

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/responses", &request.to_string(), &headers),
    );

    assert_eq!(parse_status(&raw), 200, "round trip should succeed: {raw}");
    let sreqs = search.requests();
    assert_eq!(sreqs.len(), 1, "one search callout");
    let h = sreqs[0].headers.to_ascii_lowercase();
    assert!(
        h.contains("x-subscription-token: alice-brave-secret"),
        "the provider callout carries the per-user secret, not the shared key: {}",
        sreqs[0].headers
    );
    assert!(
        !h.contains("test-key"),
        "the shared api_key does not appear on the callout: {}",
        sreqs[0].headers
    );
    assert!(
        !h.contains("x-user-brave-key"),
        "the ingress credential header is stripped before the provider callout: {}",
        sreqs[0].headers
    );
    let mreqs = model.requests();
    assert!(
        !mreqs[0].headers.to_ascii_lowercase().contains("x-user-brave-key"),
        "the ingress credential header is stripped before the model backend too: {}",
        mreqs[0].headers
    );
}

#[test]
fn missing_credential_fails_closed_with_401_before_any_provider_callout() {
    let first_response = json!({
        "id": "chatcmpl_search",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"Praxis Proxy latest release\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    });
    let model = StatefulCapturingBackend::new(vec![(200, first_response.to_string())]).start_with_shutdown();
    let brave_body = json!({
        "web": {"results": [{
            "title": "Praxis Proxy releases",
            "url": "https://github.com/praxis-proxy/praxis/releases",
            "description": "Current Praxis Proxy releases."
        }]}
    });
    let search = StatefulCapturingBackend::new(vec![(200, brave_body.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "tools": [{
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "FR"}
        }],
        "tool_choice": {"type": "web_search"},
        "include": ["web_search_call.action.sources"],
        "store": false
    });
    let headers = [("x-auth-tenant", "acme"), ("x-auth-user", "alice")];

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/responses", &request.to_string(), &headers),
    );

    assert_eq!(
        parse_status(&raw),
        401,
        "a missing configured credential fails closed with 401: {raw}"
    );
    assert!(
        parse_body(&raw).contains("missing_callout_context"),
        "the 401 carries the missing_callout_context code: {}",
        parse_body(&raw)
    );
    assert!(
        search.requests().is_empty(),
        "no provider callout on a missing credential"
    );
    assert_eq!(
        model.requests().len(),
        1,
        "only the first inference round runs before the 401 on re-entry"
    );
}

#[test]
fn per_user_credentials_are_isolated_across_requests() {
    let first_response = json!({
        "id": "chatcmpl_search",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"Praxis Proxy latest release\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    });
    let second_response = json!({
        "id": "chatcmpl_answer",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Praxis Proxy has a current release."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
    });
    let model = StatefulCapturingBackend::new(vec![
        (200, first_response.to_string()),
        (200, second_response.to_string()),
        (200, first_response.to_string()),
        (200, second_response.to_string()),
    ])
    .start_with_shutdown();
    let brave_body = json!({
        "web": {"results": [{
            "title": "Praxis Proxy releases",
            "url": "https://github.com/praxis-proxy/praxis/releases",
            "description": "Current Praxis Proxy releases."
        }]}
    });
    let search = StatefulCapturingBackend::new(vec![(200, brave_body.to_string()), (200, brave_body.to_string())])
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "tools": [{
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "FR"}
        }],
        "tool_choice": {"type": "web_search"},
        "include": ["web_search_call.action.sources"],
        "store": false
    });

    let headers_alice = [
        ("x-auth-tenant", "acme"),
        ("x-auth-user", "alice"),
        ("x-user-brave-key", "alice-brave-secret"),
    ];
    let raw1 = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/responses", &request.to_string(), &headers_alice),
    );
    assert_eq!(parse_status(&raw1), 200, "alice request should succeed");

    let headers_bob = [
        ("x-auth-tenant", "acme"),
        ("x-auth-user", "bob"),
        ("x-user-brave-key", "bob-brave-secret"),
    ];
    let raw2 = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/responses", &request.to_string(), &headers_bob),
    );
    assert_eq!(parse_status(&raw2), 200, "bob request should succeed");

    let sreqs = search.requests();
    assert_eq!(sreqs.len(), 2, "two search callouts, one per request");
    assert!(
        sreqs[0].headers.to_ascii_lowercase().contains("x-subscription-token: alice-brave-secret"),
        "alice's credential arrives on her callout: {}",
        sreqs[0].headers
    );
    assert!(
        !sreqs[0].headers.to_ascii_lowercase().contains("bob-brave-secret"),
        "bob's credential does not leak into alice's callout: {}",
        sreqs[0].headers
    );
    assert!(
        sreqs[1].headers.to_ascii_lowercase().contains("x-subscription-token: bob-brave-secret"),
        "bob's credential arrives on his callout: {}",
        sreqs[1].headers
    );
}
