// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for trusted ownership of persisted OpenAI state.

use std::collections::HashMap;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use praxis_test_utils::{
    Backend, TempSqlite, example_config_path, free_port, http_send, parse_body, parse_status, patch_yaml,
    start_header_echo_backend, start_proxy,
};

const OWNER_HEADER: &str = "x-authenticated-state-owner";
const RESPONSE_JSON: &str = r#"{"id":"resp_owned","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"hello","output":[{"id":"msg_owned","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi"}]}]}"#;

fn assertion(tenant: &str, issuer: &str, subject: &str) -> String {
    let payload = serde_json::to_vec(&[tenant, issuer, subject]).unwrap();
    format!("v1.{}", URL_SAFE_NO_PAD.encode(payload))
}

fn request(method: &str, path: &str, body: Option<&str>, assertions: &[&str]) -> String {
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n");
    for value in assertions {
        request.push_str(&format!("{OWNER_HEADER}: {value}\r\n"));
    }
    if let Some(body) = body {
        request.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    request.push_str("Connection: close\r\n\r\n");
    if let Some(body) = body {
        request.push_str(body);
    }
    request
}

fn owned_config(db_url: &str, proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/state-ownership.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://owned-state.db?mode=rwc", db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_port)]),
    );
    praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse")
}

#[test]
fn validates_assertions_and_strips_the_transport_header() {
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let config = owned_config("sqlite::memory:", proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let owner = assertion("tenant-a", "https://issuer.example", "alice");

    let raw = http_send(
        proxy.addr(),
        &request(
            "POST",
            "/v1/responses",
            Some(r#"{"model":"gpt-4.1","input":"hello","store":false}"#),
            &[&owner],
        ),
    );
    assert_eq!(parse_status(&raw), 200, "valid assertion should pass: {raw}");
    assert!(
        !parse_body(&raw).to_ascii_lowercase().contains(OWNER_HEADER),
        "the owner assertion must be removed before the inference backend"
    );

    let cases = [
        (Vec::new(), 401),
        (vec![owner.as_str(), owner.as_str()], 400),
        (vec!["v1.***"], 400),
        (vec!["v2.abc"], 400),
    ];
    for (headers, expected_status) in cases {
        let raw = http_send(
            proxy.addr(),
            &request(
                "POST",
                "/v1/responses",
                Some(r#"{"model":"gpt-4.1","input":"hello"}"#),
                &headers,
            ),
        );
        assert_eq!(parse_status(&raw), expected_status, "unexpected result for {headers:?}");
    }

    let oversized = format!("v1.{}", "a".repeat(4_097));
    let raw = http_send(
        proxy.addr(),
        &request(
            "POST",
            "/v1/responses",
            Some(r#"{"model":"gpt-4.1","input":"hello"}"#),
            &[&oversized],
        ),
    );
    assert_eq!(parse_status(&raw), 400);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn responses_lifecycle_is_scoped_to_the_exact_owner() {
    let backend = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("owned_responses");
    let config = owned_config(db.url(), proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let alice = assertion("tenant-a", "issuer-a", "alice");
    let bob = assertion("tenant-a", "issuer-a", "bob");
    let other_tenant = assertion("tenant-b", "issuer-a", "alice");

    let raw = http_send(
        proxy.addr(),
        &request(
            "POST",
            "/v1/responses",
            Some(r#"{"model":"gpt-4.1","input":"hello"}"#),
            &[&alice],
        ),
    );
    assert_eq!(parse_status(&raw), 200, "owner should create a stored response: {raw}");

    for (path, owner) in [
        ("/v1/responses/resp_owned", bob.as_str()),
        ("/v1/responses/resp_owned/input_items", bob.as_str()),
        ("/v1/responses/resp_owned", other_tenant.as_str()),
    ] {
        let raw = http_send(proxy.addr(), &request("GET", path, None, &[owner]));
        assert_eq!(parse_status(&raw), 404, "foreign owner must not discover {path}");
    }

    let denied = http_send(proxy.addr(), &request("GET", "/v1/responses/resp_owned", None, &[&bob]));
    let raw = http_send(
        proxy.addr(),
        &request("DELETE", "/v1/responses/resp_owned", None, &[&bob]),
    );
    assert_eq!(parse_status(&raw), 404, "foreign delete must not mutate the response");

    let raw = http_send(
        proxy.addr(),
        &request("GET", "/v1/responses/resp_owned", None, &[&alice]),
    );
    assert_eq!(parse_status(&raw), 200, "owner should still retrieve the response");

    let continuation = r#"{"model":"gpt-4.1","input":"next","previous_response_id":"resp_owned"}"#;
    let raw = http_send(
        proxy.addr(),
        &request("POST", "/v1/responses", Some(continuation), &[&bob]),
    );
    assert_eq!(parse_status(&raw), 400, "foreign continuation must not rehydrate state");
    let raw = http_send(
        proxy.addr(),
        &request("POST", "/v1/responses", Some(continuation), &[&alice]),
    );
    assert_eq!(parse_status(&raw), 200, "owner continuation should succeed");

    let raw = http_send(
        proxy.addr(),
        &request("DELETE", "/v1/responses/resp_owned", None, &[&alice]),
    );
    assert_eq!(parse_status(&raw), 200, "owner should delete its response: {raw}");
    let now_missing = http_send(proxy.addr(), &request("GET", "/v1/responses/resp_owned", None, &[&bob]));
    assert_eq!(parse_status(&now_missing), 404);
    assert_eq!(
        parse_body(&denied),
        parse_body(&now_missing),
        "an unauthorized resource and the same nonexistent resource must be indistinguishable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conversation_lifecycle_rehydration_and_append_are_owner_scoped() {
    let backend = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("owned_conversations");
    let config = owned_config(db.url(), proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let alice = assertion("tenant-a", "issuer-a", "alice");
    let bob = assertion("tenant-a", "issuer-a", "bob");

    let raw = http_send(
        proxy.addr(),
        &request(
            "POST",
            "/v1/conversations",
            Some(
                r#"{"metadata":{"private":"yes"},"items":[{"id":"item_private","type":"message","role":"user","content":"first"}]}"#,
            ),
            &[&alice],
        ),
    );
    assert_eq!(parse_status(&raw), 200, "owner should create conversation: {raw}");
    let created: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    let conversation_id = created["id"].as_str().unwrap();

    for path in [
        format!("/v1/conversations/{conversation_id}"),
        format!("/v1/conversations/{conversation_id}/items"),
        format!("/v1/conversations/{conversation_id}/items/item_private"),
    ] {
        let raw = http_send(proxy.addr(), &request("GET", &path, None, &[&bob]));
        assert_eq!(parse_status(&raw), 404, "foreign owner must not discover {path}");
    }

    let raw = http_send(
        proxy.addr(),
        &request(
            "POST",
            &format!("/v1/conversations/{conversation_id}"),
            Some(r#"{"metadata":{"private":"no"}}"#),
            &[&bob],
        ),
    );
    assert_eq!(parse_status(&raw), 404, "foreign metadata update must fail");
    let raw = http_send(
        proxy.addr(),
        &request(
            "POST",
            &format!("/v1/conversations/{conversation_id}/items"),
            Some(r#"{"items":[{"id":"item_intruder","type":"message","role":"user","content":"intruder"}]}"#),
            &[&bob],
        ),
    );
    assert_eq!(parse_status(&raw), 404, "foreign item append must fail");

    let response_request = format!(r#"{{"model":"gpt-4.1","input":"next","conversation":"{conversation_id}"}}"#);
    let raw = http_send(
        proxy.addr(),
        &request("POST", "/v1/responses", Some(&response_request), &[&bob]),
    );
    assert_eq!(parse_status(&raw), 400, "foreign conversation must not rehydrate");
    let raw = http_send(
        proxy.addr(),
        &request("POST", "/v1/responses", Some(&response_request), &[&alice]),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "owner conversation should rehydrate and append"
    );

    let raw = http_send(
        proxy.addr(),
        &request("GET", &format!("/v1/conversations/{conversation_id}"), None, &[&alice]),
    );
    assert_eq!(parse_status(&raw), 200);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&parse_body(&raw)).unwrap()["metadata"]["private"],
        "yes"
    );

    let raw = http_send(
        proxy.addr(),
        &request(
            "GET",
            &format!("/v1/conversations/{conversation_id}/items?order=asc"),
            None,
            &[&alice],
        ),
    );
    assert_eq!(parse_status(&raw), 200);
    let items: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        items["data"].as_array().unwrap().len(),
        3,
        "initial item plus rehydrated input and model output should be synchronized"
    );

    let raw = http_send(
        proxy.addr(),
        &request("DELETE", &format!("/v1/conversations/{conversation_id}"), None, &[&bob]),
    );
    assert_eq!(parse_status(&raw), 404);
    let raw = http_send(
        proxy.addr(),
        &request("GET", &format!("/v1/conversations/{conversation_id}"), None, &[&alice]),
    );
    assert_eq!(parse_status(&raw), 200, "foreign delete must not mutate conversation");
}
