// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for Anthropic Messages web search with per-user callout
//! credentials and owner attribution.
//!
//! The `anthropic/web-search-scoped-credentials.yaml` example demonstrates PR1
//! of issue #880: `state_owner` maps trusted ingress identity into a normalized
//! owner, and `callout_credentials` captures a per-user provider key from the
//! ingress header `x-user-brave-key` into a slot, strips the header, and stages
//! the per-user secret into the Brave provider callout instead of the shared
//! api_key. A request missing the configured credential fails closed with a 401
//! authentication_error before any provider callout runs.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
};

use praxis_test_utils::{
    StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send, parse_body, parse_status,
    patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "anthropic/web-search-scoped-credentials.yaml";

/// Read the unified example and rewrite the proxy/backend ports and the search
/// provider endpoint so the loop calls the local stubs.
fn base_example_yaml(proxy_port: u16, model_port: u16, search_port: u16) -> String {
    let yaml =
        std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read web-search-scoped-credentials example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", model_port)]));
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

fn load_config(proxy_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    praxis_core::config::Config::from_yaml(&base_example_yaml(proxy_port, model_port, search_port))
        .expect("parse web-search-scoped-credentials example")
}

/// A Brave-shaped search response body stub.
struct SearchStub {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl SearchStub {
    fn start(response: &Value) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind search stub");
        let port = listener.local_addr().expect("stub address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let body = response.to_string();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept search request");
            let request = read_full_request(&mut stream);
            captured.lock().expect("capture search request").push(request);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).expect("write search response");
        });
        Self { port, requests }
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn request_count(&self) -> usize {
        self.requests.lock().expect("read search requests").len()
    }
}

fn read_full_request(stream: &mut std::net::TcpStream) -> String {
    use std::time::Duration;
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("stub read timeout should be set");
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = stream.read(&mut buffer).expect("request read should succeed");
        assert!(count > 0, "request must complete before the connection closes");
        request.extend_from_slice(&buffer[..count]);
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
            })
            .unwrap_or(0);
        if request.len() >= header_end + 4 + content_length {
            return String::from_utf8_lossy(&request).into_owned();
        }
    }
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
fn anthropic_web_search_scoped_credentials_example_builds() {
    let config = load_config(free_port(), 19_501, 19_502);
    let _pipeline = build_pipeline(&config);
}

#[test]
fn missing_credential_fails_closed_with_401_authentication_error() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
    ))
    .expect("parse web-search fixture");
    let first_model_response = fixture["first_model_response"].to_string();
    let model = StatefulCapturingBackend::new(vec![(200, first_model_response)]).start_with_shutdown();
    let search = SearchStub::start(&fixture["search_response"]);
    let proxy_port = free_port();
    let config = load_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "openai/gpt-oss-20b",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use web search to look up potato."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }]
    });
    let headers = [("x-auth-tenant", "acme"), ("x-auth-user", "alice")];

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/messages", &request.to_string(), &headers),
    );

    assert_eq!(
        parse_status(&raw),
        401,
        "a missing configured credential fails closed with 401: {raw}"
    );
    let body: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(body["type"], "error", "response carries an error envelope");
    assert_eq!(
        body["error"]["type"], "authentication_error",
        "a missing credential maps to authentication_error"
    );
    assert!(
        body["error"].get("code").is_none(),
        "Anthropic error envelope has no code field"
    );
    assert!(
        body["error"]["message"].as_str().unwrap().contains("brave_search"),
        "the error message names the missing slot: {}",
        body["error"]["message"]
    );
    assert_eq!(search.request_count(), 0, "no provider callout on a missing credential");
}

#[test]
fn per_user_credential_arrives_at_the_anthropic_provider_and_ingress_header_is_stripped() {
    // Mirror of the OpenAI positive test for the Anthropic Messages family: the
    // per-user credential captured in the outer chain is staged into the in-IRR
    // web-search callout (proving outer callout_credentials runs before the
    // in-step anthropic_web_search reads the slot), and the ingress header is
    // stripped from both the provider callout and the model backend.
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
    ))
    .expect("parse web-search fixture");
    let model = StatefulCapturingBackend::new(vec![
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let brave_body = json!({
        "web": {"results": [{
            "title": "Potato - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Potato",
            "description": "Potato is a starchy underground tuber native to the Americas."
        }]}
    });
    let search = StatefulCapturingBackend::new(vec![(200, brave_body.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "openai/gpt-oss-20b",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use web search to look up potato."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }]
    });
    let headers = [
        ("x-auth-tenant", "acme"),
        ("x-auth-user", "alice"),
        ("x-user-brave-key", "alice-brave-secret"),
    ];

    let raw = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/messages", &request.to_string(), &headers),
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
fn per_user_credentials_are_isolated_across_requests() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
    ))
    .expect("parse web-search fixture");
    let model = StatefulCapturingBackend::new(vec![
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let brave_body = json!({
        "web": {"results": [{
            "title": "Potato - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Potato",
            "description": "Potato is a starchy underground tuber native to the Americas."
        }]}
    });
    let search = StatefulCapturingBackend::new(vec![(200, brave_body.to_string()), (200, brave_body.to_string())])
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "openai/gpt-oss-20b",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Use web search to look up potato."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }]
    });

    let headers_alice = [
        ("x-auth-tenant", "acme"),
        ("x-auth-user", "alice"),
        ("x-user-brave-key", "alice-brave-secret"),
    ];
    let raw1 = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/messages", &request.to_string(), &headers_alice),
    );
    assert_eq!(parse_status(&raw1), 200, "alice request should succeed: {raw1}");

    let headers_bob = [
        ("x-auth-tenant", "acme"),
        ("x-auth-user", "bob"),
        ("x-user-brave-key", "bob-brave-secret"),
    ];
    let raw2 = http_send(
        proxy.addr(),
        &json_post_with_headers("/v1/messages", &request.to_string(), &headers_bob),
    );
    assert_eq!(parse_status(&raw2), 200, "bob request should succeed: {raw2}");

    let sreqs = search.requests();
    assert_eq!(sreqs.len(), 2, "two search callouts, one per request");
    assert!(
        sreqs[0]
            .headers
            .to_ascii_lowercase()
            .contains("x-subscription-token: alice-brave-secret"),
        "alice's credential arrives on her callout: {}",
        sreqs[0].headers
    );
    assert!(
        !sreqs[0].headers.to_ascii_lowercase().contains("bob-brave-secret"),
        "bob's credential does not leak into alice's callout: {}",
        sreqs[0].headers
    );
    assert!(
        sreqs[1]
            .headers
            .to_ascii_lowercase()
            .contains("x-subscription-token: bob-brave-secret"),
        "bob's credential arrives on his callout: {}",
        sreqs[1].headers
    );
}
