// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional test for the file-resolve example config.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::TcpListener,
    time::Duration,
};

use praxis_test_utils::{
    free_port, http_get, http_send, json_post, load_example_config, parse_body, parse_status,
    start_backend_with_shutdown, start_capturing_backend, start_proxy,
};

const FILE_METADATA: &str = r#"{"id":"test-file-123","object":"file","bytes":13,"created_at":1750000000,"filename":"test.txt","purpose":"user_data"}"#;
const IMAGE_METADATA: &str =
    r#"{"id":"img-456","object":"file","bytes":4,"created_at":1750000000,"filename":"photo.png","purpose":"vision"}"#;
const OVERSIZED_METADATA: &str = r#"{"id":"huge-file","object":"file","bytes":999999999,"created_at":1750000000,"filename":"huge.bin","purpose":"user_data"}"#;
const FILE_CONTENT: &str = "Hello, world!";
const IMAGE_CONTENT: &[u8] = b"\x89PNG";

#[test]
fn example_config_resolves_input_file_to_openai_shape() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "test-file-123"}
                ]
            }
        ]
    }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should forward the resolved request");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    // https://developers.openai.com/api/docs/guides/file-inputs#base64-encoded-files
    assert_eq!(
        captured,
        serde_json::json!({
            "model": "gpt-4.1",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "filename": "test.txt",
                    "file_data": "SGVsbG8sIHdvcmxkIQ=="
                }]
            }]
        }),
        "upstream request should match the OpenAI inline input_file shape"
    );
}

#[test]
fn example_config_resolves_input_image_to_data_url() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "file_id": "img-456"}
                ]
            }
        ]
    }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should forward the resolved request");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "gpt-4.1",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw=="
                }]
            }]
        }),
        "upstream request should match the OpenAI inline input_image shape"
    );
}

#[test]
fn example_config_rejects_missing_file_with_on_missing_reject() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "nonexistent-file"}
                ]
            }
        ]
    }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        502,
        "missing file should return 502 when on_missing is reject"
    );
    let error_body: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("error response should be valid JSON");
    assert_eq!(
        error_body["error"]["type"].as_str().unwrap(),
        "file_resolve_error",
        "error type should be file_resolve_error"
    );
}

#[test]
fn example_config_rejects_oversized_file_from_metadata() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "huge-file"}
                ]
            }
        ]
    }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        413,
        "oversized file should be rejected with 413 based on metadata bytes"
    );
    let error_body: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("error response should be valid JSON");
    assert_eq!(
        error_body["error"]["type"].as_str().unwrap(),
        "file_resolve_error",
        "error type should be file_resolve_error"
    );
}

#[test]
fn example_config_non_responses_traffic_passes_through() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "non-responses traffic should pass through");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "non-responses traffic should route to default backend"
    );
}

#[test]
fn example_config_routes_files_api_paths_to_files_backend() {
    let files_guard = start_backend_with_shutdown("files-api");
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_guard.port()),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let upload_body = "--test-boundary\r\n\
        Content-Disposition: form-data; name=\"purpose\"\r\n\r\n\
        assistants\r\n\
        --test-boundary\r\n\
        Content-Disposition: form-data; name=\"file\"; filename=\"test.txt\"\r\n\
        Content-Type: text/plain\r\n\r\n\
        uploaded file contents\r\n\
        --test-boundary--\r\n";
    let upload = format!(
        "POST /v1/files HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: multipart/form-data; boundary=test-boundary\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {upload_body}",
        upload_body.len()
    );
    let upload_raw = http_send(proxy.addr(), &upload);
    assert_eq!(parse_status(&upload_raw), 200, "Files API root should be proxied");
    assert_eq!(
        parse_body(&upload_raw),
        "files-api",
        "POST /v1/files should route to Files API backend"
    );
    let (content_status, content_body) = http_get(proxy.addr(), "/v1/files/file-123/content", None);
    assert_eq!(content_status, 200, "Files API subresource should be proxied");
    assert_eq!(
        content_body, "files-api",
        "Files API subresources should route to Files API backend"
    );

    let (non_files_status, non_files_body) = http_get(proxy.addr(), "/v1/filesystem", None);
    assert_eq!(non_files_status, 200, "non-Files API path should be proxied");
    assert_eq!(
        non_files_body, "default-backend",
        "path-prefix matching must stop at a segment boundary"
    );
}

#[test]
fn example_config_forwards_headers_to_files_api() {
    let files_api_port = start_files_api_stub_auth_required();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{
        "model": "gpt-4.1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": "test-file-123"}
                ]
            }
        ]
    }"#;
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Authorization: Bearer test-token\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "Files API requires auth; 200 proves authorization header was forwarded"
    );

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");
    assert_eq!(
        captured["input"][0]["content"][0]["file_data"].as_str().unwrap(),
        "SGVsbG8sIHdvcmxkIQ==",
        "file should be resolved after auth header forwarding"
    );
}

fn start_files_api_stub_auth_required() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || handle_files_api_request_auth(stream));
        }
    });

    port
}

pub(super) fn start_files_api_stub() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || handle_files_api_request(stream));
        }
    });

    port
}

fn handle_files_api_request_auth(mut stream: std::net::TcpStream) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        if data.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    let raw = String::from_utf8_lossy(&data);
    let has_auth = raw
        .lines()
        .any(|line| line.to_ascii_lowercase().starts_with("authorization:"));
    if !has_auth {
        let body = br#"{"error":"unauthorized"}"#;
        let header = format!(
            "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _sent = stream.write_all(header.as_bytes());
        let _sent = stream.write_all(body);
        return;
    }

    let path = raw
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    let (status, content_type, body_bytes): (u16, &str, Vec<u8>) =
        if path.ends_with("/content") && path.contains("test-file-123") {
            (200, "text/plain", FILE_CONTENT.as_bytes().to_vec())
        } else if path.contains("test-file-123") {
            (200, "application/json", FILE_METADATA.as_bytes().to_vec())
        } else {
            (404, "application/json", br#"{"error":"not found"}"#.to_vec())
        };

    let header = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        if status == 200 { "OK" } else { "Not Found" },
        body_bytes.len()
    );
    let _sent = stream.write_all(header.as_bytes());
    let _sent = stream.write_all(&body_bytes);
}

fn handle_files_api_request(mut stream: std::net::TcpStream) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        if data.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    let raw = String::from_utf8_lossy(&data);
    let path = raw
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    let (status, content_type, body_bytes): (u16, &str, Vec<u8>) =
        if path.ends_with("/content") && path.contains("test-file-123") {
            (200, "text/plain", FILE_CONTENT.as_bytes().to_vec())
        } else if path.contains("test-file-123") {
            (200, "application/json", FILE_METADATA.as_bytes().to_vec())
        } else if path.ends_with("/content") && path.contains("img-456") {
            (200, "image/png", IMAGE_CONTENT.to_vec())
        } else if path.contains("img-456") {
            (200, "application/json", IMAGE_METADATA.as_bytes().to_vec())
        } else if path.contains("huge-file") {
            (200, "application/json", OVERSIZED_METADATA.as_bytes().to_vec())
        } else {
            (404, "application/json", br#"{"error":"not found"}"#.to_vec())
        };

    let header = format!(
        "HTTP/1.1 {status} {}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        if status == 200 { "OK" } else { "Not Found" },
        body_bytes.len()
    );
    let _sent = stream.write_all(header.as_bytes());
    let _sent = stream.write_all(&body_bytes);
}

#[test]
fn example_config_resolves_file_url_to_data_uri() {
    let files_api_port = start_files_api_stub();
    let file_url_port = start_file_url_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [file-resolve-pipeline]

filter_chains:
  - name: file-resolve-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format
          model: x-praxis-ai-model

      - filter: openai_file_resolve
        files_api_url: "http://127.0.0.1:{files_api_port}"
        allow_private_files_api_url: true
        allow_pre_security_callout: true
        file_url: resolve
        allowed_file_url_origins:
          - "http://127.0.0.1:{file_url_port}"
        on_missing: reject
        timeout_ms: 10000

      - filter: router
        routes:
          - path: "/v1/responses"
            cluster: "inference-backend"
          - path_prefix: "/"
            cluster: "default-backend"

      - filter: load_balancer
        clusters:
          - name: "inference-backend"
            endpoints:
              - "127.0.0.1:{}"
          - name: "default-backend"
            endpoints:
              - "127.0.0.1:{}"
"#,
        inference_guard.port(),
        default_guard.port()
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).expect("config should parse");
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{
            "model": "gpt-4.1",
            "input": [
                {{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {{"type": "input_file", "file_url": "http://127.0.0.1:{file_url_port}/document.txt"}}
                    ]
                }}
            ]
        }}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "proxy should forward the resolved request");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured["input"][0]["content"][0]["type"].as_str().unwrap(),
        "input_file",
        "content type should be input_file"
    );
    assert_eq!(
        captured["input"][0]["content"][0]["file_data"].as_str().unwrap(),
        "data:text/plain;base64,SGVsbG8sIHdvcmxkIQ==",
        "file_url should be resolved to data URI in file_data"
    );
    assert!(
        captured["input"][0]["content"][0]["file_url"].is_null(),
        "file_url should be removed after resolution"
    );
}

#[test]
fn file_url_passthrough_reaches_upstream_unchanged() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let proxy_port = free_port();

    let yaml = format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [file-resolve-pipeline]

filter_chains:
  - name: file-resolve-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format

      - filter: openai_file_resolve
        files_api_url: "http://127.0.0.1:{files_api_port}"
        allow_private_files_api_url: true
        allow_pre_security_callout: true
        file_url: passthrough
        on_missing: reject

      - filter: router
        routes:
          - path: "/v1/responses"
            cluster: "inference-backend"

      - filter: load_balancer
        clusters:
          - name: "inference-backend"
            endpoints:
              - "127.0.0.1:{}"
"#,
        inference_guard.port()
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).expect("config should parse");
    let proxy = start_proxy(&config);

    let file_url = "https://storage.example.com/document.pdf?sig=opaque";
    let body = format!(
        r#"{{
            "model": "gpt-4.1",
            "input": [{{
                "type": "message",
                "role": "user",
                "content": [{{"type": "input_file", "file_url": "{file_url}"}}]
            }}]
        }}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(parse_status(&raw), 200, "passthrough request should reach the backend");
    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");
    assert_eq!(
        captured["input"][0]["content"][0]["file_url"].as_str(),
        Some(file_url),
        "file_url should reach the upstream unchanged"
    );
    assert!(
        captured["input"][0]["content"][0].get("file_data").is_none(),
        "passthrough mode should not synthesize file_data"
    );
}

#[test]
fn example_config_rejects_ssrf_blocked_file_url_with_403() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/file-resolve.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);
    let body = r#"{
        "model": "gpt-4.1",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "file_url": "http://169.254.169.254/latest/meta-data/"
            }]
        }]
    }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        403,
        "metadata file_url should be rejected before proxying"
    );
    let error: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("error response should be valid JSON");
    assert_eq!(
        error["error"]["type"].as_str(),
        Some("file_resolve_error"),
        "blocked file_url should use the file resolution error envelope"
    );
    assert!(
        error["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("blocked by security policy")),
        "blocked file_url response should explain the security rejection"
    );
}

#[test]
fn example_config_file_url_no_credentials_forwarded() {
    let files_api_port = start_files_api_stub_auth_required();
    let file_url_port = start_file_url_stub_auth_check();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = format!(
        r#"
listeners:
  - name: ai-gateway
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [file-resolve-pipeline]

filter_chains:
  - name: file-resolve-pipeline
    filters:
      - filter: openai_responses_format
        on_invalid: continue
        headers:
          format: x-praxis-ai-format
          model: x-praxis-ai-model

      - filter: openai_file_resolve
        files_api_url: "http://127.0.0.1:{files_api_port}"
        allow_private_files_api_url: true
        allow_pre_security_callout: true
        file_url: resolve
        allowed_file_url_origins:
          - "http://127.0.0.1:{file_url_port}"
        forward_headers:
          - authorization
        on_missing: reject
        timeout_ms: 10000

      - filter: router
        routes:
          - path: "/v1/responses"
            cluster: "inference-backend"
          - path_prefix: "/"
            cluster: "default-backend"

      - filter: load_balancer
        clusters:
          - name: "inference-backend"
            endpoints:
              - "127.0.0.1:{}"
          - name: "default-backend"
            endpoints:
              - "127.0.0.1:{}"
"#,
        inference_guard.port(),
        default_guard.port()
    );
    let config = praxis_core::config::Config::from_yaml(&yaml).expect("config should parse");
    let proxy = start_proxy(&config);

    let body = format!(
        r#"{{
            "model": "gpt-4.1",
            "input": [
                {{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {{"type": "input_file", "file_id": "test-file-123"}},
                        {{"type": "input_file", "file_url": "http://127.0.0.1:{file_url_port}/document.txt"}}
                    ]
                }}
            ]
        }}"#
    );
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Authorization: Bearer test-token\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(
        parse_status(&raw),
        200,
        "request should succeed: Files API gets Authorization, file URL stub does not"
    );

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");
    assert_eq!(
        captured["input"][0]["content"][0]["file_data"].as_str().unwrap(),
        "SGVsbG8sIHdvcmxkIQ==",
        "file_id should be resolved with auth header forwarding"
    );
    assert_eq!(
        captured["input"][0]["content"][1]["file_data"].as_str().unwrap(),
        "data:text/plain;base64,SGVsbG8sIHdvcmxkIQ==",
        "file_url should be resolved without auth header forwarding"
    );
}

pub(super) fn start_file_url_stub() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || handle_file_url_request(stream));
        }
    });

    port
}

fn start_file_url_stub_auth_check() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || handle_file_url_request_auth_check(stream));
        }
    });

    port
}

fn handle_file_url_request(mut stream: std::net::TcpStream) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        if data.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    let body = FILE_CONTENT.as_bytes();
    let header = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );
    let _sent = stream.write_all(header.as_bytes());
    let _sent = stream.write_all(body);
}

fn handle_file_url_request_auth_check(mut stream: std::net::TcpStream) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        if data.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    let raw = String::from_utf8_lossy(&data);
    let has_auth = raw
        .lines()
        .any(|line| line.to_ascii_lowercase().starts_with("authorization:"));

    if has_auth {
        let body = br#"{"error":"credentials leaked to file URL"}"#;
        let header = format!(
            "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _sent = stream.write_all(header.as_bytes());
        let _sent = stream.write_all(body);
        return;
    }

    let body = FILE_CONTENT.as_bytes();
    let header = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/plain\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len()
    );
    let _sent = stream.write_all(header.as_bytes());
    let _sent = stream.write_all(body);
}
