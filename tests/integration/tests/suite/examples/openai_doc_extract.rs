// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the doc-extract example config.
//!
//! Each test verifies the exact JSON shape the inference backend
//! receives after the `file_resolve → doc_extract` pipeline, proving
//! vLLM-compatible output rather than only inspecting individual
//! fields.

use std::collections::HashMap;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_capturing_backend, start_proxy,
};

use super::openai_file_resolve::{start_file_url_stub, start_files_api_stub};

fn text_file_data(text: &str) -> String {
    format!("data:text/plain;base64,{}", BASE64.encode(text.as_bytes()))
}

fn json_file_data(text: &str) -> String {
    format!("data:application/json;base64,{}", BASE64.encode(text.as_bytes()))
}

fn pdf_file_data() -> String {
    format!("data:application/pdf;base64,{}", BASE64.encode(b"%PDF-1.4"))
}

fn setup_proxy(files_api_port: u16, inference_port: u16, default_port: u16) -> praxis_test_utils::ProxyGuard {
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/doc-extract.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", inference_port),
            ("127.0.0.1:3002", default_port),
        ]),
    );
    start_proxy(&config)
}

fn setup_proxy_with_file_url(
    files_api_port: u16,
    file_url_port: u16,
    inference_port: u16,
    default_port: u16,
) -> praxis_test_utils::ProxyGuard {
    let proxy_port = free_port();
    let path = praxis_test_utils::example_config_path("openai/responses/doc-extract.yaml");
    let yaml = std::fs::read_to_string(path).expect("doc-extract example should be readable");
    let file_url_policy = format!(
        "        file_url: resolve\n        allowed_file_url_origins:\n          - \"http://127.0.0.1:{file_url_port}\""
    );
    let yaml = yaml
        .replace("127.0.0.1:8080", &format!("127.0.0.1:{proxy_port}"))
        .replace("127.0.0.1:9999", &format!("127.0.0.1:{files_api_port}"))
        .replace("127.0.0.1:3001", &format!("127.0.0.1:{inference_port}"))
        .replace("127.0.0.1:3002", &format!("127.0.0.1:{default_port}"))
        .replace("        file_url: passthrough", &file_url_policy);
    let config = praxis_core::config::Config::from_yaml(&yaml).expect("patched doc-extract config should parse");
    start_proxy(&config)
}

// -- Inline file_data extraction (data-URI) -----------------------------------

#[test]
fn text_file_extracted_to_input_text() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "filename": "notes.txt",
                "file_data": text_file_data("Hello from file!")
            }]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "proxy should forward the extracted request");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "[Source: notes.txt]\nHello from file!"
                }]
            }]
        }),
        "backend should receive exact vLLM-compatible input_text shape"
    );
}

#[test]
fn json_file_extracted_to_input_text() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "filename": "data.json",
                "file_data": json_file_data(r#"{"key":"value"}"#)
            }]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "proxy should forward the extracted request");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "[Source: data.json]\n{\"key\":\"value\"}"
                }]
            }]
        }),
        "backend should receive exact vLLM-compatible input_text shape for JSON"
    );
}

// -- Resolved file_id extraction ----------------------------------------------

#[test]
fn resolved_file_id_extracted_to_input_text() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "file_id": "test-file-123"
            }]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(
        parse_status(&raw),
        200,
        "file_resolve should resolve file_id and doc_extract should extract text"
    );

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "[Source: test.txt]\nHello, world!"
                }]
            }]
        }),
        "file_id should be resolved by file_resolve, then extracted to input_text by doc_extract"
    );
}

#[test]
fn resolved_file_url_extracted_to_input_text() {
    let files_api_port = start_files_api_stub();
    let file_url_port = start_file_url_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy_with_file_url(
        files_api_port,
        file_url_port,
        inference_guard.port(),
        default_guard.port(),
    );

    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "file_url": format!("http://127.0.0.1:{file_url_port}/document.txt")
            }]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(
        parse_status(&raw),
        200,
        "file_url should resolve and extract before reaching the backend"
    );
    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");
    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "[Source: document.txt]\nHello, world!"
                }]
            }]
        }),
        "backend should receive input_text produced from fetched file_url content"
    );
}

// -- file_url passthrough (no network I/O in doc_extract) ---------------------

#[test]
fn file_url_without_file_data_passes_through() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "file_url": "https://example.com/report.txt"
            }]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "file_url-only part should pass through");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "file_url": "https://example.com/report.txt"
                }]
            }]
        }),
        "file_url without file_data should pass through unchanged (no network I/O)"
    );
}

// -- Unsupported / passthrough ------------------------------------------------

#[test]
fn unsupported_file_left_unchanged() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let file_data = pdf_file_data();
    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_file",
                "filename": "doc.pdf",
                "file_data": file_data
            }]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "unsupported file should pass through");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "filename": "doc.pdf",
                    "file_data": file_data
                }]
            }]
        }),
        "PDF should remain as input_file with on_unsupported: continue"
    );
}

#[test]
fn non_responses_traffic_passes_through() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_backend_with_shutdown("inference-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

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
fn string_input_passes_through() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let body = r#"{"model":"test","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "string input should reach the backend");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": "Hello, world!"
        }),
        "string input should pass through unchanged"
    );
}

// -- Mixed content / comprehensive pipeline -----------------------------------

#[test]
fn mixed_content_extracts_text_only() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let pdf_data = pdf_file_data();
    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "notes.txt",
                    "file_data": text_file_data("extractable")
                },
                {
                    "type": "input_file",
                    "filename": "doc.pdf",
                    "file_data": pdf_data
                },
                {
                    "type": "input_text",
                    "text": "question"
                }
            ]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "mixed content request should succeed");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    assert_eq!(
        captured,
        serde_json::json!({
            "model": "test",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "[Source: notes.txt]\nextractable"},
                    {"type": "input_file", "filename": "doc.pdf", "file_data": pdf_data},
                    {"type": "input_text", "text": "question"}
                ]
            }]
        }),
        "text file should be extracted, PDF unchanged, input_text preserved"
    );
}

#[test]
fn full_pipeline_file_id_inline_data_and_file_url() {
    let files_api_port = start_files_api_stub();
    let inference_guard = start_capturing_backend("{}");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy = setup_proxy(files_api_port, inference_guard.port(), default_guard.port());

    let pdf_data = pdf_file_data();
    let body = serde_json::json!({
        "model": "test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": "test-file-123"},
                {"type": "input_file", "filename": "data.json",
                 "file_data": json_file_data(r#"{"key":"val"}"#)},
                {"type": "input_file", "filename": "doc.pdf",
                 "file_data": pdf_data},
                {"type": "input_file",
                 "file_url": "https://example.com/report.txt"},
                {"type": "input_text", "text": "Summarize all documents"}
            ]
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));

    assert_eq!(parse_status(&raw), 200, "full pipeline should succeed");

    let captured: serde_json::Value =
        serde_json::from_str(&inference_guard.body()).expect("captured body should be valid JSON");

    let parts = captured["input"][0]["content"]
        .as_array()
        .expect("content should be an array");

    assert_eq!(parts.len(), 5, "all five content parts should be present");

    assert_eq!(
        parts[0],
        serde_json::json!({
            "type": "input_text",
            "text": "[Source: test.txt]\nHello, world!"
        }),
        "resolved file_id text file should become input_text"
    );

    assert_eq!(
        parts[1],
        serde_json::json!({
            "type": "input_text",
            "text": "[Source: data.json]\n{\"key\":\"val\"}"
        }),
        "inline JSON data URI should become input_text"
    );

    assert_eq!(
        parts[2],
        serde_json::json!({
            "type": "input_file",
            "filename": "doc.pdf",
            "file_data": pdf_data
        }),
        "unsupported PDF should remain as input_file"
    );

    assert_eq!(
        parts[3],
        serde_json::json!({
            "type": "input_file",
            "file_url": "https://example.com/report.txt"
        }),
        "file_url without file_data should pass through unchanged"
    );

    assert_eq!(
        parts[4],
        serde_json::json!({
            "type": "input_text",
            "text": "Summarize all documents"
        }),
        "original input_text should be preserved"
    );

    for (i, part) in parts.iter().enumerate() {
        if part["type"] == "input_text" {
            assert!(
                part.get("file_data").is_none(),
                "part {i}: input_text must not have file_data"
            );
            assert!(
                part.get("file_id").is_none(),
                "part {i}: input_text must not have file_id"
            );
            assert!(
                part.get("file_url").is_none(),
                "part {i}: input_text must not have file_url"
            );
        }
    }
}

// -- Live vLLM downstream compatibility ---------------------------------------
//
// Downstream compatibility (issue #397) is tested against actual
// vLLM inference in the SDK test harness:
//
//   tests/integration/sdk/openai/test_openai_responses_vllm.py
//     → test_doc_extract_inline_file_to_input_text
//
// That test sends an input_file with inline file_data through the
// full proxy pipeline (file_resolve → doc_extract → responses_proxy
// → vLLM), then asserts a unique marker from the document appears
// in the model's actual output. This proves vLLM consumes the
// extracted text end-to-end.
//
// The Rust tests above verify proxy rewriting correctness via exact
// JSON equality; the Python SDK test covers live vLLM acceptance.
