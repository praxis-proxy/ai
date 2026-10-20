// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the file-sourced web-search provider key example
//! configs.
//!
//! Both examples source the shared provider `api_key` from a projected `file:`
//! rather than `env_var:`. These tests prove the shipped configs parse and the
//! pipeline builds (reading the key file at startup), and that an unreadable key
//! file fails the build loud rather than dialing a provider unauthenticated.

use std::{
    collections::HashMap,
    fs,
    io::Write as _,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, allow_loopback_endpoints, example_config_path, free_port, http_send, json_post,
    parse_status, patch_yaml, start_proxy,
};
use serde_json::{Value, json};

/// The projected-file path both example configs reference for the provider key.
const KEY_FILE_PLACEHOLDER: &str = "/var/run/secrets/web-search/api-key";

/// Resolve the full AI pipeline from raw YAML, surfacing any build error.
///
/// Exercises the real server pipeline-resolution path, so a config whose
/// file-sourced `api_key` cannot be read fails here rather than at request time.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client = praxis_test_utils::test_subrequest_client();
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_ai::resolve_pipelines(
        &config,
        &registry,
        &health,
        &kv_stores,
        &praxis_ai::InFlightRegistry::new(),
        &subrequest_client,
    )
    .map(|_pipelines| ())
    .map_err(|error| error.to_string())
}

/// Read an example config and point its `api_key.file` at `path`.
fn example_yaml_with_key_file(example: &str, path: &str) -> String {
    let yaml = fs::read_to_string(example_config_path(example)).unwrap_or_else(|e| panic!("read {example}: {e}"));
    let patched = yaml.replace(KEY_FILE_PLACEHOLDER, path);
    assert_ne!(
        patched, yaml,
        "example drift: {KEY_FILE_PLACEHOLDER} not found in {example}"
    );
    patched
}

fn example_yaml_for_callout(
    example: &str,
    key_path: &str,
    listener_port: u16,
    model_port: u16,
    search_port: u16,
) -> String {
    let original_yaml =
        fs::read_to_string(example_config_path(example)).unwrap_or_else(|e| panic!("read {example}: {e}"));
    let yaml = original_yaml.replace(
        &format!("file: {KEY_FILE_PLACEHOLDER}"),
        &format!("file: {key_path}\n                base_url: http://127.0.0.1:{search_port}"),
    );
    assert_ne!(
        yaml, original_yaml,
        "example {example} did not contain the web-search key-file placeholder"
    );
    let model_endpoint = if example.starts_with("openai/") {
        "127.0.0.1:3001"
    } else {
        "127.0.0.1:8000"
    };
    let yaml = patch_yaml(&yaml, listener_port, &HashMap::from([(model_endpoint, model_port)]));
    let yaml = allow_loopback_endpoints(&yaml);
    yaml.replace(
        "allow_private_endpoints: true",
        "allow_private_endpoints: true\n  allow_private_upstreams: true",
    )
}

/// Each example, keyed by the path passed to `example_config_path`.
const EXAMPLES: &[&str] = &[
    "openai/responses/web-search-file-key.yaml",
    "anthropic/web-search-file-key.yaml",
];

#[test]
fn web_search_file_key_examples_build() {
    for example in EXAMPLES {
        let mut key = tempfile::NamedTempFile::new().unwrap();
        write!(key, "test-web-search-key").unwrap();
        let patched = example_yaml_with_key_file(example, key.path().to_str().unwrap());
        resolve(&patched).unwrap_or_else(|e| panic!("file-key example {example} must build: {e}"));
    }
}

#[test]
fn web_search_file_key_examples_fail_build_without_key_file() {
    // The examples reference a file that does not exist; building must fail loud
    // at startup rather than silently dialing the provider unauthenticated.
    for example in EXAMPLES {
        let patched = example_yaml_with_key_file(example, "/nonexistent/praxis-web-search-key-xyz");
        assert!(
            resolve(&patched).is_err(),
            "an unreadable key file must fail the pipeline build for {example}"
        );
    }
}

#[test]
fn file_sourced_web_search_key_reaches_provider_and_rotates() {
    const INITIAL_KEY: &str = "file-web-search-key-initial";
    const ROTATED_KEY: &str = "file-web-search-key-rotated";
    let brave_response = r#"{"web":{"results":[]}}"#;

    for example in EXAMPLES {
        let key_dir = tempfile::tempdir().unwrap();
        let key_path = key_dir.path().join("api-key");
        fs::write(&key_path, INITIAL_KEY).unwrap();

        let (first_model_response, final_model_response) = model_responses(example);
        let model_responses = (0..128)
            .flat_map(|_| [(200, first_model_response.clone()), (200, final_model_response.clone())])
            .collect();
        let model = StatefulCapturingBackend::new(model_responses).start_with_shutdown();
        let search = StatefulCapturingBackend::new(vec![(200, brave_response.to_owned()); 256]).start_with_shutdown();
        let proxy_port = free_port();
        let yaml = example_yaml_for_callout(
            example,
            key_path.to_str().unwrap(),
            proxy_port,
            model.port(),
            search.port(),
        );
        let config = Config::from_yaml(&yaml).unwrap_or_else(|e| panic!("parse {example}: {e}"));
        let proxy = start_proxy(&config);

        let raw = send_search_request(example, proxy.addr());
        assert_eq!(
            parse_status(&raw),
            200,
            "file-key web search should complete for {example}: {raw}"
        );
        assert_provider_key(&search, INITIAL_KEY, example);

        // Projected credentials replace the file atomically within its parent.
        let replacement_path = key_dir.path().join("api-key-next");
        fs::write(&replacement_path, ROTATED_KEY).unwrap();
        fs::rename(&replacement_path, &key_path).unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let raw = send_search_request(example, proxy.addr());
            assert_eq!(
                parse_status(&raw),
                200,
                "rotated file-key web search should complete: {raw}"
            );
            if provider_key_is_latest(&search, ROTATED_KEY) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "{example} provider did not observe the rotated file key"
            );
            thread::sleep(Duration::from_millis(25));
        }
        assert_provider_key(&search, ROTATED_KEY, example);
    }
}

fn model_responses(example: &str) -> (String, String) {
    if example.starts_with("openai/") {
        let first = json!({
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
        let final_response = json!({
            "id": "chatcmpl_answer",
            "object": "chat.completion",
            "model": "chat-only-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "The search completed."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
        });
        (first.to_string(), final_response.to_string())
    } else {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
        ))
        .expect("Anthropic web-search fixture JSON");
        (
            fixture["first_model_response"].to_string(),
            fixture["final_model_response"].to_string(),
        )
    }
}

fn send_search_request(example: &str, proxy_addr: &str) -> String {
    if example.starts_with("openai/") {
        let body = json!({
            "model": "chat-only-model",
            "input": "Find the latest Praxis Proxy release.",
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "web_search"},
            "store": false
        });
        http_send(proxy_addr, &json_post("/v1/responses", &body.to_string()))
    } else {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
        ))
        .expect("Anthropic web-search fixture JSON");
        http_send(
            proxy_addr,
            &json_post("/v1/messages", &fixture["initial_request"].to_string()),
        )
    }
}

fn provider_key_is_latest(search: &praxis_test_utils::StatefulCapturingGuard, expected: &str) -> bool {
    search
        .requests()
        .into_iter()
        .rev()
        .find(|request| request.uri.contains("/res/v1/web/search"))
        .is_some_and(|request| {
            request
                .headers
                .to_ascii_lowercase()
                .contains(&format!("x-subscription-token: {expected}"))
        })
}

fn assert_provider_key(search: &praxis_test_utils::StatefulCapturingGuard, expected: &str, example: &str) {
    let requests = search.requests();
    let provider_request = requests
        .iter()
        .rev()
        .find(|request| request.uri.contains("/res/v1/web/search"))
        .unwrap_or_else(|| {
            panic!(
                "{example} must call the Brave provider; captured {} requests",
                requests.len()
            )
        });
    assert!(
        provider_request
            .headers
            .to_ascii_lowercase()
            .contains(&format!("x-subscription-token: {expected}")),
        "{example} provider request must carry file-sourced key {expected}: {}",
        provider_request.headers
    );
}
