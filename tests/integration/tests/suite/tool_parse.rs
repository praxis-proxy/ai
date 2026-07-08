// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `tool_parse` filter.

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, start_backend_with_shutdown, start_echo_backend,
    start_proxy,
};

// =============================================================================
// Classification and Routing (branch_chains on filter results)
// =============================================================================

#[test]
fn request_with_tools_routes_to_tools_cluster() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request with tools should return 200");
    assert_eq!(
        parse_body(&raw),
        "tools-backend",
        "request with tools should route to tools cluster"
    );
}

#[test]
fn request_without_tools_routes_to_default_cluster() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","model":"gpt-4.1"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request without tools should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "request without tools should route to default cluster"
    );
}

#[test]
fn web_search_tool_routes_to_tools_cluster() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"web_search","search_context_size":"high"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "web_search request should return 200");
    assert_eq!(
        parse_body(&raw),
        "tools-backend",
        "web_search tool should route to tools cluster"
    );
}

#[test]
fn web_search_2025_08_26_routes_to_tools_cluster() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"web_search_2025_08_26"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "web_search_2025_08_26 should return 200");
    assert_eq!(
        parse_body(&raw),
        "tools-backend",
        "web_search_2025_08_26 variant should route to tools cluster"
    );
}

#[test]
fn file_search_tool_routes_to_tools_cluster() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "file_search request should return 200");
    assert_eq!(
        parse_body(&raw),
        "tools-backend",
        "file_search tool should route to tools cluster"
    );
}

#[test]
fn code_interpreter_tool_routes_to_tools_cluster() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"code_interpreter"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "code_interpreter request should return 200");
    assert_eq!(
        parse_body(&raw),
        "tools-backend",
        "code_interpreter tool should route to tools cluster"
    );
}

#[test]
fn empty_tools_array_routes_to_default() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "empty tools should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "empty tools array should route to default cluster"
    );
}

// =============================================================================
// Non-Responses Path (must not classify)
// =============================================================================

#[test]
fn non_responses_path_not_classified() {
    let tools_guard = start_backend_with_shutdown("tools-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = routing_yaml(proxy_port, tools_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "non-Responses path should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "tools on a non-Responses path should not trigger tool routing"
    );
}

// =============================================================================
// Tool Choice Routing (branch_chains on filter results)
// =============================================================================

#[test]
fn tool_choice_routes_via_filter_result() {
    let required_guard = start_backend_with_shutdown("required-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = tool_choice_routing_yaml(proxy_port, required_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tool_choice":"required","tools":[{"type":"function","name":"f"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "required tool_choice should return 200");
    assert_eq!(
        parse_body(&raw),
        "required-backend",
        "tool_choice=required should route via filter result branch"
    );
}

#[test]
fn allowed_tools_required_routes_via_filter_result() {
    let required_guard = start_backend_with_shutdown("required-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = tool_choice_routing_yaml(proxy_port, required_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{
        "input":"test",
        "tool_choice":{
            "type":"allowed_tools",
            "mode":"required",
            "tools":[{"type":"function","name":"f"}]
        },
        "tools":[{"type":"function","name":"f"}]
    }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "allowed_tools required tool_choice should return 200"
    );
    assert_eq!(
        parse_body(&raw),
        "required-backend",
        "allowed_tools mode=required should route via filter result branch"
    );
}

#[test]
fn auto_tool_choice_falls_through() {
    let required_guard = start_backend_with_shutdown("required-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = tool_choice_routing_yaml(proxy_port, required_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tool_choice":"auto","tools":[{"type":"function","name":"f"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "auto tool_choice should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "tool_choice=auto should not match required route"
    );
}

// =============================================================================
// Body Preservation
// =============================================================================

#[test]
fn body_preserved_after_classification() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let yaml = echo_yaml(proxy_port, backend_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"function","name":"calc","parameters":{"type":"object"}}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "should return 200");
    assert_eq!(parse_body(&raw), body, "body should be preserved unchanged");
}

// =============================================================================
// Branch Condition
// =============================================================================

#[test]
fn branch_on_has_web_search() {
    let web_guard = start_backend_with_shutdown("web-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = branch_yaml(proxy_port, web_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"web_search"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "web_search branch should return 200");
    assert_eq!(
        parse_body(&raw),
        "web-backend",
        "web_search should branch to web-backend"
    );
}

#[test]
fn branch_no_web_search_falls_through() {
    let web_guard = start_backend_with_shutdown("web-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = branch_yaml(proxy_port, web_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"function","name":"f"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "no web_search should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "function-only tools should fall through to default"
    );
}

#[test]
fn branch_on_has_web_search_after_preceding_filter() {
    let web_guard = start_backend_with_shutdown("web-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = chained_branch_yaml(proxy_port, web_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"web_search"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "chained branch should return 200");
    assert_eq!(
        parse_body(&raw),
        "web-backend",
        "web_search branch should fire even after a preceding filter"
    );
}

#[test]
fn branch_no_web_search_falls_through_after_preceding_filter() {
    let web_guard = start_backend_with_shutdown("web-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let yaml = chained_branch_yaml(proxy_port, web_guard.port(), default_guard.port());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"test","tools":[{"type":"function","name":"f"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "chained no-web_search should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "function-only tools should fall through even after a preceding filter"
    );
}

// =============================================================================
// YAML Helpers
// =============================================================================

/// Branches requests with tools to one cluster, others to default.
fn routing_yaml(proxy_port: u16, tools_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: tool_parse
        branch_chains:
          - name: has_tools_branch
            on_result:
              filter: tool_parse
              key: has_tools
              result: "true"
            rejoin: terminal
            chains:
              - name: tools_chain
                filters:
                  - filter: router
                    routes:
                      - path_prefix: "/"
                        cluster: "tools"
                  - filter: load_balancer
                    clusters:
                      - name: "tools"
                        endpoints:
                          - "127.0.0.1:{tools_port}"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
"#
    )
}

/// Branches by tool_choice filter result value.
fn tool_choice_routing_yaml(proxy_port: u16, required_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: tool_parse
        branch_chains:
          - name: required_branch
            on_result:
              filter: tool_parse
              key: tool_choice
              result: "required"
            rejoin: terminal
            chains:
              - name: required_chain
                filters:
                  - filter: router
                    routes:
                      - path_prefix: "/"
                        cluster: "required"
                  - filter: load_balancer
                    clusters:
                      - name: "required"
                        endpoints:
                          - "127.0.0.1:{required_port}"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
"#
    )
}

/// Routes all traffic to an echo (body) backend.
fn echo_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: tool_parse
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
"#
    )
}

/// Branch on has_web_search with `openai_responses_format` preceding `tool_parse`.
///
/// Verifies that filter_results and branch conditions work correctly
/// when tool_parse is NOT the first body-reading filter in the chain.
fn chained_branch_yaml(proxy_port: u16, web_port: u16, default_port: u16) -> String {
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
      - filter: tool_parse
        branch_chains:
          - name: web_search_branch
            on_result:
              filter: tool_parse
              key: has_web_search
              result: "true"
            rejoin: terminal
            chains:
              - name: web_chain
                filters:
                  - filter: router
                    routes:
                      - path_prefix: "/"
                        cluster: "web"
                  - filter: load_balancer
                    clusters:
                      - name: "web"
                        endpoints:
                          - "127.0.0.1:{web_port}"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
"#
    )
}

/// Branch on has_web_search filter result.
fn branch_yaml(proxy_port: u16, web_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: tool_parse
        branch_chains:
          - name: web_search_branch
            on_result:
              filter: tool_parse
              key: has_web_search
              result: "true"
            rejoin: terminal
            chains:
              - name: web_chain
                filters:
                  - filter: router
                    routes:
                      - path_prefix: "/"
                        cluster: "web"
                  - filter: load_balancer
                    clusters:
                      - name: "web"
                        endpoints:
                          - "127.0.0.1:{web_port}"
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
"#
    )
}
