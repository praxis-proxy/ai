// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the llm-d ext_proc routing example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_post, start_backend_with_shutdown, start_mock_routing_processor, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn llmd_ext_proc_routing_example_routes_to_processor_selected_endpoint() {
    let backend = start_backend_with_shutdown("llmd-selected-backend");
    let processor = start_mock_routing_processor(&format!("127.0.0.1:{}", backend.port()));
    let proxy_port = free_port();

    let config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", processor.port())]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"llmd-demo","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(status, 200, "example request should return 200");
    assert_eq!(
        body, "llmd-selected-backend",
        "request should route to the processor-selected backend"
    );
    assert!(
        processor.stream_count() >= 1,
        "example should invoke the ext_proc processor"
    );
}
