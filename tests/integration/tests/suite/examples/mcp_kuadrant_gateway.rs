// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Config-validity functional test for the mcp-kuadrant-gateway example config.
//!
//! The live end-to-end acceptance (per-user discovery filtering + per-call
//! authorization through a real Kuadrant MCP Gateway) runs in the gated Kuadrant
//! lane; see `tests/integration/kuadrant/` and
//! `tests/integration/sdk/openai/test_openai_responses_mcp_kuadrant.py`. This test
//! only proves the example config loads and its pipeline builds, so the example
//! stays valid under normal CI without the live gateway.

use std::collections::HashMap;

use praxis_test_utils::{free_port, load_example_config, start_proxy};

#[test]
fn example_config_loads_and_pipeline_builds() {
    let proxy_port = free_port();
    // The model endpoint and gateway connector are not dialed at startup, so the
    // pipeline builds without the live stack; mapping the model endpoint keeps the
    // rewrite explicit.
    let config = load_example_config(
        "openai/responses/mcp-kuadrant-gateway.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", free_port())]),
    );
    // start_proxy builds the full AI filter pipeline (incl. the MCP connector and
    // credential slots); an invalid connector/slot config would fail here.
    let _proxy = start_proxy(&config);

    // The Kuadrant connector + per-user Bearer slot (Topology A) must be wired on
    // BOTH openai_mcp_tool_resolve and openai_mcp_dispatch.
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/configs/openai/responses/mcp-kuadrant-gateway.yaml"
    ))
    .expect("example config present");
    assert!(
        raw.contains("id: trusted-mcp"),
        "connector id trusted-mcp must be declared"
    );
    assert_eq!(
        raw.matches("user_credential: mcp_gateway").count(),
        2,
        "per-user Bearer slot must be wired on both resolve and dispatch"
    );
}
