// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the token_count filter example configuration.
//!
//! The filter writes to `filter_metadata` which is not observable from
//! an HTTP response, so we only verify the proxy starts and proxies
//! traffic correctly. Token extraction correctness is covered by unit
//! tests in `praxis-ai-filters`.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_status, start_backend_with_shutdown, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn token_count_proxies_response() {
    let backend_port_guard = start_backend_with_shutdown("ok");
    let backend_port = backend_port_guard.port();
    let proxy_port = free_port();
    let config = super::load_example_config(
        "token-counting.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
}
