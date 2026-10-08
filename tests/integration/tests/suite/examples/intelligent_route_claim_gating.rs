// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Example test for `intelligent-route-claim-gating.yaml`.
//!
//! The example fences candidates by identity claims (`match_claims`). It runs no
//! auth filter, so no authenticated identity is published, and the fence must
//! fail closed: the request is denied before any upstream is contacted.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, load_example_config, parse_status, start_proxy, start_stateful_backend};

const CLAIM_GATING_EXAMPLE: &str = "intelligent-route-claim-gating.yaml";

/// Endpoints as written in `intelligent-route-claim-gating.yaml`.
const SITE_US_A: &str = "127.0.0.1:8001";
const SITE_US_B: &str = "127.0.0.1:8002";
const SITE_EU: &str = "127.0.0.1:8003";
const SITE_UK: &str = "127.0.0.1:8004";

fn chat_post(body: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

/// A claim gate with no authenticated identity present fails closed: the request
/// is denied (403) before any upstream is contacted.
#[test]
fn claim_gate_without_identity_fails_closed() {
    let us_a = start_stateful_backend(vec![(200, "site-us-a".to_owned())]);
    let us_b = start_stateful_backend(vec![(200, "site-us-b".to_owned())]);
    let eu = start_stateful_backend(vec![(200, "site-eu".to_owned())]);
    let uk = start_stateful_backend(vec![(200, "site-uk".to_owned())]);
    let proxy_port = free_port();

    let config = load_example_config(
        CLAIM_GATING_EXAMPLE,
        proxy_port,
        HashMap::from([
            (SITE_US_A, us_a.port()),
            (SITE_US_B, us_b.port()),
            (SITE_EU, eu.port()),
            (SITE_UK, uk.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"Qwen3-Coder-30B-A3B","messages":[]}"#;
    let raw = http_send(proxy.addr(), &chat_post(body));

    assert_eq!(
        parse_status(&raw),
        403,
        "a residency gate with no authenticated identity must deny: {raw}"
    );
    assert!(
        [&us_a, &us_b, &eu, &uk].iter().all(|b| b.requests().is_empty()),
        "no upstream may be contacted when the fence denies: {raw}"
    );
}
