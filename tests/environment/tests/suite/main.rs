// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `llm-d` environment integration test suite.
//!
//! These tests require Docker or Podman for the
//! `llm-d-inference-sim` container backend and the
//! `llmd-ext-proc` feature.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::tests_outside_test_module,
    clippy::too_many_lines,
    missing_docs,
    reason = "integration test binary"
)]

mod sim_backend;
