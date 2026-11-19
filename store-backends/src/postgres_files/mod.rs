// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `PostgreSQL` backend for the file catalog.
//!
//! Only the schema exists so far; the catalog operations arrive in later steps.

// Remove once the catalog constructor (the next step) calls the schema entry point.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "no caller until the catalog constructor lands")
)]

pub(crate) mod schema;
