// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Drift check between the runtime Vector Stores registry and the pinned spec.
//!
//! The registry is the runtime source of truth for Vector Stores operation identity.
//! This check fails when a registered operation's method, path, or operation ID
//! no longer agrees with the pinned OpenAI specification.

use super::{
    area::{OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC},
    model::OperationScope,
    registry_check::{ok_summary, tally},
    spec::load_reference_source,
};

/// Vector Stores operations selected from the pinned specification.
const VECTOR_STORES_SCOPE: OperationScope = OperationScope::new("vector_stores", "Vector Stores", &["/vector_stores"]);

/// Compare the runtime Vector Stores registry against the pinned specification.
pub(super) fn check() -> Result<String, String> {
    let reference = load_reference_source(OPENAI_REFERENCE_SPEC, Some(OPENAI_REFERENCE_MANIFEST))?;
    let operations = reference.document.scoped_operations(VECTOR_STORES_SCOPE)?;
    let tally = tally(
        praxis_ai_apis::openai::vector_stores_operation_specs(),
        &[],
        &operations,
    );

    if tally.failures.is_empty() {
        Ok(ok_summary("vector stores", &tally))
    } else {
        Err(tally.failures.join("\n"))
    }
}
