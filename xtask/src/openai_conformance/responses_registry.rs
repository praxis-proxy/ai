// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Drift check between the runtime Responses registry and the pinned spec.
//!
//! The registry is the runtime source of truth for Responses operation
//! identity. This check fails when a registered operation's method, path, or
//! operation ID no longer agrees with the pinned OpenAI specification, and when
//! an operation declared as a Praxis protocol extension turns out to exist in
//! the specification after all.

use super::{
    area::{OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC},
    model::OperationScope,
    registry_check::{ok_summary, tally},
    spec::{load_reference_source, project_reference},
};

/// Responses operations selected from the pinned specification.
const RESPONSES_SCOPE: OperationScope = OperationScope::new("responses", "Responses", &["/responses"]);

/// Compare the runtime Responses registry against the pinned specification.
pub(super) fn check() -> Result<String, String> {
    let reference = load_reference_source(OPENAI_REFERENCE_SPEC, Some(OPENAI_REFERENCE_MANIFEST))?;
    let operations = project_reference(&reference, RESPONSES_SCOPE)?.operations;
    let tally = tally(
        praxis_ai_apis::openai::responses_operation_specs(),
        praxis_ai_apis::openai::RESPONSES_PROTOCOL_EXTENSION_OPERATION_IDS,
        &operations,
    );

    if tally.failures.is_empty() {
        Ok(ok_summary("responses", &tally))
    } else {
        Err(tally.failures.join("\n"))
    }
}
