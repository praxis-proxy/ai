// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Drift check between the runtime Files registry and the pinned spec.
//!
//! The registry is the runtime source of truth for Files operation identity.
//! This check fails when a registered operation's method, path, or operation ID
//! no longer agrees with the pinned OpenAI specification.

use super::{
    area::{OPENAI_REFERENCE_MANIFEST, OPENAI_REFERENCE_SPEC},
    model::OperationScope,
    registry_check::{ok_summary, tally},
    spec::load_reference_source,
};

/// Files operations selected from the pinned specification.
const FILES_SCOPE: OperationScope = OperationScope::new("files", "Files", &["/files"]);

/// Compare the runtime Files registry against the pinned specification.
pub(super) fn check() -> Result<String, String> {
    let reference = load_reference_source(OPENAI_REFERENCE_SPEC, Some(OPENAI_REFERENCE_MANIFEST))?;
    let operations = reference.document.scoped_operations(FILES_SCOPE)?;
    let tally = tally(praxis_ai_apis::openai::files_operation_specs(), &[], &operations);

    if tally.failures.is_empty() {
        Ok(ok_summary("files", &tally))
    } else {
        Err(tally.failures.join("\n"))
    }
}
