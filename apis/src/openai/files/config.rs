// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_files` filter.

use serde::Deserialize;

/// YAML configuration for the [`OpenaiFilesFilter`].
///
/// No settings yet: catalog and store provisioning land with the serving
/// runtime wiring. Unknown fields are rejected so a misspelled setting fails
/// at pipeline construction.
///
/// [`OpenaiFilesFilter`]: super::OpenaiFilesFilter
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::empty_structs_with_brackets,
    reason = "temporarily empty until catalog provisioning adds settings (#504); braces keep the YAML mapping form"
)]
pub(crate) struct FilesConfig {}
