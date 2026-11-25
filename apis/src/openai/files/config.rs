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
pub(crate) struct FilesConfig {}
