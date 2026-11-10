// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI Files API family.

#[cfg(feature = "openai-files")]
pub(crate) mod contracts;
#[cfg(feature = "openai-files")]
mod openapi;
pub(crate) mod routes;

#[cfg(feature = "openai-files")]
pub use openapi::implementation_openapi_json;
