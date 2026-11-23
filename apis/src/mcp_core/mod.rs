// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-neutral MCP tool resolution and execution core.
//!
//! [`crate::mcp_client`] speaks the MCP wire protocol (`tools/list`,
//! `tools/call`, sessions, bounded transport). This module holds the layer
//! above it that every inference-protocol adapter shares: operator connector
//! configuration, tool-metadata admission, model-facing name encoding,
//! approval classification, target fingerprints, argument validation, and
//! result normalization. Protocol adapters such as the OpenAI Responses MCP
//! filters own only their request rewrite, response shapes, and loop wiring.

pub(crate) mod admission;
pub(crate) mod classify;
pub(crate) mod connector;
pub(crate) mod execute;
pub(crate) mod fingerprint;
pub(crate) mod naming;
pub(crate) mod tool_map;
pub(crate) mod validate;
