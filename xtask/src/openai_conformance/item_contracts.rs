// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Source-derived Conversation item schema artifact.

use std::{collections::BTreeMap, path::PathBuf};

use clap::Parser;
use praxis_test_utils::openapi::{Value as YamlValue, component_schema_closure, parse_document};
use serde::Serialize;

use super::area::OPENAI_REFERENCE_SPEC;

/// Checked-in schema artifact consumed by the Conversations runtime.
pub(super) const ITEM_CONTRACTS_ARTIFACT: &str = "apis/src/openai/conversations/item_contracts.json";

/// Components that define the two owned Conversation item boundaries.
const ROOT_SCHEMAS: [&str; 2] = ["ConversationItem", "InputItem"];

/// CLI arguments for regenerating the Conversation item contract artifact.
#[derive(Parser)]
pub(crate) struct Args {
    /// Verify the artifact without changing it.
    #[arg(long)]
    check: bool,

    /// Pinned OpenAI document to derive from.
    #[arg(long, default_value = OPENAI_REFERENCE_SPEC)]
    openai_spec: PathBuf,

    /// Artifact path to write or verify.
    #[arg(long, default_value = ITEM_CONTRACTS_ARTIFACT)]
    output: PathBuf,
}

/// Regenerate or verify the source-derived item schema artifact.
pub(crate) fn run(args: &Args) {
    let source = std::fs::read(&args.openai_spec).unwrap_or_else(|error| {
        eprintln!("failed to read {}: {error}", args.openai_spec.display());
        std::process::exit(1);
    });
    let generated = generate_item_contracts(&source).unwrap_or_else(|error| {
        eprintln!("failed to generate Conversation item contracts: {error}");
        std::process::exit(1);
    });
    if args.check {
        verify_item_contracts(&args.output, &generated).unwrap_or_else(|error| {
            eprintln!("Conversation item contract check failed: {error}");
            std::process::exit(1);
        });
        println!("Conversation item contracts are current: {}", args.output.display());
    } else {
        std::fs::write(&args.output, generated).unwrap_or_else(|error| {
            eprintln!("failed to write {}: {error}", args.output.display());
            std::process::exit(1);
        });
        println!("wrote Conversation item contracts: {}", args.output.display());
    }
}

/// Derive the recursive item-schema closure from the pinned OpenAI document.
pub(super) fn generate_item_contracts(source: &[u8]) -> Result<String, String> {
    let source = std::str::from_utf8(source).map_err(|error| format!("pinned OpenAI schema is not UTF-8: {error}"))?;
    let document = parse_document(source)?;
    let selected = component_schema_closure(&document, &ROOT_SCHEMAS)?;

    let artifact = ItemContractArtifact {
        schema_version: 1,
        schemas: selected,
    };
    serde_json::to_string_pretty(&artifact)
        .map(|content| format!("{content}\n"))
        .map_err(|error| format!("failed to serialize Conversation item schemas: {error}"))
}

/// Serializable generated-artifact shape.
#[derive(Serialize)]
struct ItemContractArtifact<'a> {
    /// Artifact format version.
    schema_version: u8,

    /// Recursive item component closure.
    schemas: BTreeMap<String, &'a YamlValue>,
}

/// Fail when the checked-in artifact differs from generated content.
pub(super) fn verify_item_contracts(path: &std::path::Path, generated: &str) -> Result<(), String> {
    let actual =
        std::fs::read_to_string(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if actual != generated {
        return Err(format!(
            "{} is stale; run cargo xtask openai-conversation-item-contracts",
            path.display()
        ));
    }
    Ok(())
}
