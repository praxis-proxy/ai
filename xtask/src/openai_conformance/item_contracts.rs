// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Source-derived Conversation item schema artifact.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use clap::Parser;
use serde::Serialize;

use super::{
    area::OPENAI_REFERENCE_SPEC,
    semantic_yaml::{Mapping as YamlMapping, Value as YamlValue},
};

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
    let document: YamlValue =
        serde_yaml::from_str(source).map_err(|error| format!("failed to parse pinned OpenAI schema: {error}"))?;
    let schemas = document
        .as_mapping()
        .and_then(|root| yaml_get(root, "components"))
        .and_then(YamlValue::as_mapping)
        .and_then(|components| yaml_get(components, "schemas"))
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| "pinned OpenAI schema has no components.schemas object".to_owned())?;

    let mut pending = ROOT_SCHEMAS.into_iter().map(str::to_owned).collect::<BTreeSet<_>>();
    let mut selected = BTreeMap::new();
    while let Some(name) = pending.pop_first() {
        if selected.contains_key(&name) {
            continue;
        }
        let schema = schemas
            .get(&YamlValue::String(name.clone()))
            .ok_or_else(|| format!("item schema references missing component {name}"))?;
        collect_schema_references(schema, &mut pending)?;
        selected.insert(name, schema);
    }

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

/// Add local schema component references found below one JSON value.
fn collect_schema_references(value: &YamlValue, references: &mut BTreeSet<String>) -> Result<(), String> {
    match value {
        YamlValue::Sequence(values) => {
            for value in values {
                collect_schema_references(value, references)?;
            }
        },
        YamlValue::Mapping(fields) => {
            for (name, value) in fields.iter() {
                if name.as_str() == Some("$ref") {
                    let reference = value
                        .as_str()
                        .ok_or_else(|| "item schema contains a non-string $ref".to_owned())?;
                    let component = reference
                        .strip_prefix("#/components/schemas/")
                        .ok_or_else(|| format!("item schema contains unsupported reference {reference}"))?;
                    references.insert(component.to_owned());
                } else {
                    collect_schema_references(value, references)?;
                }
            }
        },
        YamlValue::Bool(_)
        | YamlValue::Float(_)
        | YamlValue::Null
        | YamlValue::Signed(_)
        | YamlValue::String(_)
        | YamlValue::Unsigned(_) => {},
    }
    Ok(())
}

/// Borrow one string-keyed YAML mapping value.
fn yaml_get<'a>(mapping: &'a YamlMapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(&YamlValue::String(key.to_owned()))
}
