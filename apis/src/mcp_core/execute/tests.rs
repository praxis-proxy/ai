// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! MCP result normalization regression coverage.

use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};

use super::{MCP_RESULT_TOO_LARGE, call_result_to_output};

#[test]
fn structured_result_preserves_both_content_representations() -> Result<(), Box<dyn std::error::Error>> {
    for content in [
        vec![],
        vec![ContentBlock::text("Weather fetched")],
        vec![
            ContentBlock::text("caption"),
            ContentBlock::image("aW1hZ2U=", "image/png"),
        ],
    ] {
        for structured in [json!({"temperature": 22.5}), json!({}), json!({"text": "שלום\n\"\\"})] {
            let mut result = CallToolResult::success(content.clone());
            result.structured_content = Some(structured.clone());
            let output = call_result_to_output(&result, usize::MAX)?;
            let recovered: Value = serde_json::from_str(&output)?;
            assert_eq!(recovered, json!({"content": content, "structuredContent": structured}));
        }
    }
    Ok(())
}

#[test]
fn structured_result_limit_counts_the_combined_escaped_json_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let mut result = CallToolResult::success(vec![ContentBlock::text("caption\n\"\\")]);
    result.structured_content = Some(json!({"text": "שלום\n\"\\"}));
    let output = call_result_to_output(&result, usize::MAX)?;
    let limit = output.len();
    assert_eq!(call_result_to_output(&result, limit)?, output);
    for cap in [0, limit.saturating_sub(1)] {
        assert!(
            matches!(call_result_to_output(&result, cap), Err(error) if error == MCP_RESULT_TOO_LARGE),
            "the complete UTF-8 JSON envelope and escaping count toward the cap"
        );
    }
    Ok(())
}

#[test]
fn oversized_structured_result_is_rejected() {
    let mut result = CallToolResult::success(vec![]);
    result.structured_content = Some(json!({"payload": "x".repeat(4096)}));
    assert!(
        matches!(call_result_to_output(&result, 1024), Err(error) if error == MCP_RESULT_TOO_LARGE),
        "oversized structured content must produce the configured limit error"
    );
}

#[test]
fn results_without_structured_content_keep_their_existing_output() -> Result<(), Box<dyn std::error::Error>> {
    for (content, expected) in [
        (vec![], String::new()),
        (
            vec![ContentBlock::text("hello"), ContentBlock::text("world")],
            "hello\nworld".to_owned(),
        ),
        (
            vec![ContentBlock::image("aW1hZ2U=", "image/png")],
            json!([{"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"}]).to_string(),
        ),
    ] {
        assert_eq!(
            call_result_to_output(&CallToolResult::success(content), usize::MAX)?,
            expected
        );
    }
    Ok(())
}
