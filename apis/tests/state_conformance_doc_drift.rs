// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Drift guard for `docs/conformance/openai-responses-state-conformance.md`.
//!
//! The ownership of each Responses state flag (Praxis-stateful / inspected /
//! rejected / backend-passthrough) is a semantic judgment that cannot be
//! statically auto-derived, so that conformance doc is hand-written. This test
//! guards the concrete behavioral facts the doc cites: if one of these anchors
//! is renamed or removed, the test fails and points back at the doc so the
//! table gets re-checked in the same change.
//!
//! It only reads source text (no crate features, no pipeline), so it runs under
//! a plain `cargo test -p praxis-ai-apis`.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, reason = "integration test")]
mod tests {
    use std::{fs, path::PathBuf};

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    const DOC: &str = "../docs/conformance/openai-responses-state-conformance.md";

    /// `(source file relative to the apis crate, substring that must be present, why it matters)`
    const ANCHORS: &[(&str, &str, &str)] = &[
        (
            "src/openai/responses/mod.rs",
            "background mode is not supported",
            "background=true is rejected with 400 (doc: background = rejected)",
        ),
        (
            "src/openai/responses/mod.rs",
            "prompt templates are supported only for OpenAI-owned upstreams",
            "non-null prompt is rejected with 400 (doc: prompt = rejected)",
        ),
        (
            "src/openai/responses/request/mod.rs",
            "mutually_exclusive_parameters",
            "previous_response_id + conversation together is rejected (doc: mutual exclusion rule)",
        ),
        (
            "src/openai/responses/rehydrate/mod.rs",
            "cannot continue from response with status",
            "continuation from a non-completed stored response is rejected (doc: completion gate)",
        ),
        (
            "src/openai/responses/request/mod.rs",
            "classified.store.unwrap_or(true)",
            "store defaults to true (doc: store default)",
        ),
        (
            "src/classifier/mod.rs",
            "has_previous_response_id",
            "classifier extracts the previous_response_id selector (doc: state-flag extraction)",
        ),
        (
            "src/classifier/mod.rs",
            "has_conversation",
            "classifier extracts the conversation selector (doc: state-flag extraction)",
        ),
    ];

    /// Every flag that owns a row in the state-flag table. The guard checks for
    /// the row marker itself (`` | `flag` ``) so deleting a whole row fails even
    /// if the flag name still appears in prose elsewhere.
    const TABLE_FLAGS: &[&str] = &[
        "previous_response_id",
        "conversation",
        "store",
        "background",
        "prompt",
        "stream",
        "include",
        "context_management",
        "max_tool_calls",
        "parallel_tool_calls",
        "tools",
    ];

    #[test]
    fn conformance_doc_anchors_present() {
        let missing: Vec<String> = ANCHORS
            .iter()
            .filter(|(file, needle, _)| !repo_file(file).contains(needle))
            .map(|(file, needle, why)| format!("  - {file}: missing {needle:?} — {why}"))
            .collect();

        assert!(
            missing.is_empty(),
            "Behavioral anchors cited by {DOC} are gone — re-check the doc's claims:\n{}",
            missing.join("\n"),
        );
    }

    #[test]
    fn conformance_doc_keeps_every_table_row() {
        let doc = repo_file(DOC);
        let missing: Vec<String> = TABLE_FLAGS
            .iter()
            .map(|flag| format!("| `{flag}`"))
            .filter(|row| !doc.contains(row))
            .collect();

        assert!(missing.is_empty(), "{DOC} dropped state-flag table rows: {missing:?}",);
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn repo_file(rel: &str) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
        fs::read_to_string(&path).expect("conformance-guarded file must be readable")
    }
}
