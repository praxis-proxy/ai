//! Recognition and constraint of client `custom` tool input *formats* for the
//! function-only Responses backend.
//!
//! A Codex-style client declares a freeform `custom` tool whose `format` is a
//! Lark grammar. The compatibility filter lowers every `custom` tool to a private
//! `function` carrying a `{"input":"<raw text>"}` envelope; this module decides,
//! per declaration, whether that input is unconstrained text or one of the pinned
//! `apply_patch` patch grammars, and (for the latter) supplies:
//!
//! - the backend JSON-Schema `pattern` that constrains generation of the lowered function's `input` argument, and
//! - the host-side validator that checks the *decoded* patch before a completed `custom_tool_call` is exposed to the
//!   client.
//!
//! Recognition is by exact, byte-for-byte match of the declared grammar against
//! the two pinned Codex 0.144.1 definitions. The tool *name* never enables
//! support; any other grammar is rejected before an upstream request.
//!
//! ## Two regexes, two jobs
//!
//! `XGrammar` (`vLLM`'s constrained-decoding backend) applies a JSON-Schema string
//! `pattern` to the **raw JSON-string body**, so the backend pattern must match
//! the *escaped* form of the patch (a newline is the two characters `\` `n`). A
//! spec-compliant backend that matches the **decoded** value instead needs the
//! raw regex. The encoding is therefore configurable ([`CustomInputPatternEncoding`]).
//! Regardless of what the backend does, the **host** validator ([`validate_decoded_input`])
//! runs against the decoded patch and is the authoritative fail-closed guarantee.
//!
//! ## Provenance
//!
//! `apply_patch.lark` is vendored verbatim from `openai/codex@rust-v0.144.1`
//! (`codex-rs/core/src/tools/handlers/apply_patch.lark`, Apache-2.0),
//! sha256 `d6367f4826ed608c424b0a308f3d6163527df63c22513d089b91863552f8bfeb`.
//! The Environment-ID variant is derived with the exact substitution Codex
//! performs in `apply_patch_spec.rs`, so it is byte-identical to what a client
//! that sets `include_environment_id` sends.

use std::sync::LazyLock;

use praxis_filter::FilterAction;
use regex::Regex;
use serde_json::Value;

use super::reject_bad_request;

/// The pinned base `apply_patch` Lark grammar, verbatim from Codex 0.144.1.
pub(crate) const APPLY_PATCH_LARK_BASE: &str = include_str!("apply_patch.lark");

/// The `start` rule Codex rewrites when `include_environment_id` is set.
const BASE_START_RULE: &str = "start: begin_patch hunk+ end_patch";
/// Its replacement, verbatim from `apply_patch_spec.rs`.
const ENVID_START_RULE: &str =
    "start: begin_patch environment_id? hunk+ end_patch\nenvironment_id: \"*** Environment ID: \" filename LF";

/// The pinned Environment-ID `apply_patch` grammar: the base grammar with the
/// exact single-rule substitution Codex applies. Derived (not transcribed) so it
/// cannot drift from the base.
pub(crate) static APPLY_PATCH_LARK_ENVID: LazyLock<String> =
    LazyLock::new(|| APPLY_PATCH_LARK_BASE.replace(BASE_START_RULE, ENVID_START_RULE));

/// The recognized input format a lowered `custom` tool carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CustomInputFormat {
    /// Unconstrained text (`format` absent or `{"type":"text"}`).
    Text,
    /// The pinned base `apply_patch` grammar.
    ApplyPatch,
    /// The pinned Environment-ID `apply_patch` grammar.
    ApplyPatchEnvId,
}

/// How the backend interprets a JSON-Schema string `pattern`, which decides the
/// encoding of the pattern we emit on the lowered function's `input`.
///
/// Scopes the `XGrammar`-specific escaping workaround: the default `JsonString`
/// targets `XGrammar`/`vLLM` (which matches the raw JSON-string body), while a
/// spec-compliant backend matching the decoded value uses `Decoded`, and `None`
/// disables the backend `pattern` entirely (host validation still enforces).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CustomInputPatternEncoding {
    /// Backend matches the raw JSON-string body (`XGrammar`/`vLLM`): emit the
    /// escaping-aware pattern. The default target.
    #[default]
    JsonString,
    /// Backend matches the decoded string value (spec-compliant JSON Schema):
    /// emit the raw regex.
    Decoded,
    /// Emit no `pattern`; rely solely on host validation.
    None,
}

// -----------------------------------------------------------------------------
// Backend patterns (JSON-escaping-aware — enforced by `XGrammar` during generation)
// -----------------------------------------------------------------------------

/// Escaping-aware `pattern` for the base grammar: matches the patch as it appears
/// inside a JSON string (newline → `\n`, i.e. backslash-n; non-newline char → a
/// literal or a JSON escape). Verified under `XGrammar` 0.2.8 (see the feasibility
/// gate). `\xNN`/`\u` are regex escapes, not literal control bytes.
///
/// Best-effort by design: the `\\u[0-9a-fA-F]{4}` alternative admits `\u000A` (an
/// encoded newline), which the grammar's `[^\n]` runs exclude. A model would have to
/// encode a newline *inside* a line to hit it; if it did, [`validate_decoded_input`]
/// (the authoritative, backend-independent host check against the real-newline host
/// regex) rejects the decoded patch and the call fails closed. The backend pattern
/// only guides generation; it never replaces host validation.
pub(crate) const APPLY_PATCH_BACKEND_PATTERN_BASE: &str = r#"\*\*\* Begin Patch\\n(?:(?:\*\*\* Add File: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n(?:\+(?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})*\\n)+|\*\*\* Delete File: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n|\*\*\* Update File: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n(?:\*\*\* Move to: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n)?(?:(?:@@(?: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+)?\\n|[+\- ](?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})*\\n)+(?:\*\*\* End of File\\n)?)?))+\*\*\* End Patch(?:\\n)?"#;

/// Escaping-aware `pattern` for the Environment-ID grammar.
pub(crate) const APPLY_PATCH_BACKEND_PATTERN_ENVID: &str = r#"\*\*\* Begin Patch\\n(?:\*\*\* Environment ID: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n)?(?:(?:\*\*\* Add File: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n(?:\+(?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})*\\n)+|\*\*\* Delete File: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n|\*\*\* Update File: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n(?:\*\*\* Move to: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+\\n)?(?:(?:@@(?: (?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})+)?\\n|[+\- ](?:[^"\\\x00-\x1F]|\\["\\/bfrt]|\\u[0-9a-fA-F]{4})*\\n)+(?:\*\*\* End of File\\n)?)?))+\*\*\* End Patch(?:\\n)?"#;

// -----------------------------------------------------------------------------
// Host patterns (raw — matched against the DECODED patch, real newlines)
// -----------------------------------------------------------------------------

/// Raw (unanchored) host pattern for the base grammar: the body the anchored host
/// validator and the anchored `Decoded` backend pattern are both built from. Not used
/// as a backend pattern directly — JSON-Schema `pattern` is a substring match, so the
/// `Decoded` encoding emits the anchored form ([`APPLY_PATCH_DECODED_PATTERN_BASE`]).
pub(crate) const APPLY_PATCH_HOST_PATTERN_BASE: &str = r"\*\*\* Begin Patch\n(?:(?:\*\*\* Add File: [^\n]+\n(?:\+[^\n]*\n)+|\*\*\* Delete File: [^\n]+\n|\*\*\* Update File: [^\n]+\n(?:\*\*\* Move to: [^\n]+\n)?(?:(?:@@(?: [^\n]+)?\n|[+\- ][^\n]*\n)+(?:\*\*\* End of File\n)?)?))+\*\*\* End Patch\n?";

/// Raw (unanchored) host pattern for the Environment-ID grammar.
pub(crate) const APPLY_PATCH_HOST_PATTERN_ENVID: &str = r"\*\*\* Begin Patch\n(?:\*\*\* Environment ID: [^\n]+\n)?(?:(?:\*\*\* Add File: [^\n]+\n(?:\+[^\n]*\n)+|\*\*\* Delete File: [^\n]+\n|\*\*\* Update File: [^\n]+\n(?:\*\*\* Move to: [^\n]+\n)?(?:(?:@@(?: [^\n]+)?\n|[+\- ][^\n]*\n)+(?:\*\*\* End of File\n)?)?))+\*\*\* End Patch\n?";

/// Anchored full-string host validator for the base grammar. A compile failure
/// (impossible for the pinned pattern) is surfaced fail-closed at the use site.
static HOST_RE_BASE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(&format!(r"\A(?:{APPLY_PATCH_HOST_PATTERN_BASE})\z")));

/// Anchored full-string host validator for the Environment-ID grammar.
static HOST_RE_ENVID: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(&format!(r"\A(?:{APPLY_PATCH_HOST_PATTERN_ENVID})\z")));

/// Full-string-anchored `Decoded` backend pattern for the base grammar. JSON-Schema
/// `pattern` is a substring match, so a `custom_input_pattern_encoding: decoded`
/// backend must be given an anchored form. The end anchor is `(?![\s\S])` (no
/// character follows) rather than `$`: it is a strict end-of-input in every regex
/// dialect, whereas `$` admits a trailing line terminator under Python/PCRE
/// semantics, which — combined with the grammar's optional final newline — would let
/// an extra trailing newline satisfy the constraint.
static APPLY_PATCH_DECODED_PATTERN_BASE: LazyLock<String> =
    LazyLock::new(|| format!(r"^(?:{APPLY_PATCH_HOST_PATTERN_BASE})(?![\s\S])"));

/// Full-string-anchored `Decoded` backend pattern for the Environment-ID grammar.
static APPLY_PATCH_DECODED_PATTERN_ENVID: LazyLock<String> =
    LazyLock::new(|| format!(r"^(?:{APPLY_PATCH_HOST_PATTERN_ENVID})(?![\s\S])"));

// -----------------------------------------------------------------------------
// API
// -----------------------------------------------------------------------------

/// Resolve a `custom` tool's declared `format` to a recognized [`CustomInputFormat`],
/// or reject it. `descriptor` names the offending tool in the reject message.
///
/// Accepts an absent format or `{"type":"text"}` as [`CustomInputFormat::Text`],
/// and a `{"type":"grammar","syntax":"lark","definition":D}` whose `D` is exactly
/// one of the two pinned `apply_patch` grammars. Any other format — including a
/// differently-named tool carrying an unknown grammar, a non-`lark` syntax, or an
/// unrecognized definition — is rejected before an upstream request.
pub(crate) fn resolve_custom_input_format(tool: &Value, descriptor: &str) -> Result<CustomInputFormat, FilterAction> {
    match tool.get("format") {
        None => Ok(CustomInputFormat::Text),
        Some(format) => match format.get("type").and_then(Value::as_str) {
            Some("text") => Ok(CustomInputFormat::Text),
            Some("grammar") => resolve_grammar(format, descriptor),
            _ => Err(unsupported_format(descriptor)),
        },
    }
}

/// Resolve a `{"type":"grammar", …}` format to a pinned `apply_patch` variant, or
/// reject. The `syntax` must be `lark` and the `definition` must match a pinned
/// grammar byte-for-byte.
fn resolve_grammar(format: &Value, descriptor: &str) -> Result<CustomInputFormat, FilterAction> {
    if format.get("syntax").and_then(Value::as_str) == Some("lark") {
        match format.get("definition").and_then(Value::as_str) {
            Some(def) if def == APPLY_PATCH_LARK_BASE => return Ok(CustomInputFormat::ApplyPatch),
            Some(def) if def == APPLY_PATCH_LARK_ENVID.as_str() => return Ok(CustomInputFormat::ApplyPatchEnvId),
            _ => {},
        }
    }
    Err(unsupported_format(descriptor))
}

/// The HTTP 400 for a custom format this filter cannot preserve.
fn unsupported_format(descriptor: &str) -> FilterAction {
    reject_bad_request(&format!(
        "{descriptor} uses an unsupported custom format; only text or the pinned apply_patch grammar can be lowered for \
         a function-only backend"
    ))
}

/// The backend `pattern` to place on the lowered function's `input` for `format`
/// under `encoding`, or `None` when no constraint is emitted (text format, or the
/// `None` encoding that relies solely on host validation).
pub(crate) fn backend_input_pattern(
    format: CustomInputFormat,
    encoding: CustomInputPatternEncoding,
) -> Option<&'static str> {
    match (format, encoding) {
        (CustomInputFormat::Text, _) | (_, CustomInputPatternEncoding::None) => None,
        (CustomInputFormat::ApplyPatch, CustomInputPatternEncoding::JsonString) => {
            Some(APPLY_PATCH_BACKEND_PATTERN_BASE)
        },
        (CustomInputFormat::ApplyPatchEnvId, CustomInputPatternEncoding::JsonString) => {
            Some(APPLY_PATCH_BACKEND_PATTERN_ENVID)
        },
        (CustomInputFormat::ApplyPatch, CustomInputPatternEncoding::Decoded) => {
            Some(APPLY_PATCH_DECODED_PATTERN_BASE.as_str())
        },
        (CustomInputFormat::ApplyPatchEnvId, CustomInputPatternEncoding::Decoded) => {
            Some(APPLY_PATCH_DECODED_PATTERN_ENVID.as_str())
        },
    }
}

/// Whether a decoded `input` satisfies `format`'s grammar. Always `true` for
/// [`CustomInputFormat::Text`]; for an `apply_patch` variant, a full-string match
/// of the host regex against the decoded patch.
pub(crate) fn validate_decoded_input(format: CustomInputFormat, input: &str) -> bool {
    match format {
        CustomInputFormat::Text => true,
        CustomInputFormat::ApplyPatch => HOST_RE_BASE.as_ref().is_ok_and(|re| re.is_match(input)),
        CustomInputFormat::ApplyPatchEnvId => HOST_RE_ENVID.as_ref().is_ok_and(|re| re.is_match(input)),
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    fn status_of(err: &FilterAction) -> u16 {
        let FilterAction::Reject(rejection) = err else {
            panic!("expected FilterAction::Reject");
        };
        rejection.status
    }

    #[test]
    fn grammar_base_is_pinned_bytes() {
        // Pins the vendored file; sha256 d6367f4826ed...bfeb at vendor time.
        assert_eq!(APPLY_PATCH_LARK_BASE.len(), 578, "vendored grammar byte length changed");
        assert!(APPLY_PATCH_LARK_BASE.starts_with("start: begin_patch hunk+ end_patch\n"));
        assert!(APPLY_PATCH_LARK_BASE.ends_with("%import common.LF\n"));
        assert!(!APPLY_PATCH_LARK_BASE.contains("environment_id"));
    }

    #[test]
    fn envid_is_base_plus_substitution() {
        let envid = &*APPLY_PATCH_LARK_ENVID;
        assert_ne!(envid, APPLY_PATCH_LARK_BASE, "env-id variant must differ");
        assert!(envid.starts_with("start: begin_patch environment_id? hunk+ end_patch\nenvironment_id:"));
        assert!(envid.contains("environment_id: \"*** Environment ID: \" filename LF"));
        // Only the start rule changed: the remainder is identical.
        assert!(envid.ends_with("%import common.LF\n"));
    }

    #[test]
    fn resolve_text_formats() {
        assert_eq!(
            resolve_custom_input_format(&json!({"type": "custom", "name": "x"}), "custom tool 'x'").unwrap(),
            CustomInputFormat::Text,
            "absent format is text",
        );
        assert_eq!(
            resolve_custom_input_format(
                &json!({"type": "custom", "name": "x", "format": {"type": "text"}}),
                "custom tool 'x'",
            )
            .unwrap(),
            CustomInputFormat::Text,
        );
    }

    #[test]
    fn resolve_pinned_grammars() {
        let base = json!({
            "type": "custom", "name": "apply_patch",
            "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_LARK_BASE},
        });
        assert_eq!(
            resolve_custom_input_format(&base, "custom tool 'apply_patch'").unwrap(),
            CustomInputFormat::ApplyPatch,
        );
        let envid = json!({
            "type": "custom", "name": "apply_patch",
            "format": {"type": "grammar", "syntax": "lark", "definition": &*APPLY_PATCH_LARK_ENVID},
        });
        assert_eq!(
            resolve_custom_input_format(&envid, "custom tool 'apply_patch'").unwrap(),
            CustomInputFormat::ApplyPatchEnvId,
        );
    }

    #[test]
    fn rejects_unknown_grammar() {
        let unknown = json!({
            "type": "custom", "name": "apply_patch",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"},
        });
        let err = resolve_custom_input_format(&unknown, "custom tool 'apply_patch'").expect_err("unknown grammar");
        assert_eq!(status_of(&err), 400);

        // A non-lark syntax with the pinned definition is still unrecognized.
        let wrong_syntax = json!({
            "type": "custom", "name": "apply_patch",
            "format": {"type": "grammar", "syntax": "regex", "definition": APPLY_PATCH_LARK_BASE},
        });
        assert_eq!(
            status_of(&resolve_custom_input_format(&wrong_syntax, "custom tool 'apply_patch'").expect_err("non-lark")),
            400,
        );
    }

    #[test]
    fn rejects_name_only() {
        // The name `apply_patch` with no grammar is plain text — the name never
        // enables grammar support.
        assert_eq!(
            resolve_custom_input_format(
                &json!({"type": "custom", "name": "apply_patch"}),
                "custom tool 'apply_patch'"
            )
            .unwrap(),
            CustomInputFormat::Text,
        );
    }

    #[test]
    fn host_regex_accepts_valid_patches() {
        let f = CustomInputFormat::ApplyPatch;
        let valid = [
            "*** Begin Patch\n*** Add File: a.txt\n+hello\n+world\n*** End Patch\n",
            "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch", // no trailing LF on end
            "*** Begin Patch\n*** Delete File: old.py\n*** End Patch\n",
            "*** Begin Patch\n*** Update File: x.rs\n@@ fn main\n-old\n+new\n context\n*** End Patch\n",
            "*** Begin Patch\n*** Update File: x.rs\n@@\n-old\n+new\n*** End Patch\n", // bare @@
            "*** Begin Patch\n*** Update File: a\n*** Move to: b\n@@\n-x\n+y\n*** End Patch\n",
            "*** Begin Patch\n*** Update File: a\n context\n*** End of File\n*** End Patch\n",
            "*** Begin Patch\n*** Update File: a\n*** End Patch\n", // update header only
            "*** Begin Patch\n*** Add File: a\n+1\n*** Delete File: b\n*** End Patch\n", // multi-hunk
            "*** Begin Patch\n*** Add File: café/日本.txt\n+naïve ☃\n*** End Patch\n", // unicode
            "*** Begin Patch\n*** Add File: a\n+say \"hi\" \\ bye\n*** End Patch\n", // quotes+backslash
        ];
        for patch in valid {
            assert!(validate_decoded_input(f, patch), "should accept: {patch:?}");
        }
    }

    #[test]
    fn host_regex_rejects_malformed_patches() {
        let f = CustomInputFormat::ApplyPatch;
        let invalid = [
            "*** Begin Patch\n*** End Patch\n",                               // no hunks
            "*** Add File: a\n+1\n*** End Patch\n",                           // missing begin
            "*** Begin Patch\n*** Add File: a\n*** End Patch\n",              // zero add lines
            "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch\nGARBAGE\n", // trailing garbage
            "*** Begin Patch\n*** Frobnicate File: a\n+1\n*** End Patch\n",   // bogus header
            "*** Begin Patch\n*** Update File: a\nXnope\n*** End Patch\n",    // bad change prefix
            "",                                                               // empty
            "*** Begin Patch\n*** Add File: a\n+1\n",                         // truncated (no end)
            // env-id line is not allowed under the base grammar
            "*** Begin Patch\n*** Environment ID: e\n*** Add File: a\n+1\n*** End Patch\n",
        ];
        for patch in invalid {
            assert!(!validate_decoded_input(f, patch), "should reject: {patch:?}");
        }
    }

    #[test]
    fn host_regex_envid_variant() {
        let f = CustomInputFormat::ApplyPatchEnvId;
        assert!(validate_decoded_input(
            f,
            "*** Begin Patch\n*** Environment ID: env-1\n*** Add File: a\n+1\n*** End Patch\n",
        ));
        // Env-id is optional under its grammar.
        assert!(validate_decoded_input(
            f,
            "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch\n"
        ));
    }

    #[test]
    fn text_input_is_always_valid() {
        assert!(validate_decoded_input(CustomInputFormat::Text, "anything at all"));
        assert!(validate_decoded_input(CustomInputFormat::Text, ""));
    }

    #[test]
    fn backend_pattern_selection() {
        use CustomInputFormat::{ApplyPatch, ApplyPatchEnvId, Text};
        use CustomInputPatternEncoding::{Decoded, JsonString, None as NoEnc};
        assert_eq!(backend_input_pattern(Text, JsonString), None);
        assert_eq!(backend_input_pattern(ApplyPatch, NoEnc), None);
        assert_eq!(
            backend_input_pattern(ApplyPatch, JsonString),
            Some(APPLY_PATCH_BACKEND_PATTERN_BASE)
        );
        assert_eq!(
            backend_input_pattern(ApplyPatchEnvId, JsonString),
            Some(APPLY_PATCH_BACKEND_PATTERN_ENVID)
        );
        // The Decoded encoding is the host pattern, anchored for full-string match with
        // a dialect-portable end-of-input assertion (JSON-Schema `pattern` is otherwise a
        // substring match, and `$` admits a trailing newline under Python/PCRE).
        assert_eq!(
            backend_input_pattern(ApplyPatch, Decoded),
            Some(format!(r"^(?:{APPLY_PATCH_HOST_PATTERN_BASE})(?![\s\S])").as_str())
        );
        assert_eq!(
            backend_input_pattern(ApplyPatchEnvId, Decoded),
            Some(format!(r"^(?:{APPLY_PATCH_HOST_PATTERN_ENVID})(?![\s\S])").as_str())
        );
    }

    #[test]
    fn backend_pattern_fixtures_match_consts() {
        // The committed fixtures are the single source the Python feasibility gate
        // exercises under the pinned `XGrammar`; they must equal the runtime consts so
        // the gate proves the exact string the backend receives.
        assert_eq!(
            APPLY_PATCH_BACKEND_PATTERN_BASE,
            include_str!("apply_patch_backend_pattern_base.txt").trim_end_matches('\n'),
        );
        assert_eq!(
            APPLY_PATCH_BACKEND_PATTERN_ENVID,
            include_str!("apply_patch_backend_pattern_envid.txt").trim_end_matches('\n'),
        );
    }
}
