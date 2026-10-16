// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Validate MCP tool-call arguments against the tool's advertised input schema.
//!
//! `openai_mcp_tool_resolve` advertises each MCP tool's `inputSchema` to the
//! model as the function-tool `parameters`, but the model's emitted arguments
//! are otherwise forwarded to the upstream MCP server unchecked. This module
//! enforces that contract at the gateway, before any outbound `tools/call`, so a
//! model that hallucinates a wrong type, omits a required field, or supplies a
//! field the schema forbids is rejected with an actionable tool error instead of
//! triggering a side effect against the upstream server (issue #1125).
//!
//! The MCP specification already requires the tool server to validate its own
//! inputs; this gateway check is defense in depth, not a substitute, so it fails
//! open on a schema it cannot safely evaluate rather than bricking an otherwise
//! valid call for an upstream server's advertisement.
//!
//! Three properties keep the check safe against an adversarial, client-selected
//! advertisement:
//!
//! * **Bounded work.** A compact schema can expand combinatorially through repeated `$ref`s under
//!   `allOf`/`anyOf`/`oneOf`. Both `is_valid` (on an all-passing instance) and `validate`/`iter_errors` (which collect
//!   every branch failure) then run exponentially. The advertised schema's fully-expanded subschema count is estimated
//!   up front and validation is skipped when it exceeds [`WORK_BUDGET`], so no instance can pin a worker.
//! * **No instance disclosure.** The surfaced error is built only from the schema-derived `schema_path`. The library's
//!   rendered message and the instance path can embed instance-supplied values or dynamic object keys (for example a
//!   credential passed as an argument, or a `propertyNames` failure that quotes the offending key), so neither is used.
//! * **Bounded size.** The message is capped at [`MAX_MESSAGE_BYTES`]; the caller additionally fits it to the per-call
//!   retained-byte reservation.

use std::{collections::HashMap, sync::LazyLock};

use jsonschema::{ValidationError, Validator};
use serde_json::{Value, json};
use tracing::warn;

/// Upper bound, in bytes, on a validation error surfaced as a tool error.
///
/// A deeply nested schema can produce a long `schema_path`; capping it keeps the
/// warning log line and the stored tool result bounded. The caller fits the
/// message further to the retained-byte reservation.
const MAX_MESSAGE_BYTES: usize = 160;

/// Maximum estimated fully-expanded subschema count a schema may reach before
/// argument validation is skipped.
///
/// A legitimate tool `inputSchema` expands to at most a few hundred subschemas;
/// a combinatorial `$ref` bomb reaches millions. Skipping validation past this
/// threshold keeps both the validity decision and the error rendering bounded.
const WORK_BUDGET: u64 = 10_000;

/// Maximum schema nesting depth the work estimator descends before treating the
/// schema as over budget. Guards against stack exhaustion on a linear `$ref`
/// chain or a deeply nested literal schema; real schemas are shallow.
const MAX_SCHEMA_DEPTH: usize = 128;

/// The `{"type": "object"}` schema the resolver advertises when a tool carries
/// no `inputSchema`.
///
/// Validating against the same default the model was shown keeps dispatch
/// consistent with advertisement: a tool with no declared schema still must be
/// called with a JSON object, never a bare scalar or array.
static DEFAULT_OBJECT_SCHEMA: LazyLock<Value> = LazyLock::new(|| json!({"type": "object"}));

/// Return the input schema advertised for a dispatch tool-map entry's stored
/// `tool_definition`.
///
/// Accepts both the `camelCase` `inputSchema` from a fresh `tools/list` and the
/// `snake_case` `input_schema` of a cached listing, mirroring
/// `mcp_tool_to_function_tool` so dispatch enforces exactly what resolution
/// advertised. Returns `None` when no schema was advertised, so the caller
/// applies the shared `{"type": "object"}` default.
fn advertised_input_schema(tool_definition: Option<&Value>) -> Option<&Value> {
    let definition = tool_definition?;
    definition.get("inputSchema").or_else(|| definition.get("input_schema"))
}

/// Validate parsed tool-call `arguments` against the tool's advertised input
/// schema.
///
/// Returns `Ok(())` when the arguments conform — or when the advertised schema
/// cannot be compiled or is too complex to evaluate within [`WORK_BUDGET`] (fail
/// open; see the module docs). Returns `Err(message)` with a bounded, schema-only
/// explanation the caller surfaces as a tool error when the arguments violate the
/// schema. Extra fields are rejected only when the schema sets
/// `additionalProperties: false`; JSON Schema permits them by default, so this
/// check does not add stricter semantics than the advertisement.
pub(super) fn validate_arguments(tool_definition: Option<&Value>, arguments: &Value) -> Result<(), String> {
    let schema = advertised_input_schema(tool_definition).unwrap_or(&DEFAULT_OBJECT_SCHEMA);
    // Estimate before compiling: `validator_for` recurses through `$ref`s, so a
    // deep reference chain can overflow the stack during compilation. The
    // depth-capped estimate never recurses unbounded, so it is the safe gate.
    if schema_work_estimate(schema) > WORK_BUDGET {
        warn!(
            budget = WORK_BUDGET,
            "mcp_dispatch: tool input schema is too complex to validate within the work budget; skipping"
        );
        return Ok(());
    }
    let Ok(validator) = jsonschema::validator_for(schema) else {
        warn!("mcp_dispatch: tool input schema did not compile; skipping argument validation");
        return Ok(());
    };
    check(&validator, arguments)
}

/// Render the first schema violation as one bounded, schema-only line, or
/// `Ok(())` when the arguments conform.
fn check(validator: &Validator, arguments: &Value) -> Result<(), String> {
    // The work budget bounds this call, so `validate` (first error) is safe. The
    // message is built from `schema_path` only — never the instance path or the
    // library's rendered message, which can embed instance values or keys.
    match validator.validate(arguments) {
        Ok(()) => Ok(()),
        Err(error) => Err(describe(&error)),
    }
}

/// Build a schema-only, bounded description of a single violation.
fn describe(error: &ValidationError<'_>) -> String {
    let schema_path = error.schema_path().to_string();
    let message = if schema_path.is_empty() {
        "tool call arguments do not match the tool input schema".to_owned()
    } else {
        format!(
            "tool call arguments do not match the tool input schema (failed constraint at schema path '{schema_path}')"
        )
    };
    truncate_on_char_boundary(message, MAX_MESSAGE_BYTES)
}

/// Shorten `message` to at most `max_bytes`, appending an ellipsis when it was
/// truncated. Splits only on a UTF-8 character boundary so the result is valid.
pub(super) fn truncate_on_char_boundary(mut message: String, max_bytes: usize) -> String {
    if message.len() <= max_bytes {
        return message;
    }
    let mut end = max_bytes.min(message.len());
    while end > 0 && !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message.push('…');
    message
}

/// Estimate the number of subschema evaluations the advertised schema can induce.
///
/// Composition keywords (`allOf`/`anyOf`/`oneOf`) sum their children, so a chain
/// of `$ref`s that each fan out doubles the count per level — a compact schema
/// can force exponential validation work. Each `$ref` target is estimated once
/// (memoized) and the running total saturates at [`WORK_BUDGET`], so the estimate
/// stays linear in the schema size even when the true expansion is exponential.
/// `$ref` cycles and over-deep nesting resolve to over budget, so recursive
/// schemas simply fail open.
fn schema_work_estimate(schema: &Value) -> u64 {
    let mut memo: HashMap<String, u64> = HashMap::new();
    estimate(schema, schema, &mut memo, 0)
}

/// Recursive worker for [`schema_work_estimate`].
fn estimate(node: &Value, root: &Value, memo: &mut HashMap<String, u64>, depth: usize) -> u64 {
    if depth > MAX_SCHEMA_DEPTH {
        return WORK_BUDGET.saturating_add(1);
    }
    match node {
        Value::Object(map) => estimate_object(map, root, memo, depth),
        Value::Array(items) => estimate_sum(items.iter(), root, memo, depth, 0),
        _ => 0,
    }
}

/// Estimate an object schema.
///
/// `$defs`/`definitions` are stores reached only through `$ref`, so they are
/// skipped. Every other key is summed, including a `$ref` *and its siblings*:
/// JSON Schema 2020-12 evaluates sibling keywords alongside the reference, so
/// ignoring them would let a bomb hide next to a harmless reference.
fn estimate_object(
    map: &serde_json::Map<String, Value>,
    root: &Value,
    memo: &mut HashMap<String, u64>,
    depth: usize,
) -> u64 {
    let mut total: u64 = 1;
    for (key, child) in map {
        if key == "$defs" || key == "definitions" {
            continue;
        }
        let added = if key == "$ref" {
            child
                .as_str()
                .map_or(0, |reference| estimate_ref(reference, root, memo, depth))
        } else {
            estimate(child, root, memo, depth + 1)
        };
        total = total.saturating_add(added);
        if total > WORK_BUDGET {
            return total;
        }
    }
    total
}

/// Sum [`estimate`] over child schemas, saturating and stopping once the running
/// total exceeds [`WORK_BUDGET`]. `base` is the node's own weight (1 for a schema
/// object, 0 for an array of schemas).
fn estimate_sum<'a>(
    children: impl Iterator<Item = &'a Value>,
    root: &'a Value,
    memo: &mut HashMap<String, u64>,
    depth: usize,
    base: u64,
) -> u64 {
    let mut total = base;
    for child in children {
        total = total.saturating_add(estimate(child, root, memo, depth + 1));
        if total > WORK_BUDGET {
            return total;
        }
    }
    total
}

/// Estimate a `$ref`, memoizing by reference string and treating cycles and
/// non-local or unresolvable references conservatively.
fn estimate_ref(reference: &str, root: &Value, memo: &mut HashMap<String, u64>, depth: usize) -> u64 {
    if let Some(&cached) = memo.get(reference) {
        return cached;
    }
    let Some(target) = resolve_local_ref(reference, root) else {
        // The validator may resolve a form this estimator does not (an `$anchor`,
        // an absolute URI, an exotic encoding). Counting it as a cheap leaf would
        // let such a reference smuggle a bomb past the budget, so treat any
        // reference this estimator cannot follow as over budget (skip validation).
        return WORK_BUDGET.saturating_add(1);
    };
    // Seed with an over-budget sentinel so a reference that re-enters itself
    // (a cycle) resolves to over budget instead of recursing until the depth cap.
    memo.insert(reference.to_owned(), WORK_BUDGET.saturating_add(1));
    let value = estimate(target, root, memo, depth + 1);
    memo.insert(reference.to_owned(), value);
    value
}

/// Resolve a local (`#/...`) JSON-pointer `$ref` against the root schema.
///
/// A `$ref` is a URI reference, so its fragment is percent-encoded; it is decoded
/// before JSON-pointer parsing so an encoded pointer such as `#/$defs/%6422`
/// resolves to the same target the validator reaches (`#/$defs/d22`).
fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    let fragment = reference.strip_prefix('#')?;
    let decoded = percent_encoding::percent_decode_str(fragment).decode_utf8().ok()?;
    if decoded.is_empty() {
        return Some(root);
    }
    let mut current = root;
    for raw in decoded.split('/').skip(1) {
        let token = raw.replace("~1", "/").replace("~0", "~");
        current = match current {
            Value::Object(map) => map.get(&token)?,
            Value::Array(items) => items.get(token.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(current)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::exit,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use super::*;

    /// A tool definition carrying a strict object schema with one required
    /// string field and no additional properties.
    fn strict_definition() -> Value {
        json!({
            "name": "get_weather",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false,
            },
        })
    }

    #[test]
    fn conforming_arguments_pass() {
        let def = strict_definition();
        assert!(validate_arguments(Some(&def), &json!({"city": "Paris"})).is_ok());
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let def = strict_definition();
        let error = validate_arguments(Some(&def), &json!({})).unwrap_err();
        assert!(
            error.contains("required"),
            "error should name the failed constraint: {error}"
        );
    }

    #[test]
    fn type_mismatch_is_rejected_with_schema_path() {
        let def = strict_definition();
        let error = validate_arguments(Some(&def), &json!({"city": 42})).unwrap_err();
        assert!(
            error.contains("properties/city"),
            "error should point at the offending field via the schema path: {error}"
        );
    }

    #[test]
    fn offending_value_is_excluded_from_the_error() {
        // A model may pass private content or a credential as an argument value.
        // The surfaced error is built from the schema path only, so the value
        // never appears. The field is typed `integer` so the secret string is a
        // genuine violation whose library rendering would otherwise quote it.
        let def = json!({
            "name": "set_limit",
            "inputSchema": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": ["limit"],
            },
        });
        let secret = "sk-super-secret-credential-value";
        let error = validate_arguments(Some(&def), &json!({"limit": secret})).unwrap_err();
        assert!(
            !error.contains(secret),
            "the offending value must not appear in the error: {error}"
        );
        assert!(
            error.contains("limit"),
            "the error should still locate the field via the schema path: {error}"
        );
    }

    #[test]
    fn dynamic_property_name_is_excluded_from_the_error() {
        // `propertyNames` failures and `additionalProperties` rejections name the
        // offending instance key in the library's rendering. The key itself can
        // be sensitive, so the schema-only message must not echo it.
        let property_names_def = json!({
            "name": "tagger",
            "inputSchema": {
                "type": "object",
                "propertyNames": {"pattern": "^[a-z]+$"},
            },
        });
        let secret_key = "sk-secret-123";
        let error = validate_arguments(Some(&property_names_def), &json!({secret_key: "x"})).unwrap_err();
        assert!(
            !error.contains(secret_key),
            "a dynamic key must not leak through a propertyNames error: {error}"
        );

        let forbidden_def = strict_definition();
        let extra_error =
            validate_arguments(Some(&forbidden_def), &json!({"city": "Paris", "sk-secret-456": "x"})).unwrap_err();
        assert!(
            !extra_error.contains("sk-secret-456"),
            "a forbidden key must not leak through an additionalProperties error: {extra_error}"
        );
    }

    #[test]
    fn extra_field_rejected_only_when_additional_properties_false() {
        let strict = strict_definition();
        let strict_error = validate_arguments(Some(&strict), &json!({"city": "Paris", "unit": "c"})).unwrap_err();
        assert!(
            strict_error.to_lowercase().contains("additionalproperties"),
            "additionalProperties:false must reject the extra field: {strict_error}"
        );

        // JSON Schema permits extra fields by default, so the same arguments pass
        // against a schema that does not forbid them.
        let lenient = json!({
            "name": "get_weather",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        });
        assert!(validate_arguments(Some(&lenient), &json!({"city": "Paris", "unit": "c"})).is_ok());
    }

    #[test]
    fn snake_case_input_schema_spelling_is_honored() {
        let def = json!({
            "name": "get_weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        });
        assert!(validate_arguments(Some(&def), &json!({"city": "Paris"})).is_ok());
        assert!(validate_arguments(Some(&def), &json!({})).is_err());
    }

    #[test]
    fn absent_schema_defaults_to_object_and_rejects_non_objects() {
        let def = json!({"name": "get_weather"});
        assert!(validate_arguments(Some(&def), &json!({"anything": true})).is_ok());
        assert!(validate_arguments(Some(&def), &json!("a string")).is_err());
        assert!(validate_arguments(Some(&def), &json!([1, 2, 3])).is_err());
        // A missing tool_definition behaves identically to an absent schema.
        assert!(validate_arguments(None, &json!({})).is_ok());
        assert!(validate_arguments(None, &json!(5)).is_err());
    }

    #[test]
    fn uncompilable_schema_fails_open() {
        // `type` must be a string or array of strings; a number makes the schema
        // uncompilable, so validation is skipped rather than failing the call.
        let def = json!({"name": "broken", "inputSchema": {"type": 123}});
        assert!(validate_arguments(Some(&def), &json!({"anything": "goes"})).is_ok());
        assert!(validate_arguments(Some(&def), &json!("even a scalar")).is_ok());
    }

    #[test]
    fn deep_schema_path_message_is_truncated() {
        // A deeply (but legitimately) nested schema yields a long schema path;
        // the message must still stay within the byte cap.
        let mut schema = json!({"type": "string"});
        for _ in 0..40 {
            schema = json!({"type": "object", "properties": {"child": schema}, "required": ["child"]});
        }
        let def = json!({"name": "nested", "inputSchema": schema});
        // Drill a conforming object down to the leaf, then violate the leaf type.
        let mut args = json!(42);
        for _ in 0..40 {
            args = json!({"child": args});
        }
        let error = validate_arguments(Some(&def), &args).unwrap_err();
        assert!(
            error.len() <= MAX_MESSAGE_BYTES + '…'.len_utf8(),
            "message must stay within the byte cap: {} bytes",
            error.len()
        );
    }

    #[test]
    fn under_budget_composed_schema_still_validates() {
        // A modest `allOf` is well within budget and must still be enforced.
        let def = json!({
            "name": "composed",
            "inputSchema": {
                "allOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                    {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
                ],
            },
        });
        assert!(validate_arguments(Some(&def), &json!({"a": "x", "b": 1})).is_ok());
        assert!(validate_arguments(Some(&def), &json!({"a": "x"})).is_err());
    }

    /// `$defs` for an `allOf`-doubling `$ref` bomb: `d_n` references `d_{n-1}`
    /// twice, so a full expansion is `2^n` leaves while the definition stays
    /// compact.
    fn bomb_defs(depth: usize) -> serde_json::Map<String, Value> {
        let mut defs = serde_json::Map::new();
        defs.insert("d0".to_owned(), json!({"not": {}}));
        for i in 1..=depth {
            let child = format!("#/$defs/d{}", i - 1);
            defs.insert(format!("d{i}"), json!({"allOf": [{"$ref": child}, {"$ref": child}]}));
        }
        defs
    }

    /// A tool definition whose schema reaches the bomb through `wrapper`.
    fn ref_bomb(depth: usize, wrapper: &str) -> Value {
        json!({
            "name": "boom",
            "inputSchema": {
                "type": "object",
                wrapper: [{"$ref": format!("#/$defs/d{depth}")}],
                "$defs": Value::Object(bomb_defs(depth)),
            },
        })
    }

    #[test]
    fn combinatorial_allof_schema_is_skipped_within_the_work_budget() {
        use std::time::{Duration, Instant};
        let def = ref_bomb(22, "allOf");
        let start = Instant::now();
        let result = validate_arguments(Some(&def), &json!({}));
        let elapsed = start.elapsed();
        // Over budget: validation is skipped (fail open), and the estimate stays
        // linear so this completes far faster than enumerating 2^22 branches.
        assert!(result.is_ok(), "an over-budget schema must fail open, not reject");
        assert!(
            elapsed < Duration::from_millis(500),
            "the work estimate must not expand 2^22 branches; took {elapsed:?}"
        );
    }

    #[test]
    fn combinatorial_anyof_schema_is_skipped_within_the_work_budget() {
        use std::time::{Duration, Instant};
        // Wrapping the bomb in `anyOf` is the composed-failure case: `validate`
        // collects every branch failure, so the up-front budget must gate it too.
        let def = ref_bomb(22, "anyOf");
        let start = Instant::now();
        let result = validate_arguments(Some(&def), &json!({}));
        let elapsed = start.elapsed();
        assert!(result.is_ok(), "an over-budget anyOf schema must fail open");
        assert!(
            elapsed < Duration::from_millis(500),
            "the anyOf bomb must not be enumerated; took {elapsed:?}"
        );
    }

    #[test]
    fn recursive_schema_fails_open() {
        // A self-referential schema expands without bound in the static estimate,
        // so it is skipped rather than risking unbounded work.
        let def = json!({
            "name": "tree",
            "inputSchema": {
                "type": "object",
                "properties": {"child": {"$ref": "#"}},
            },
        });
        assert!(validate_arguments(Some(&def), &json!({"child": {"child": {}}})).is_ok());
    }

    // --- Deterministic work-estimate assertions (no wall-clock) ---------------

    #[test]
    fn work_estimate_is_small_for_a_normal_schema() {
        let schema = json!({
            "type": "object",
            "properties": {"city": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["city"],
            "additionalProperties": false,
        });
        assert!(
            schema_work_estimate(&schema) < WORK_BUDGET,
            "a normal tool schema must stay well under budget"
        );
    }

    #[test]
    fn work_estimate_exceeds_budget_for_allof_bomb() {
        let schema = ref_bomb(22, "allOf")["inputSchema"].clone();
        assert!(schema_work_estimate(&schema) > WORK_BUDGET);
    }

    #[test]
    fn work_estimate_counts_ref_siblings() {
        // 2020-12 evaluates keywords beside a `$ref`, so a bomb hidden next to a
        // harmless reference must still be counted (regression: early-return on
        // `$ref` let it bypass the guard).
        let mut defs = bomb_defs(22);
        defs.insert("harmless".to_owned(), json!({"type": "object"}));
        let schema = json!({
            "$ref": "#/$defs/harmless",
            "allOf": [{"$ref": "#/$defs/d22"}],
            "$defs": Value::Object(defs),
        });
        assert!(
            schema_work_estimate(&schema) > WORK_BUDGET,
            "a bomb beside a harmless $ref must not slip through"
        );
    }

    #[test]
    fn work_estimate_follows_percent_encoded_refs() {
        // `%64` decodes to 'd', so `#/$defs/%6422` targets `d22` just as the
        // validator resolves it (regression: estimator counted it as one leaf).
        let schema = json!({
            "allOf": [{"$ref": "#/$defs/%6422"}],
            "$defs": Value::Object(bomb_defs(22)),
        });
        assert!(
            schema_work_estimate(&schema) > WORK_BUDGET,
            "a percent-encoded $ref must resolve to its real target"
        );
    }

    #[test]
    fn percent_encoded_ref_resolves_to_its_target() {
        let schema = json!({"$defs": {"d22": {"const": "x"}}});
        assert_eq!(
            resolve_local_ref("#/$defs/%6422", &schema),
            Some(&json!({"const": "x"}))
        );
    }

    #[test]
    fn work_estimate_exceeds_budget_for_deep_linear_chain() {
        let mut defs = serde_json::Map::new();
        defs.insert("d0".to_owned(), json!({"type": "object"}));
        for i in 1..=500 {
            defs.insert(format!("d{i}"), json!({"$ref": format!("#/$defs/d{}", i - 1)}));
        }
        let schema = json!({"$ref": "#/$defs/d500", "$defs": Value::Object(defs)});
        assert!(
            schema_work_estimate(&schema) > WORK_BUDGET,
            "a deep $ref chain must exceed the budget via the depth cap"
        );
    }

    // --- Production-path (validate_arguments) adversarial regressions ---------

    #[test]
    fn ref_sibling_bomb_is_skipped_in_validation() {
        use std::time::{Duration, Instant};
        let mut defs = bomb_defs(22);
        defs.insert("harmless".to_owned(), json!({"type": "object"}));
        let def = json!({
            "name": "sibling_boom",
            "inputSchema": {
                "$ref": "#/$defs/harmless",
                "allOf": [{"$ref": "#/$defs/d22"}],
                "$defs": Value::Object(defs),
            },
        });
        let start = Instant::now();
        assert!(validate_arguments(Some(&def), &json!({})).is_ok());
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "sibling bomb must not be enumerated"
        );
    }

    #[test]
    fn percent_encoded_ref_bomb_is_skipped_in_validation() {
        use std::time::{Duration, Instant};
        let def = json!({
            "name": "encoded_boom",
            "inputSchema": {"allOf": [{"$ref": "#/$defs/%6422"}], "$defs": Value::Object(bomb_defs(22))},
        });
        let start = Instant::now();
        assert!(validate_arguments(Some(&def), &json!({})).is_ok());
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "encoded bomb must not be enumerated"
        );
    }

    #[test]
    fn deep_ref_chain_does_not_crash_the_process() {
        // A deep `$ref` chain overflows the stack inside `validator_for` if
        // compilation runs before the depth-capped estimate. Run the check in an
        // isolated subprocess so a regression surfaces as a non-zero child exit,
        // not a crash of the whole test binary.
        const MARKER: &str = "DEEP_REF_CHAIN_OK";
        const TEST_PATH: &str =
            "openai::responses::mcp_dispatch::validate::tests::deep_ref_chain_does_not_crash_the_process";

        if std::env::var("PRAXIS_MCP_DEEP_REF_CHILD").is_ok() {
            let mut defs = serde_json::Map::new();
            defs.insert("d0".to_owned(), json!({"type": "object"}));
            for i in 1..=3000 {
                defs.insert(format!("d{i}"), json!({"$ref": format!("#/$defs/d{}", i - 1)}));
            }
            let def = json!({
                "name": "chain",
                "inputSchema": {"$ref": "#/$defs/d3000", "$defs": Value::Object(defs)},
            });
            // Must return (fail open) without overflowing the stack.
            assert!(validate_arguments(Some(&def), &json!({})).is_ok());
            println!("{MARKER}");
            std::process::exit(0);
        }

        let exe = std::env::current_exe().expect("current test binary path");
        let output = std::process::Command::new(exe)
            .args(["--exact", TEST_PATH, "--nocapture"])
            .env("PRAXIS_MCP_DEEP_REF_CHILD", "1")
            .output()
            .expect("spawn isolated child test process");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "deep $ref chain must not crash the process: status={:?}\nstdout={stdout}\nstderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(
            stdout.contains(MARKER),
            "child did not run the deep-chain check; stdout={stdout}"
        );
    }
}
