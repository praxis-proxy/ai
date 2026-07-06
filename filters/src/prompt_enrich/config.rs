// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the prompt enrichment filter.

use praxis_filter::{
    FilterError, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// PromptEnrichConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the prompt enrichment filter.
///
/// ```yaml
/// filter: prompt_enrich
/// max_body_bytes: 10485760
/// on_invalid: continue
/// prepend:
///   - role: system
///     content: "You are a helpful assistant."
/// append:
///   - role: user
///     content: "Remember to cite your sources."
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PromptEnrichConfig {
    /// Maximum request body size to buffer before parsing.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Behavior when the body is not valid JSON or lacks a
    /// `messages` array.
    #[serde(default)]
    pub on_invalid: InvalidBodyBehavior,

    /// Messages to prepend at the beginning of the `messages` array.
    #[serde(default)]
    pub prepend: Vec<MessageConfig>,

    /// Messages to append at the end of the `messages` array.
    #[serde(default)]
    pub append: Vec<MessageConfig>,
}

/// Default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

// -----------------------------------------------------------------------------
// MessageConfig
// -----------------------------------------------------------------------------

/// A single message to inject into the `messages` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MessageConfig {
    /// Role for the injected message.
    pub role: MessageRole,

    /// Text content of the injected message.
    pub content: String,
}

// -----------------------------------------------------------------------------
// MessageRole
// -----------------------------------------------------------------------------

// v1 intentionally supports only roles requested by issue #137.

/// Allowed roles for injected messages.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum MessageRole {
    /// System role, allowed in both `prepend` and `append`.
    System,

    /// User role, allowed only in `append`.
    User,
}

// -----------------------------------------------------------------------------
// InvalidBodyBehavior
// -----------------------------------------------------------------------------

/// Behavior when the request body cannot be enriched.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum InvalidBodyBehavior {
    /// Pass the original body through unchanged.
    #[default]
    Continue,

    /// Return HTTP 400.
    Reject,
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate a parsed config, returning an error for invalid combinations.
///
/// # Errors
///
/// Returns [`FilterError`] if:
/// - Both `prepend` and `append` are empty
/// - Any message has empty `content`
/// - `max_body_bytes` is zero
/// - A `prepend` message uses a role other than `system`
///
/// [`FilterError`]: praxis_filter::FilterError
pub(super) fn validate_config(cfg: &PromptEnrichConfig) -> Result<(), FilterError> {
    if cfg.prepend.is_empty() && cfg.append.is_empty() {
        return Err("prompt_enrich: at least one of 'prepend' or 'append' must be non-empty".into());
    }

    validate_max_body_bytes("prompt_enrich", cfg.max_body_bytes)?;

    for msg in &cfg.prepend {
        if msg.content.is_empty() {
            return Err("prompt_enrich: message 'content' must not be empty".into());
        }
        if msg.role != MessageRole::System {
            return Err("prompt_enrich: 'prepend' messages must use role 'system'".into());
        }
    }

    for msg in &cfg.append {
        if msg.content.is_empty() {
            return Err("prompt_enrich: message 'content' must not be empty".into());
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Serialization Helpers
// -----------------------------------------------------------------------------

/// Convert a [`MessageConfig`] to a [`serde_json::Value`] for injection.
pub(super) fn message_to_value(msg: &MessageConfig) -> serde_json::Value {
    let role_str = match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
    };
    serde_json::json!({
        "role": role_str,
        "content": msg.content,
    })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    // -- Serde defaults -------------------------------------------------------

    #[test]
    fn serde_defaults_prompt_enrich_config() {
        let cfg: PromptEnrichConfig = serde_yaml::from_str(
            r#"
prepend:
  - role: system
    content: hello
"#,
        )
        .unwrap();

        assert_eq!(cfg.max_body_bytes, DEFAULT_JSON_BODY_MAX_BYTES);
        assert_eq!(cfg.on_invalid, InvalidBodyBehavior::Continue);
        assert!(cfg.append.is_empty());
    }

    #[test]
    fn invalid_body_behavior_defaults_to_continue() {
        let b = InvalidBodyBehavior::default();
        assert_eq!(b, InvalidBodyBehavior::Continue);
    }

    // -- MessageRole serde ----------------------------------------------------

    #[test]
    fn message_role_serde_system() {
        let r: MessageRole = serde_yaml::from_str("system").unwrap();
        assert_eq!(r, MessageRole::System);
    }

    #[test]
    fn message_role_serde_user() {
        let r: MessageRole = serde_yaml::from_str("user").unwrap();
        assert_eq!(r, MessageRole::User);
    }

    #[test]
    fn message_role_serde_unknown_rejected() {
        let res = serde_yaml::from_str::<MessageRole>("admin");
        assert!(res.is_err());
    }

    // -- InvalidBodyBehavior serde --------------------------------------------

    #[test]
    fn invalid_body_behavior_serde_continue() {
        let b: InvalidBodyBehavior = serde_yaml::from_str("continue").unwrap();
        assert_eq!(b, InvalidBodyBehavior::Continue);
    }

    #[test]
    fn invalid_body_behavior_serde_reject() {
        let b: InvalidBodyBehavior = serde_yaml::from_str("reject").unwrap();
        assert_eq!(b, InvalidBodyBehavior::Reject);
    }

    // -- deny_unknown_fields --------------------------------------------------

    #[test]
    fn deny_unknown_fields_prompt_enrich_config() {
        let res = serde_yaml::from_str::<PromptEnrichConfig>(
            r#"
prepend:
  - role: system
    content: hello
unknown_field: true
"#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn deny_unknown_fields_message_config() {
        let res = serde_yaml::from_str::<MessageConfig>(
            r#"
role: system
content: hello
extra: true
"#,
        );
        assert!(res.is_err());
    }

    // -- validate_config ------------------------------------------------------

    #[test]
    fn validate_empty_prepend_and_append_rejected() {
        let cfg = PromptEnrichConfig {
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            on_invalid: InvalidBodyBehavior::Continue,
            prepend: vec![],
            append: vec![],
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("at least one"),
            "expected 'at least one' error, got: {err}"
        );
    }

    #[test]
    fn validate_prepend_only_ok() {
        let cfg = PromptEnrichConfig {
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            on_invalid: InvalidBodyBehavior::Continue,
            prepend: vec![MessageConfig {
                role: MessageRole::System,
                content: "hello".into(),
            }],
            append: vec![],
        };
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn validate_append_only_ok() {
        let cfg = PromptEnrichConfig {
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            on_invalid: InvalidBodyBehavior::Continue,
            prepend: vec![],
            append: vec![MessageConfig {
                role: MessageRole::User,
                content: "cite sources".into(),
            }],
        };
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn validate_empty_content_rejected() {
        let cfg = PromptEnrichConfig {
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            on_invalid: InvalidBodyBehavior::Continue,
            prepend: vec![MessageConfig {
                role: MessageRole::System,
                content: String::new(),
            }],
            append: vec![],
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("content"),
            "expected 'content' error, got: {err}"
        );
    }

    #[test]
    fn validate_prepend_with_user_role_rejected() {
        let cfg = PromptEnrichConfig {
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            on_invalid: InvalidBodyBehavior::Continue,
            prepend: vec![MessageConfig {
                role: MessageRole::User,
                content: "not allowed".into(),
            }],
            append: vec![],
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("system"),
            "expected 'system' role error, got: {err}"
        );
    }

    #[test]
    fn validate_zero_max_body_bytes_rejected() {
        let cfg = PromptEnrichConfig {
            max_body_bytes: 0,
            on_invalid: InvalidBodyBehavior::Continue,
            prepend: vec![MessageConfig {
                role: MessageRole::System,
                content: "hello".into(),
            }],
            append: vec![],
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("must be greater than 0"),
            "expected 'must be greater than 0' error, got: {err}"
        );
    }

    // -- message_to_value -----------------------------------------------------

    #[test]
    fn message_to_value_system() {
        let msg = MessageConfig {
            role: MessageRole::System,
            content: "be helpful".into(),
        };
        let v = message_to_value(&msg);
        assert_eq!(v["role"], "system");
        assert_eq!(v["content"], "be helpful");
    }

    #[test]
    fn message_to_value_user() {
        let msg = MessageConfig {
            role: MessageRole::User,
            content: "cite sources".into(),
        };
        let v = message_to_value(&msg);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "cite sources");
    }
}
