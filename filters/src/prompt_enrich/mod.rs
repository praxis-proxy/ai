// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Prompt enrichment filter: injects configured messages into
//! OpenAI-compatible chat completion request bodies.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::json_body::replace_json_body;
#[cfg(feature = "openai-responses")]
use praxis_ai_apis::openai::AgenticBudgetPolicy;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};

use self::config::{InvalidBodyBehavior, PromptEnrichConfig, message_to_value, validate_config};

// -----------------------------------------------------------------------------
// PromptEnrichFilter
// -----------------------------------------------------------------------------

/// Injects statically configured messages into the `messages`
/// array of OpenAI-compatible chat completion request bodies.
///
/// Messages are pre-serialized at construction time. At
/// request time, the filter parses the JSON body, splices
/// prepend messages at the beginning and appends messages at
/// the end, then re-serializes the modified body.
///
/// At least one of `prepend` or `append` must be non-empty.
/// JSON is re-serialized, so byte-for-byte body identity is
/// not preserved.
///
/// In chains that also use `json_body_field` or
/// `model_to_header`, place `prompt_enrich` first.
///
/// # YAML configuration
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
///
/// # Example
///
/// ```rust
/// use praxis_ai_filters::PromptEnrichFilter;
///
/// let yaml: serde_yaml::Value = serde_yaml::from_str(
///     r#"
/// prepend:
///   - role: system
///     content: "You are a helpful assistant."
/// "#,
/// )
/// .unwrap();
/// let filter = PromptEnrichFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "prompt_enrich");
/// ```
pub struct PromptEnrichFilter {
    /// Pre-serialized messages to append after existing messages.
    append: Vec<serde_json::Value>,

    /// Per-request charge for copies of the configured messages during enrichment.
    #[cfg(feature = "openai-responses")]
    enrichment_charge: usize,

    /// Maximum request body size to buffer.
    max_body_bytes: usize,

    /// Behavior when the body cannot be enriched.
    on_invalid: InvalidBodyBehavior,

    /// Pre-serialized messages to prepend before existing messages.
    prepend: Vec<serde_json::Value>,
}

impl PromptEnrichFilter {
    /// Create from parsed YAML config.
    ///
    /// Validates the config and pre-serializes all configured
    /// messages to [`serde_json::Value`] at construction time.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    ///
    /// ```rust
    /// use praxis_ai_filters::PromptEnrichFilter;
    ///
    /// let yaml: serde_yaml::Value =
    ///     serde_yaml::from_str("prepend:\n  - role: system\n    content: \"Hello\"").unwrap();
    /// let filter = PromptEnrichFilter::from_config(&yaml).unwrap();
    /// assert_eq!(filter.name(), "prompt_enrich");
    /// ```
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: PromptEnrichConfig = parse_filter_config("prompt_enrich", config)?;
        validate_config(&cfg)?;

        let prepend: Vec<_> = cfg.prepend.iter().map(message_to_value).collect();
        let append: Vec<_> = cfg.append.iter().map(message_to_value).collect();
        #[cfg(feature = "openai-responses")]
        let enrichment_charge = configured_enrichment_charge(&prepend, &append);

        Ok(Box::new(Self {
            append,
            #[cfg(feature = "openai-responses")]
            enrichment_charge,
            max_body_bytes: cfg.max_body_bytes,
            on_invalid: cfg.on_invalid,
            prepend,
        }))
    }
}

#[async_trait]
impl HttpFilter for PromptEnrichFilter {
    fn name(&self) -> &'static str {
        "prompt_enrich"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the budgeted body preflight and enrichment share one borrowed body"
    )]
    async fn on_request_body(
        &self,
        #[cfg_attr(
            not(feature = "openai-responses"),
            expect(unused_variables, reason = "the budgeted create guard is feature-gated")
        )]
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(raw) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        #[cfg(feature = "openai-responses")]
        let budget_policy = budgeted_create_policy(ctx);
        #[cfg(feature = "openai-responses")]
        if budget_policy.is_some_and(|policy| !policy.has_body_headroom(ctx, raw, 0)) {
            return Ok(reject_retained_enrichment_budget());
        }

        let mut value: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(_) => return Ok(invalid_body_action(self.on_invalid, "invalid JSON body")),
        };

        let Some(messages) = value.get_mut("messages").and_then(serde_json::Value::as_array_mut) else {
            return Ok(invalid_body_action(
                self.on_invalid,
                "missing or invalid messages array",
            ));
        };

        #[cfg(feature = "openai-responses")]
        if budget_policy.is_some_and(|policy| !policy.reserve_body_projection(ctx, raw, self.enrichment_charge)) {
            return Ok(reject_retained_enrichment_budget());
        }

        messages.splice(0..0, self.prepend.iter().cloned());
        messages.extend(self.append.iter().cloned());

        replace_json_body(body, &value, self.name(), "messages")
            .map_err(|e| -> FilterError { format!("{}: {e}", self.name()).into() })?;

        Ok(FilterAction::Continue)
    }
}

/// Find the budget only for a Responses create on this listener.
#[cfg(feature = "openai-responses")]
fn budgeted_create_policy(ctx: &HttpFilterContext<'_>) -> Option<AgenticBudgetPolicy> {
    (ctx.request.method == http::Method::POST && ctx.request.uri.path().trim_end_matches('/') == "/v1/responses")
        .then(|| ctx.extensions.get::<AgenticBudgetPolicy>().copied())
        .flatten()
}

/// Charge configured JSON at construction, before request-time cloning.
#[cfg(feature = "openai-responses")]
fn configured_enrichment_charge(prepend: &[serde_json::Value], append: &[serde_json::Value]) -> usize {
    prepend
        .iter()
        .chain(append)
        .try_fold(0_usize, |total, message| {
            let bytes = serde_json::to_vec(message).ok()?;
            total.checked_add(AgenticBudgetPolicy::input_charge(&bytes)?)
        })
        .unwrap_or(usize::MAX)
}

/// Reject an enrichment projection that exceeds the listener's retained budget.
#[cfg(feature = "openai-responses")]
fn reject_retained_enrichment_budget() -> FilterAction {
    FilterAction::Reject(
        Rejection::status(413)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from_static(
                br#"{"error":{"type":"invalid_request_error","message":"prompt enrichment exceeds openai_agentic_loop.max_retained_bytes","param":null,"code":"invalid_request_error"}}"#,
            )),
    )
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Map [`InvalidBodyBehavior`] to the appropriate [`FilterAction`].
fn invalid_body_action(behavior: InvalidBodyBehavior, message: &'static str) -> FilterAction {
    match behavior {
        InvalidBodyBehavior::Continue => FilterAction::Continue,
        InvalidBodyBehavior::Reject => FilterAction::Reject(
            Rejection::status(400)
                .with_header("content-type", "text/plain")
                .with_body(message),
        ),
    }
}
