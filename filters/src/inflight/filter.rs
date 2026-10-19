// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! The `inflight_tracker` HTTP filter.
//!
//! On request start this filter increments a per-model in-flight count and adds
//! the request's `max_tokens`/`max_completion_tokens`/`max_output_tokens` to a
//! per-model reservation. Both figures live in [`InFlightRegistry`], a pipeline
//! extension a scorer can read to bias model selection toward the
//! least-loaded backend.
//!
//! The decrement is driven by [`Drop`], not a response hook. On request start
//! the filter parks an `InFlightGuard` in `ctx.filter_state`; the per-request
//! context is dropped on *every* terminal path — success, client abort, upstream
//! error, timeout — so the guard's `Drop` always runs exactly once and the
//! counters can never leak upward. Response hooks are not reliable on aborted
//! streams, so the filter has none.

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};

use super::{
    config::{InFlightConfig, validate_config},
    registry::InFlightRegistry,
    scan::FieldScan,
};

/// Public filter type name used in YAML and registration.
pub const FILTER_NAME: &str = "inflight_tracker";

/// Tracks live request counts and reserved tokens per model.
pub struct InFlightTrackerFilter {
    /// Model attributed when the request body reveals none.
    default_model: String,
}

impl InFlightTrackerFilter {
    /// Build from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: InFlightConfig = parse_filter_config(FILTER_NAME, config)?;
        validate_config(&cfg)?;
        Ok(Box::new(Self {
            default_model: cfg.default_model,
        }))
    }

    /// Commit the accumulated scan as one in-flight count, parking the guard.
    ///
    /// A request that scanned nothing (no body) is left uncounted. The in-flight
    /// window is then `[request complete, response complete]`, the span a load
    /// scorer cares about.
    fn commit(&self, ctx: &mut HttpFilterContext<'_>) {
        let Some((model, cap)) = ctx
            .get_filter_state::<FieldScan>()
            .map(|scan| (scan.model().map(str::to_owned), scan.cap()))
        else {
            return;
        };
        let Some(registry) = ctx.extensions.get::<InFlightRegistry>().cloned() else {
            return; // extension not installed; no-op
        };
        let model = model.unwrap_or_else(|| self.default_model.clone());
        let reserved = registry.on_request_start(&model, cap.unwrap_or(0));
        ctx.insert_filter_state(InFlightGuard {
            registry,
            model,
            reserved,
        });
    }
}

/// RAII guard parked in `ctx.filter_state` for the request's lifetime.
///
/// Its [`Drop`] decrements the registry by exactly what request start reserved.
/// Because the per-request context owns this guard and is dropped on every
/// terminal path, the decrement runs once no matter how the request ends. The
/// guard carries its own [`InFlightRegistry`] handle — a cheap `Arc` clone —
/// because when it drops there is no `ctx` left to reach the shared table
/// through.
struct InFlightGuard {
    /// Shared handle to the counter table (clones share one inner map).
    registry: InFlightRegistry,

    /// Model this request was attributed to, so completion decrements the same
    /// key that start incremented.
    model: String,

    /// Exact token amount added on start, subtracted verbatim on completion.
    reserved: u64,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.registry.on_request_complete(&self.model, self.reserved);
    }
}

#[async_trait]
impl HttpFilter for InFlightTrackerFilter {
    fn name(&self) -> &'static str {
        "inflight_tracker"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        // Not StreamBuffer: it rejects bodies over its limit with 413, and a
        // passive tracker must never cap request size. FieldScan reads the
        // fields incrementally, so plain Stream suffices.
        BodyMode::Stream
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // Already counted (guard parked): nothing more to do this request.
        if ctx.get_filter_state::<InFlightGuard>().is_some() {
            return Ok(FilterAction::Continue);
        }

        // Scan each chunk as it streams; commit the count once at end-of-stream.
        if let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) {
            scan_chunk(ctx, bytes);
        }
        if end_of_stream {
            self.commit(ctx);
        }

        Ok(FilterAction::Continue)
    }
}

/// Feed a streamed body chunk into the request's [`FieldScan`], creating it on
/// the first nonempty chunk. Creating the state records that a body was seen,
/// so a bodyless request stays uncounted. The scan carries bounded state
/// across chunks (nesting depth, string/escape state, the field currently
/// being read) and never buffers the body, so a `model` or token cap arriving
/// after megabytes of preceding content, or split across a chunk boundary, is
/// still found; see [`FieldScan`] for the scanning contract.
fn scan_chunk(ctx: &mut HttpFilterContext<'_>, bytes: &[u8]) {
    if ctx.get_filter_state::<FieldScan>().is_none() {
        ctx.insert_filter_state(FieldScan::default());
    }
    let Some(scan) = ctx.get_filter_state_mut::<FieldScan>() else {
        return;
    };
    scan.feed_bytes(bytes);
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::{super::registry::ModelMetrics, *};

    #[test]
    fn guard_drop_decrements_on_every_path() {
        // The decrement is welded to the guard's lifetime: whatever ends the
        // request, dropping the guard nets the reservation back out.
        let registry = InFlightRegistry::new();
        let reserved = registry.on_request_start("gpt-4", 100);
        {
            let _guard = InFlightGuard {
                registry: registry.clone(),
                model: "gpt-4".to_owned(),
                reserved,
            };
            assert_eq!(
                registry.get("gpt-4"),
                ModelMetrics {
                    in_flight: 1,
                    reserved_tokens: 100
                }
            );
        } // guard dropped here, as if the per-request context were torn down

        assert_eq!(registry.get("gpt-4"), ModelMetrics::default());
    }

    #[tokio::test]
    async fn attributes_top_level_model_across_chunk_boundary_not_nested_decoy() {
        // The chunk split lands so the second chunk starts inside the nested
        // `metadata` object: scanning chunks independently would read
        // {"model":"decoy"} as top-level. A depth-correct scan of the joined
        // prefix must attribute the real top-level model instead.
        let filter = InFlightTrackerFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let registry = InFlightRegistry::new();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        ctx.extensions.insert(registry.clone());

        let mut chunk1 = Some(Bytes::from_static(br#"{"metadata":"#));
        let a1 = filter.on_request_body(&mut ctx, &mut chunk1, false).await.unwrap();
        assert!(matches!(a1, FilterAction::Continue));
        let mut chunk2 = Some(Bytes::from_static(
            br#"{"model":"decoy"},"model":"gpt-4","max_tokens":5}"#,
        ));
        let a2 = filter.on_request_body(&mut ctx, &mut chunk2, true).await.unwrap();
        assert!(matches!(a2, FilterAction::Continue));

        assert_eq!(
            registry.get("gpt-4"),
            ModelMetrics {
                in_flight: 1,
                reserved_tokens: 5
            },
            "the real top-level model must win over the nested decoy"
        );
        assert_eq!(
            registry.get("decoy"),
            ModelMetrics::default(),
            "a model nested inside metadata must never be counted"
        );
    }

    #[tokio::test]
    async fn attributes_model_streamed_well_past_a_16kib_prefix() {
        // Real client SDKs do not put `model` first (anthropic-python sends
        // max_tokens, messages, model; openai-python puts max_tokens after
        // messages), so a long conversation must not push these fields out of
        // reach of a bounded prefix scan.
        let filter = InFlightTrackerFilter::from_config(&serde_yaml::Value::Null).unwrap();
        let registry = InFlightRegistry::new();
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        ctx.extensions.insert(registry.clone());

        let filler = "a".repeat(20 * 1024);
        let prefix = format!(r#"{{"messages":[{{"content":"{filler}"}}],"#);
        let suffix = br#""model":"gpt-4","max_tokens":5}"#;

        let mut chunk1 = Some(Bytes::from(prefix.into_bytes()));
        let a1 = filter.on_request_body(&mut ctx, &mut chunk1, false).await.unwrap();
        assert!(matches!(a1, FilterAction::Continue));
        let mut chunk2 = Some(Bytes::from_static(suffix));
        let a2 = filter.on_request_body(&mut ctx, &mut chunk2, true).await.unwrap();
        assert!(matches!(a2, FilterAction::Continue));

        assert_eq!(
            registry.get("gpt-4"),
            ModelMetrics {
                in_flight: 1,
                reserved_tokens: 5
            },
            "model and cap arriving after 20 KiB of preceding content must still be attributed"
        );
    }
}
