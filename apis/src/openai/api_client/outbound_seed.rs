// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Seed filter that projects a pre-resolved callout upstream into a bound
//! outbound filter chain.
//!
//! The [`FilteredSubrequestExecutor`] dials the upstream recorded on its
//! per-callout [`HttpFilterContext::upstream`], which only a filter running
//! inside the outbound chain can set. Configured OpenAI Files API callouts own
//! their destination (`files_api_url`) rather than resolving it from a cluster,
//! so the API client validates and pins the target with
//! [`prepare_url_target`](praxis_core::connectivity::prepare_url_target),
//! stages the resulting [`Upstream`] in the sub-request extensions, and relies
//! on [`CalloutSeedUpstreamFilter`] to move it onto the context before the
//! executor reads it.
//!
//! The seed is installed at **both ends** of the bound chain (see
//! `seed_file_resolve_chain` in `praxis-ai-filters`): a head instance projects
//! the pinned upstream onto the context so the operator's own filters observe
//! the real destination, and a tail instance **reasserts** it after those
//! filters run. The tail guarantees that a well-meaning operator filter (for
//! example an `endpoint_selector`) cannot silently retarget a callout that
//! still carries Files API credentials — the executor always dials the pinned
//! destination the API client validated. To make this reassert possible the
//! filter clones, rather than consumes, the staged upstream, so it remains
//! available to every instance in the chain.
//!
//! [`FilteredSubrequestExecutor`]: praxis_filter::FilteredSubrequestExecutor
//! [`HttpFilterContext::upstream`]: praxis_filter::HttpFilterContext

use async_trait::async_trait;
use praxis_core::connectivity::Upstream;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext};

/// A pre-resolved, validated callout upstream staged into the sub-request
/// extensions by the API client and projected onto the context by
/// [`CalloutSeedUpstreamFilter`].
///
/// Carrying the upstream through the extensions keeps the destination the API
/// client already validated and pinned
/// ([`prepare_url_target`](praxis_core::connectivity::prepare_url_target))
/// authoritative: the seed filter never re-derives it from client-influenced
/// request data. The seed clones rather than consumes it, so it survives for
/// the tail instance to reassert after the operator's filters run.
pub(crate) struct StagedCalloutUpstream {
    /// The pre-resolved upstream the executor should dial.
    pub(crate) upstream: Upstream,
}

/// Internal outbound-chain filter that dials the callout upstream the API
/// client pinned for a configured Files API request.
///
/// Configured OpenAI Files API callouts own their destination
/// (`files_api_url`), so the bound outbound chain has no traffic-management
/// filter to populate the per-callout [`HttpFilterContext::upstream`] the
/// [`FilteredSubrequestExecutor`] reads. The API client validates and pins the
/// target, stages the resulting upstream as a `StagedCalloutUpstream` in the
/// sub-request extensions, and this filter projects it onto the context.
///
/// It is installed at both the head and the tail of the bound chain. The head
/// instance lets the operator's filters observe the real destination; the tail
/// instance reasserts the pinned upstream after they run, so an operator filter
/// cannot silently retarget a credentialed callout. Because it clones rather
/// than consumes `StagedCalloutUpstream`, every instance sees the same staged
/// value and the reassert is idempotent.
///
/// The filter is a no-op when no upstream was staged (the executor then fails
/// closed with its own "did not resolve an upstream" error), so referencing it
/// outside a seeded callout chain is harmless.
///
/// [`FilteredSubrequestExecutor`]: praxis_filter::FilteredSubrequestExecutor
/// [`HttpFilterContext::upstream`]: praxis_filter::HttpFilterContext
pub struct CalloutSeedUpstreamFilter;

impl CalloutSeedUpstreamFilter {
    /// Build the filter.
    ///
    /// The filter is configuration-free; any provided config is ignored. The
    /// [`Result`] and [`serde_yaml::Value`] parameter match the registry
    /// factory signature.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the registry filter-factory signature requires a fallible constructor"
    )]
    pub fn from_config(_config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for CalloutSeedUpstreamFilter {
    fn name(&self) -> &'static str {
        "openai_callout_seed_upstream"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Clone (never consume) the staged upstream so a second, tail instance
        // can reassert it after the operator's own filters have run. Cloning is
        // cheap: `Upstream` is `Arc`-backed. Leaving `StagedCalloutUpstream` in
        // the extensions is inert — nothing else reads it.
        if let Some(staged) = ctx.extensions.get::<StagedCalloutUpstream>() {
            ctx.upstream = Some(staged.upstream.clone());
        }
        Ok(FilterAction::Continue)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use std::sync::Arc;

    use praxis_core::connectivity::{ConnectionOptions, Upstream};
    use praxis_filter::{FilterAction, HttpFilter as _};

    use super::{CalloutSeedUpstreamFilter, StagedCalloutUpstream};

    fn upstream_at(address: &'static str, authority: &'static str) -> Upstream {
        Upstream {
            address: Arc::from(address),
            authority: Some(http::HeaderValue::from_static(authority)),
            connection: Arc::new(ConnectionOptions::default()),
            tls: None,
        }
    }

    fn test_upstream() -> Upstream {
        upstream_at("127.0.0.1:9999", "files-api:9999")
    }

    #[tokio::test]
    async fn seeds_staged_upstream_onto_context() {
        let req = crate::test_utils::make_request(http::Method::GET, "/v1/files/file-abc");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.extensions.insert(StagedCalloutUpstream {
            upstream: test_upstream(),
        });

        let action = CalloutSeedUpstreamFilter.on_request(&mut ctx).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "seeding should continue the chain"
        );
        let upstream = ctx
            .upstream
            .as_ref()
            .expect("upstream should be seeded onto the context");
        assert_eq!(
            &*upstream.address, "127.0.0.1:9999",
            "the staged address should be projected onto the context"
        );
        assert!(
            ctx.extensions.get::<StagedCalloutUpstream>().is_some(),
            "the staged upstream must be cloned, not consumed, so the tail instance can reassert it"
        );
    }

    #[tokio::test]
    async fn tail_reassert_defeats_a_mid_chain_override() {
        // Simulate: head seed -> operator filter retargets the upstream ->
        // tail seed reasserts the pinned destination.
        let req = crate::test_utils::make_request(http::Method::GET, "/v1/files/file-abc");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.extensions.insert(StagedCalloutUpstream {
            upstream: test_upstream(),
        });

        // Head instance projects the pinned upstream.
        let head = CalloutSeedUpstreamFilter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(head, FilterAction::Continue), "head seed should continue");

        // An operator filter overrides the upstream, e.g. an endpoint_selector
        // pointing at an attacker-controlled or wrong destination.
        ctx.upstream = Some(upstream_at("10.0.0.1:1234", "evil:1234"));

        // Tail instance reasserts the pinned upstream, defeating the override.
        let tail = CalloutSeedUpstreamFilter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(tail, FilterAction::Continue), "tail seed should continue");

        let upstream = ctx
            .upstream
            .as_ref()
            .expect("upstream should be present after the tail reassert");
        assert_eq!(
            &*upstream.address, "127.0.0.1:9999",
            "the tail seed must restore the pinned destination after an operator override"
        );
        assert_eq!(
            upstream
                .authority
                .as_ref()
                .map(http::HeaderValue::to_str)
                .transpose()
                .unwrap(),
            Some("files-api:9999"),
            "the reasserted authority must be the pinned one, not the operator's",
        );
    }

    #[tokio::test]
    async fn no_op_without_staged_upstream() {
        let req = crate::test_utils::make_request(http::Method::GET, "/v1/files/file-abc");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = CalloutSeedUpstreamFilter.on_request(&mut ctx).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "a missing upstream is not an error"
        );
        assert!(ctx.upstream.is_none(), "no upstream should be set when none was staged");
    }
}
