// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Server-injected request gate for listener-scoped store provisioning.

#![cfg(feature = "store")]

use async_trait::async_trait;
use bytes::Bytes;
use praxis_ai_apis::store::ResponseStoreRegistry;
use praxis_filter::{
    EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, body::BodyAccess,
};

/// Internal filter name prepended by the server to listeners with stores.
pub const FILTER_NAME: &str = "praxis_store_readiness_gate";

/// Reject traffic until every store configured for this listener is registered.
pub struct StoreReadinessGateFilter;

impl StoreReadinessGateFilter {
    /// Build the internal gate, rejecting any accidental configuration fields.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when `config` is not empty.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let _: EmptyFilterConfig = praxis_filter::parse_filter_config(FILTER_NAME, config)?;
        Ok(Box::new(Self))
    }

    /// Return the request action for the listener's current store state.
    fn readiness_action(ctx: &HttpFilterContext<'_>) -> FilterAction {
        if ctx
            .extensions
            .get::<ResponseStoreRegistry>()
            .is_some_and(ResponseStoreRegistry::is_ready)
        {
            return FilterAction::Continue;
        }

        FilterAction::Reject(
            Rejection::status(503)
                .with_header("content-type", "application/json")
                .with_header("retry-after", "1")
                .with_body(r#"{"error":{"message":"Persisted state is still initializing.","type":"server_error"}}"#),
        )
    }
}

#[async_trait]
impl HttpFilter for StoreReadinessGateFilter {
    fn name(&self) -> &'static str {
        FILTER_NAME
    }

    fn request_body_access(&self) -> BodyAccess {
        // StreamBuffer filters execute before on_request. Participate in that
        // pre-read so a downstream store filter cannot observe the empty
        // registry first and turn an initializing request into a 500.
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(Self::readiness_action(ctx))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        Ok(match Self::readiness_action(ctx) {
            FilterAction::Continue => FilterAction::BodyDone,
            action => action,
        })
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    #[tokio::test]
    async fn rejects_until_listener_registry_is_ready() {
        let request = make_request(http::Method::GET, "/v1/conversations/conv_1");
        let registry = ResponseStoreRegistry::new();
        let mut context = make_filter_context(&request);
        context.extensions.insert(registry.clone());
        let gate = StoreReadinessGateFilter;

        assert!(matches!(
            gate.on_request(&mut context).await.expect("pending gate"),
            FilterAction::Reject(_)
        ));

        registry.mark_ready();
        assert!(matches!(
            gate.on_request(&mut context).await.expect("ready gate"),
            FilterAction::Continue
        ));
    }

    #[tokio::test]
    async fn rejects_during_body_pre_read_until_listener_registry_is_ready() {
        let request = make_request(http::Method::POST, "/v1/responses");
        let registry = ResponseStoreRegistry::new();
        let mut context = make_filter_context(&request);
        context.extensions.insert(registry.clone());
        let gate = StoreReadinessGateFilter;
        let mut body = Some(Bytes::from_static(br#"{"model":"test","input":"hello"}"#));

        assert_eq!(gate.request_body_access(), BodyAccess::ReadOnly);
        assert!(matches!(
            gate.on_request_body(&mut context, &mut body, false)
                .await
                .expect("pending body gate"),
            FilterAction::Reject(_)
        ));

        registry.mark_ready();
        assert!(matches!(
            gate.on_request_body(&mut context, &mut body, false)
                .await
                .expect("ready body gate"),
            FilterAction::BodyDone
        ));
    }
}
