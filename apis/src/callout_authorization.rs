//! Trusted MCP authorization assertion captured at the ingress boundary.
//!
//! The assertion is opaque capability material. Praxis does not parse it or
//! treat its presence as an authorization decision; configured MCP Gateway
//! connectors validate it on every protocol exchange.

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderName;
use praxis_filter::{
    BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, TrustedHeaderMutation, parse_filter_config,
};
use secrecy::SecretString;
use serde::Deserialize;

use crate::state_owner::reject_owner;

/// Fixed trusted-boundary assertion header.
pub(crate) const MCP_AUTHORIZED_HEADER: HeaderName = HeaderName::from_static("x-mcp-authorized");
/// Maximum assertion size accepted from the trusted boundary.
const MAX_ASSERTION_BYTES: usize = 4_096;
/// Maximum byte length of the config-static assertion slot.
const MAX_SLOT_ID_BYTES: usize = 128;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// Configuration for one fixed authorization-assertion slot.
struct Config {
    /// Config-static slot selected by MCP callout filters.
    assertion_slot: String,
}

/// Opaque request-scoped authorization assertion.
#[derive(Clone)]
pub struct CalloutAuthorization {
    /// Config-static slot that owns the assertion.
    slot: String,
    #[cfg_attr(
        all(not(test), not(feature = "openai-mcp-tools")),
        expect(dead_code, reason = "opaque assertion is consumed only when MCP tools are enabled")
    )]
    /// Opaque trusted-boundary assertion.
    assertion: SecretString,
}

impl CalloutAuthorization {
    /// Return the assertion only when the configured consumer selects this slot.
    #[cfg(any(test, feature = "openai-mcp-tools"))]
    pub(crate) fn get(&self, slot: &str) -> Option<&SecretString> {
        (self.slot == slot).then_some(&self.assertion)
    }
}

impl std::fmt::Debug for CalloutAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalloutAuthorization")
            .field("slot", &self.slot)
            .field("assertion", &"[REDACTED]")
            .finish()
    }
}

/// Establishes the fixed `x-mcp-authorized` assertion before callout consumers.
///
/// The trusted boundary MUST unconditionally delete and then set
/// `x-mcp-authorized` on every ingress path. This filter cannot distinguish a
/// boundary-minted assertion from a client-spoofed header. Place it once in the
/// outer request chain before `openai_mcp_tool_resolve` and the IRR; it strips
/// the wire header and carries only the secret extension across IRR iterations.
///
/// # YAML
///
/// ```yaml
/// filter: callout_authorization
/// assertion_slot: mcp_gateway
/// ```
#[derive(Debug)]
pub struct CalloutAuthorizationFilter {
    /// Config-static slot populated by the fixed source header.
    assertion_slot: String,
}

impl CalloutAuthorizationFilter {
    /// Parse and validate the assertion slot.
    ///
    /// # Errors
    /// Returns [`FilterError`] for an empty or oversized slot or unknown fields.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: Config = parse_filter_config("callout_authorization", value)?;
        if config.assertion_slot.is_empty() || config.assertion_slot.len() > MAX_SLOT_ID_BYTES {
            return Err(format!("callout_authorization: assertion_slot must be 1..={MAX_SLOT_ID_BYTES} bytes").into());
        }
        Ok(Box::new(Self {
            assertion_slot: config.assertion_slot,
        }))
    }

    /// Capture a singular assertion from the effective header view and queue stripping.
    #[expect(
        clippy::too_many_lines,
        reason = "capture, validation, and lifecycle stripping form one boundary"
    )]
    fn resolve(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) -> FilterAction {
        if ctx.extensions.get::<CalloutAuthorization>().is_none() {
            let headers: Cow<'_, http::HeaderMap> = if body_phase {
                crate::callout_headers::effective_body_callout_headers(ctx, Cow::Borrowed(&ctx.request.headers))
            } else {
                Cow::Borrowed(&ctx.request.headers)
            };
            let mut values = headers.get_all(&MCP_AUTHORIZED_HEADER).iter();
            match (values.next(), values.next()) {
                (Some(_), Some(_)) => {
                    return reject_owner(
                        400,
                        "duplicate_callout_authorization",
                        "x-mcp-authorized must appear exactly once",
                    );
                },
                (Some(value), None) if !value.as_bytes().is_empty() => {
                    if value.as_bytes().len() > MAX_ASSERTION_BYTES {
                        return reject_owner(
                            400,
                            "callout_authorization_too_large",
                            "x-mcp-authorized exceeds the 4096-byte limit",
                        );
                    }
                    if let Ok(value) = value.to_str() {
                        ctx.extensions.insert(CalloutAuthorization {
                            slot: self.assertion_slot.clone(),
                            assertion: SecretString::from(value.to_owned()),
                        });
                    }
                },
                _ => {},
            }
        }
        ctx.request_headers_to_remove.push(MCP_AUTHORIZED_HEADER);
        if body_phase && !ctx.pre_read_mutations.is_empty() {
            ctx.pre_read_mutations
                .push(TrustedHeaderMutation::Remove(MCP_AUTHORIZED_HEADER));
        }
        FilterAction::Continue
    }
}

#[async_trait]
impl HttpFilter for CalloutAuthorizationFilter {
    fn name(&self) -> &'static str {
        "callout_authorization"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(self.resolve(ctx, false))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        Ok(match self.resolve(ctx, true) {
            FilterAction::Continue => FilterAction::BodyDone,
            action => action,
        })
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod tests {
    use http::{HeaderValue, Method};
    use secrecy::ExposeSecret as _;

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    fn filter() -> Box<dyn HttpFilter> {
        CalloutAuthorizationFilter::from_config(&serde_yaml::from_str("assertion_slot: mcp_gateway").unwrap()).unwrap()
    }

    #[tokio::test]
    async fn captures_and_strips_one_assertion() {
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert(MCP_AUTHORIZED_HEADER, HeaderValue::from_static("signed-assertion"));
        let mut ctx = make_filter_context(&request);
        assert!(matches!(
            filter().on_request(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        let staged = ctx.extensions.get::<CalloutAuthorization>().expect("assertion staged");
        assert_eq!(
            staged.get("mcp_gateway").expect("slot").expose_secret(),
            "signed-assertion"
        );
        assert!(ctx.request_headers_to_remove.contains(&MCP_AUTHORIZED_HEADER));
    }

    #[tokio::test]
    async fn duplicate_assertion_is_rejected() {
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .append(MCP_AUTHORIZED_HEADER, HeaderValue::from_static("one"));
        request
            .headers
            .append(MCP_AUTHORIZED_HEADER, HeaderValue::from_static("two"));
        let mut ctx = make_filter_context(&request);
        assert!(matches!(
            filter().on_request(&mut ctx).await.unwrap(),
            FilterAction::Reject(_)
        ));
    }

    #[tokio::test]
    async fn missing_assertion_is_optional_and_still_stripped() {
        let request = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&request);

        assert!(matches!(
            filter().on_request(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(ctx.extensions.get::<CalloutAuthorization>().is_none());
        assert!(ctx.request_headers_to_remove.contains(&MCP_AUTHORIZED_HEADER));
    }

    #[tokio::test]
    async fn empty_assertion_is_not_staged_and_is_stripped() {
        let mut request = make_request(Method::POST, "/v1/responses");
        request
            .headers
            .insert(MCP_AUTHORIZED_HEADER, HeaderValue::from_static(""));
        let mut ctx = make_filter_context(&request);

        assert!(matches!(
            filter().on_request(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(ctx.extensions.get::<CalloutAuthorization>().is_none());
        assert!(ctx.request_headers_to_remove.contains(&MCP_AUTHORIZED_HEADER));
    }

    #[tokio::test]
    async fn oversized_assertion_is_rejected() {
        let mut request = make_request(Method::POST, "/v1/responses");
        request.headers.insert(
            MCP_AUTHORIZED_HEADER,
            HeaderValue::from_bytes(&vec![b'a'; MAX_ASSERTION_BYTES + 1]).unwrap(),
        );
        let mut ctx = make_filter_context(&request);

        assert!(matches!(
            filter().on_request(&mut ctx).await.unwrap(),
            FilterAction::Reject(_)
        ));
    }

    #[test]
    fn debug_redacts_assertion() {
        let value = CalloutAuthorization {
            slot: "mcp_gateway".to_owned(),
            assertion: SecretString::from("never-print-me"),
        };
        let rendered = format!("{value:?}");
        assert!(!rendered.contains("never-print-me"));
    }
}
