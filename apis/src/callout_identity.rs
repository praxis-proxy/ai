//! Shared per-callout identity staging.
//!
//! Callout adapters (web-search, file-search, file-id resolution, MCP) call
//! [`stage_callout_identity`] to read the caller's trusted [`StateOwner`] and, when a per-user
//! credential slot is configured, the matching secret from [`CalloutCredentials`]. The result is
//! staged into the nested filtered-subrequest so the callout carries the caller's identity and a
//! per-user credential instead of a single shared provider key.

use praxis_filter::{
    DeferredCredential, FilterError, HttpFilterContext, PendingCredentials, RequestExtensions, TraceContext,
};
use secrecy::{ExposeSecret as _, SecretString};

#[cfg(feature = "openai-mcp-tools")]
use crate::CalloutAuthorization;
use crate::{CalloutCredentials, state_owner::StateOwner};

/// Identity + credential resolved for a single callout.
#[derive(Debug)]
pub(crate) struct CalloutIdentity {
    /// The caller's trusted subject/tenant attribution, projected into the child subrequest.
    pub(crate) owner: Option<StateOwner>,
    /// Approved request correlation projected into the child subrequest.
    pub(crate) trace_context: Option<TraceContext>,
    /// The per-user secret to inject, when a credential slot is configured and populated.
    pub(crate) user_credential: Option<SecretString>,
}

impl CalloutIdentity {
    /// Project approved request context into an isolated child request.
    ///
    /// The credential deliberately is not inserted directly: each callout binds
    /// it to its own validated destination as a [`praxis_filter::DeferredCredential`].
    pub(crate) fn project_context_into(&self, child: &mut RequestExtensions) {
        if let Some(owner) = self.owner.as_ref() {
            child.insert(owner.clone());
        }
        if let Some(trace_context) = self.trace_context.as_ref() {
            child.insert(trace_context.clone());
        }
    }

    /// Project trusted attribution and, when present, stage the per-user secret as an
    /// exact-authority deferred credential.
    ///
    /// The credential is never placed on the in-chain request. Praxis Core injects it only
    /// after resolving the destination and drops it on an authority mismatch.
    pub(crate) fn stage_header_credential_into(
        &self,
        child: &mut RequestExtensions,
        authority: &str,
        header: http::HeaderName,
    ) -> Result<(), FilterError> {
        self.project_context_into(child);
        let Some(secret) = self.user_credential.as_ref() else {
            return Ok(());
        };
        let mut pending = PendingCredentials::new();
        pending.push(DeferredCredential::new(authority, header, secret.expose_secret())?);
        child.insert(pending);
        Ok(())
    }
}

/// Compute the exact `host:port` authority used by Praxis Core's deferred-credential
/// matcher from an operator-configured HTTP(S) URL.
///
/// Callers invoke this during configuration and retain the result, preventing a malformed
/// authority from degrading at request time into a silently unauthenticated callout.
#[cfg(any(test, feature = "openai-responses"))]
pub(crate) fn credential_authority(filter_name: &str, raw_url: &str) -> Result<String, FilterError> {
    let url = url::Url::parse(raw_url).map_err(|error| -> FilterError {
        format!("{filter_name}: target URL is not a valid URL for credential binding: {error}").into()
    })?;
    let host = url.host_str().ok_or_else(|| -> FilterError {
        format!("{filter_name}: target URL has no host for credential binding").into()
    })?;
    let port = url.port_or_known_default().ok_or_else(|| -> FilterError {
        format!(
            "{filter_name}: target URL scheme `{}` has no known default port for credential binding",
            url.scheme()
        )
        .into()
    })?;
    let authority = if host.starts_with('[') || !host.contains(':') {
        format!("{host}:{port}")
    } else {
        format!("[{host}]:{port}")
    };
    // Validate against the same parser used by the executor's credential matcher.
    http::uri::Authority::try_from(authority.as_str()).map_err(|error| -> FilterError {
        format!("{filter_name}: invalid credential authority `{authority}`: {error}").into()
    })?;
    Ok(authority)
}

/// A required callout-identity component was missing (a security-context failure).
#[derive(Debug)]
pub(crate) enum CalloutContextMissing {
    /// The configured credential slot had no populated (non-empty) value in [`CalloutCredentials`].
    Credential {
        /// The configured slot id that resolved to no usable secret.
        slot: String,
    },
}

/// Required configured-connector context missing before MCP dispatch.
#[derive(Debug)]
#[cfg(feature = "openai-mcp-tools")]
pub(crate) enum McpCalloutContextMissing {
    /// Trusted owner attribution is absent.
    Owner,
    /// Per-user bearer slot is absent.
    Credential,
    /// Opaque authorization-assertion slot is absent.
    Authorization,
}

/// Trusted context staged for one configured MCP connector callout.
///
/// Secrets remain request-scoped and are exposed only while the RMCP transport
/// constructs destination-bound headers. This value is never serialized into
/// the MCP tool map or deferred connector state.
#[derive(Clone)]
#[cfg(feature = "openai-mcp-tools")]
pub(crate) struct McpCalloutIdentity {
    /// Trusted owner projected into the filtered child request.
    pub(crate) owner: StateOwner,
    /// Optional per-user bearer overriding the connector entry's static token.
    pub(crate) user_credential: Option<SecretString>,
    /// Optional opaque assertion injected as fixed `x-mcp-authorized`.
    pub(crate) authorization: Option<SecretString>,
}

#[cfg(feature = "openai-mcp-tools")]
impl std::fmt::Debug for McpCalloutIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpCalloutIdentity")
            .field("owner", &self.owner)
            .field("user_credential", &self.user_credential.as_ref().map(|_| "[REDACTED]"))
            .field("authorization", &self.authorization.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

/// Stage the context required by configured MCP connectors.
///
/// Existing static-only connector configurations pass both slots as `None`.
/// When trusted owner attribution is available it is still projected into the
/// connector subrequest; when no owner is available, the prior context-free
/// behavior is preserved. Configuring either request-scoped slot makes trusted
/// owner attribution mandatory and makes that slot fail closed.
#[cfg(feature = "openai-mcp-tools")]
#[expect(
    clippy::too_many_lines,
    reason = "owner plus two optional fail-closed slots are staged together"
)]
pub(crate) fn stage_mcp_callout_identity(
    ctx: &HttpFilterContext<'_>,
    credential_slot: Option<&str>,
    authorization_slot: Option<&str>,
) -> Result<Option<McpCalloutIdentity>, McpCalloutContextMissing> {
    let owner = ctx.extensions.get::<StateOwner>().cloned();
    if credential_slot.is_none() && authorization_slot.is_none() {
        return Ok(owner.map(|owner| McpCalloutIdentity {
            owner,
            user_credential: None,
            authorization: None,
        }));
    }
    let owner = owner.ok_or(McpCalloutContextMissing::Owner)?;
    let user_credential = match credential_slot {
        Some(slot) => Some(
            ctx.extensions
                .get::<CalloutCredentials>()
                .and_then(|credentials| credentials.get(slot))
                .filter(|secret| !secret.expose_secret().is_empty())
                .cloned()
                .ok_or(McpCalloutContextMissing::Credential)?,
        ),
        None => None,
    };
    let authorization = match authorization_slot {
        Some(slot) => Some(
            ctx.extensions
                .get::<CalloutAuthorization>()
                .and_then(|assertion| assertion.get(slot))
                .filter(|secret| !secret.expose_secret().is_empty())
                .cloned()
                .ok_or(McpCalloutContextMissing::Authorization)?,
        ),
        None => None,
    };
    Ok(Some(McpCalloutIdentity {
        owner,
        user_credential,
        authorization,
    }))
}

/// Resolve the caller's owner and (optionally) a per-user credential for a callout.
///
/// `slot` is the configured credential slot id, or `None` when the adapter falls back to the
/// shared provider key. Performs one bounded [`StateOwner`] clone and at most one
/// [`SecretString`] clone — the sanctioned boundary copies for crossing into the child subrequest.
///
/// # Errors
///
/// Returns [`CalloutContextMissing::Credential`] when `slot` is `Some` but the slot resolves to no
/// populated (non-empty) secret; the caller converts this into the fail-closed security terminal.
pub(crate) fn stage_callout_identity(
    ctx: &HttpFilterContext<'_>,
    slot: Option<&str>,
) -> Result<CalloutIdentity, CalloutContextMissing> {
    let owner = ctx.extensions.get::<StateOwner>().cloned();
    let trace_context = ctx.extensions.get::<TraceContext>().cloned();

    let user_credential = match slot {
        None => None,
        Some(name) => {
            let resolved = ctx
                .extensions
                .get::<CalloutCredentials>()
                .and_then(|creds| creds.get(name))
                .filter(|secret| !secret.expose_secret().is_empty())
                .cloned();
            match resolved {
                Some(secret) => Some(secret),
                None => return Err(CalloutContextMissing::Credential { slot: name.to_owned() }),
            }
        },
    };

    Ok(CalloutIdentity {
        owner,
        trace_context,
        user_credential,
    })
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::panic, clippy::unwrap_used, reason = "tests")]
mod tests {
    use http::{HeaderValue, Method};
    use praxis_filter::builtins::TraceContextFilter;
    use secrecy::SecretString;

    use super::*;
    use crate::{
        CalloutCredentials, StateOwner,
        test_utils::{make_filter_context, make_request},
    };

    #[test]
    fn no_slot_configured_yields_no_owner_and_no_credential() {
        let req = make_request(Method::POST, "/v1/responses");
        let ctx = make_filter_context(&req);

        let id = stage_callout_identity(&ctx, None).expect("no slot must succeed");
        assert!(id.owner.is_none(), "no state_owner ran, so owner is absent");
        assert!(id.trace_context.is_none(), "no trace_context ran, so tracing is absent");
        assert!(id.user_credential.is_none(), "no slot configured, so no credential");
    }

    #[test]
    fn owner_is_captured_when_present() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        ctx.extensions
            .insert(StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").expect("valid owner"));

        let id = stage_callout_identity(&ctx, None).expect("ok");
        let owner = id.owner.expect("owner captured from ctx");
        assert_eq!(owner.tenant_id(), "tenant-a");
        assert_eq!(owner.subject(), "subject-a");
    }

    #[tokio::test]
    async fn trace_context_is_the_only_additional_parent_extension_projected() {
        #[derive(Debug)]
        struct UnrelatedParentState;

        let mut req = make_request(Method::POST, "/v1/responses");
        req.headers
            .insert("x-request-id", HeaderValue::from_static("request-parent"));
        req.headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        let mut ctx = make_filter_context(&req);
        ctx.extensions.insert(UnrelatedParentState);
        let filter = TraceContextFilter::from_config(&serde_yaml::from_str("{}").expect("valid config"))
            .expect("trace_context filter");
        let _action = filter.on_request(&mut ctx).await.expect("trace context established");

        let parent = ctx
            .extensions
            .get::<TraceContext>()
            .expect("trace_context filter establishes typed context")
            .clone();
        let identity = stage_callout_identity(&ctx, None).expect("callout context stages");
        let mut child = RequestExtensions::default();
        identity.project_context_into(&mut child);

        let projected = child.get::<TraceContext>().expect("trace context projected");
        assert_eq!(projected.request_id(), parent.request_id());
        assert_eq!(projected.trace_id(), parent.trace_id());
        assert_eq!(projected.flags(), parent.flags());
        assert!(
            child.get::<UnrelatedParentState>().is_none(),
            "ambient parent extensions must remain isolated"
        );
    }

    #[test]
    fn present_slot_is_resolved() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("tok-xyz"));
        ctx.extensions.insert(creds);

        let id = stage_callout_identity(&ctx, Some("brave")).expect("populated slot resolves");
        assert_eq!(
            id.user_credential.expect("credential present").expose_secret(),
            "tok-xyz"
        );
    }

    #[test]
    fn required_slot_missing_is_error() {
        let req = make_request(Method::POST, "/v1/responses");
        let ctx = make_filter_context(&req);

        match stage_callout_identity(&ctx, Some("brave")) {
            Err(CalloutContextMissing::Credential { slot }) => assert_eq!(slot, "brave"),
            other => panic!("expected missing-credential error, got {other:?}"),
        }
    }

    #[test]
    fn present_slot_missing_when_empty() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        let mut creds = CalloutCredentials::new();
        creds.insert("brave".to_owned(), SecretString::from("")); // empty -> must fail closed
        ctx.extensions.insert(creds);

        match stage_callout_identity(&ctx, Some("brave")) {
            Err(CalloutContextMissing::Credential { slot }) => assert_eq!(slot, "brave"),
            other => panic!("empty credential must be rejected, got {other:?}"),
        }
    }

    #[test]
    fn credential_authority_is_exact_and_uses_scheme_default_port() {
        assert_eq!(
            credential_authority("test", "https://ogx.example/v1/files").unwrap(),
            "ogx.example:443"
        );
        assert_eq!(
            credential_authority("test", "http://[::1]:8321/v1/files").unwrap(),
            "[::1]:8321"
        );
        assert_eq!(
            credential_authority("test", "https://münich.example/v1/files").unwrap(),
            "xn--mnich-kva.example:443"
        );
    }

    #[test]
    fn stages_owner_and_exact_authority_credential() {
        let identity = CalloutIdentity {
            owner: Some(StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").unwrap()),
            trace_context: None,
            user_credential: Some(SecretString::from("Bearer user-a")),
        };
        let mut child = RequestExtensions::default();
        identity
            .stage_header_credential_into(&mut child, "ogx.example:443", http::header::AUTHORIZATION)
            .unwrap();

        assert!(child.get::<StateOwner>().is_some());
        assert!(child.get::<PendingCredentials>().is_some());
    }

    #[test]
    #[cfg(feature = "openai-mcp-tools")]
    fn mcp_without_slots_projects_available_owner() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        ctx.extensions
            .insert(StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").expect("valid owner"));

        let id = stage_mcp_callout_identity(&ctx, None, None)
            .expect("owner-only MCP identity must stage")
            .expect("available owner must produce MCP identity");

        assert_eq!(id.owner.tenant_id(), "tenant-a");
        assert_eq!(id.owner.issuer(), "issuer-a");
        assert_eq!(id.owner.subject(), "subject-a");
        assert!(id.user_credential.is_none());
        assert!(id.authorization.is_none());
    }

    #[test]
    #[cfg(feature = "openai-mcp-tools")]
    fn mcp_without_slots_or_owner_preserves_context_free_behavior() {
        let req = make_request(Method::POST, "/v1/responses");
        let ctx = make_filter_context(&req);

        assert!(
            stage_mcp_callout_identity(&ctx, None, None)
                .expect("unconfigured MCP context must not fail")
                .is_none()
        );
    }

    #[test]
    #[cfg(feature = "openai-mcp-tools")]
    fn mcp_configured_slot_requires_owner() {
        let req = make_request(Method::POST, "/v1/responses");
        let ctx = make_filter_context(&req);

        assert!(matches!(
            stage_mcp_callout_identity(&ctx, Some("mcp_gateway"), None),
            Err(McpCalloutContextMissing::Owner)
        ));
    }

    #[test]
    #[cfg(feature = "openai-mcp-tools")]
    fn mcp_configured_credential_must_be_populated() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        ctx.extensions
            .insert(StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").expect("valid owner"));

        assert!(matches!(
            stage_mcp_callout_identity(&ctx, Some("mcp_gateway"), None),
            Err(McpCalloutContextMissing::Credential)
        ));
    }

    #[test]
    #[cfg(feature = "openai-mcp-tools")]
    fn mcp_configured_assertion_must_be_populated() {
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        ctx.extensions
            .insert(StateOwner::from_trusted_parts("tenant-a", "issuer-a", "subject-a").expect("valid owner"));

        assert!(matches!(
            stage_mcp_callout_identity(&ctx, None, Some("mcp_gateway")),
            Err(McpCalloutContextMissing::Authorization)
        ));
    }
}
