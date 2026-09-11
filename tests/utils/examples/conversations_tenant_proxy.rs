// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Test-only Praxis server that resolves SDK bearer tokens to tenant metadata.

use std::path::PathBuf;

use async_trait::async_trait;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection};

/// Metadata key consumed by the OpenAI Conversations handler.
const TENANT_METADATA_KEY: &str = "responses.tenant_id";

/// Fixed identities used by the SDK integration suite.
const TENANT_TOKENS: [(&str, &str); 2] = [("tenant-a-token", "tenant-a"), ("tenant-b-token", "tenant-b")];

/// Resolve deterministic test credentials into trusted request metadata.
struct TestTenantIdentityFilter;

impl TestTenantIdentityFilter {
    /// Build the test filter using the standard custom-filter factory shape.
    ///
    /// # Errors
    ///
    /// This deterministic filter has no configuration and cannot fail to build.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "custom HTTP filter factories must return Result"
    )]
    fn from_config(_config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for TestTenantIdentityFilter {
    fn name(&self) -> &'static str {
        "test_tenant_identity"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let tenant_id = ctx
            .request
            .headers
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .and_then(tenant_for_token);

        let Some(tenant_id) = tenant_id else {
            return Ok(unauthorized());
        };
        ctx.set_metadata(TENANT_METADATA_KEY, tenant_id);
        Ok(FilterAction::Continue)
    }
}

/// Map one fixed bearer token to its test tenant.
fn tenant_for_token(token: &str) -> Option<&'static str> {
    TENANT_TOKENS
        .iter()
        .find_map(|(candidate, tenant)| (*candidate == token).then_some(*tenant))
}

/// Return an OpenAI-shaped authentication error.
fn unauthorized() -> FilterAction {
    const BODY: &str = r#"{"error":{"message":"Invalid authentication credentials","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}"#;
    FilterAction::Reject(
        Rejection::status(401)
            .with_header("content-type", "application/json")
            .with_body(BODY),
    )
}

/// Read the explicit configuration path accepted by the test executable.
fn config_path() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        return Err("usage: conversations_tenant_proxy -c <config>".to_owned());
    };
    if flag != "-c" && flag != "--config" {
        return Err(format!("unexpected argument {flag:?}; expected -c <config>"));
    }
    let Some(path) = args.next() else {
        return Err("missing value for -c".to_owned());
    };
    if args.next().is_some() {
        return Err("unexpected arguments after configuration path".to_owned());
    }
    Ok(path)
}

/// Start Praxis with the test-only identity filter registered.
fn main() {
    let explicit = config_path().unwrap_or_else(|error| praxis_ai::fatal(&error));
    let config = praxis_ai::load_config(Some(&explicit)).unwrap_or_else(|error| praxis_ai::fatal(&error));
    let _tracing_guard = praxis_ai::init_tracing(&config).unwrap_or_else(|error| praxis_ai::fatal(&error));
    let subrequest_client = praxis_ai::create_subrequest_client(&config);
    let mut registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_filter::register_filters!(
        @register registry,
        http "test_tenant_identity" => TestTenantIdentityFilter::from_config
    );
    praxis_ai::run_server_with_registry(config, registry, Some(PathBuf::from(explicit)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed tokens resolve to distinct tenant identities.
    #[test]
    fn known_tokens_resolve_to_tenants() {
        assert_eq!(tenant_for_token("tenant-a-token"), Some("tenant-a"));
        assert_eq!(tenant_for_token("tenant-b-token"), Some("tenant-b"));
        assert_eq!(tenant_for_token("unknown-token"), None);
    }
}
