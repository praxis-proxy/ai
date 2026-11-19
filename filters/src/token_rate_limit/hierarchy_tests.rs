// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Hierarchy holds across one exchange: settled against actual usage at the
//! end of the response, released when the rule denies or fails open.
//!
//! A real user level resolves an `AuthenticatedIdentity`, which only praxis
//! can construct, so these filters compile from config as `from_config`
//! does, then key the user level by the `x-user-id` header instead.

use std::{sync::Arc, time::Instant};

use http::HeaderName;
use praxis_ai_apis::RecordAiMetadata as _;
use praxis_filter::{FilterAction, HttpFilter as _, HttpFilterContext, Request};

use super::{
    CompiledHierarchyIdentity, MAX_KEYS, TokenRateLimitFilter, TokenWeights,
    backend::{
        BackendError, BackendReserve, BackendSettlement, BackendSnapshot, ReconcileRequest, ReserveRequest,
        TokenRateLimitStateBackend,
    },
    build_backend_resource, compile_hierarchy, compile_rule,
    config::{BackendOnFailure, TokenRateLimitConfig},
    keys::compile_key_spec,
};
use crate::token_usage::{META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_TOTAL};

/// Capacity of the user level, the hierarchy level these tests fill.
const USER_CAPACITY: u64 = 1000;

/// Every request reserves this many tokens at each level and on the rule.
const ESTIMATE: u64 = 400;

/// One memory-backed filter: org and team levels with ample room, a user
/// level of [`USER_CAPACITY`], and one rule with the given body (indented
/// YAML lines after `algorithm`).
fn filter(rule_body: &str) -> TokenRateLimitFilter {
    let yaml = format!(
        "backend:\n\
         \x20 kind: memory\n\
         hierarchy:\n\
         \x20 - level: org\n\
         \x20   identity_header: x-org-id\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100000\n\
         \x20 - level: team\n\
         \x20   identity_header: x-team-id\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: 100000\n\
         \x20 - level: user\n\
         \x20   identity: authenticated_subject\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   capacity: {USER_CAPACITY}\n\
         rules:\n\
         \x20 - name: default\n\
         \x20   algorithm: sliding_window\n\
         \x20   window: 1h\n\
         \x20   reserved_tokens: {ESTIMATE}\n\
         {rule_body}"
    );
    let cfg: TokenRateLimitConfig = serde_yaml::from_str(&yaml).unwrap();
    let backend = build_backend_resource(&cfg.backend).unwrap();
    let key_spec = compile_key_spec(cfg.key).unwrap();
    let fingerprint = key_spec.config_fingerprint();
    let rules = cfg
        .rules
        .into_iter()
        .map(|rule| compile_rule(rule, &backend, TokenWeights::UNITY, MAX_KEYS, &fingerprint).unwrap())
        .collect();
    let mut hierarchy = compile_hierarchy(cfg.hierarchy.as_deref().unwrap(), &backend, MAX_KEYS, &fingerprint).unwrap();
    let user = hierarchy.last_mut().unwrap();
    user.identity = CompiledHierarchyIdentity::Header(HeaderName::from_static("x-user-id"));
    TokenRateLimitFilter {
        rules,
        needs_body: false,
        key_spec,
        epoch: Instant::now(),
        valkey_clock: false,
        hierarchy: Some(hierarchy),
        on_failure: cfg.backend.on_failure,
    }
}

/// A request from alice, in one org and team.
fn request() -> Request {
    request_from("alice")
}

/// A request from `user`, in the same org and team as [`request`].
fn request_from(user: &'static str) -> Request {
    let mut req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
    for (name, value) in [("x-org-id", "acme"), ("x-team-id", "eng"), ("x-user-id", user)] {
        req.headers.insert(name, http::HeaderValue::from_static(value));
    }
    req
}

/// A filter context with a filter ID, as in a pipeline.
fn context(req: &Request) -> HttpFilterContext<'_> {
    let mut ctx = crate::test_utils::make_filter_context(req);
    ctx.current_filter_id = Some(0);
    ctx
}

/// Admit one request in a fresh context, leaving its holds unsettled.
async fn admit(filter: &TokenRateLimitFilter, req: &Request) -> FilterAction {
    filter.on_request(&mut context(req)).await.unwrap()
}

/// Finish an exchange whose response reported `usage`, as `token.*`
/// metadata keys and values.
fn complete(filter: &TokenRateLimitFilter, ctx: &mut HttpFilterContext<'_>, usage: &[(&str, &str)]) {
    for &(key, tokens) in usage {
        ctx.record_ai_metadata(key, tokens);
    }
    let mut body = None;
    drop(filter.on_response_body(ctx, &mut body, true).unwrap());
}

/// The token limit a 429 reports, which names the budget that denied it.
fn denied_by_limit(action: &FilterAction) -> Option<String> {
    let FilterAction::Reject(rejection) = action else {
        return None;
    };
    assert_eq!(rejection.status, 429, "expected a budget denial");
    rejection
        .headers
        .iter()
        .find(|(name, _)| name == "X-RateLimit-Limit-Tokens")
        .map(|(_, value)| value.clone())
}

#[tokio::test]
async fn hierarchy_holds_settle_against_actual_usage_at_end_of_stream() {
    let filter = filter("    capacity: 100000\n");
    let req = request();

    let mut ctx = context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    complete(&filter, &mut ctx, &[(META_TOKEN_TOTAL, "100")]);

    // Settled at 100: 100 + 400 + 400 fits the user's 1000; a third 400 does not.
    assert!(matches!(admit(&filter, &req).await, FilterAction::Continue));
    assert!(
        matches!(admit(&filter, &req).await, FilterAction::Continue),
        "the user hold settled at actual usage, not at its 400 estimate"
    );
    assert_eq!(
        denied_by_limit(&admit(&filter, &req).await).as_deref(),
        Some("1000"),
        "the user level enforces its budget"
    );
}

#[tokio::test]
async fn rule_denial_releases_hierarchy_holds() {
    let filter = filter("    capacity: 400\n");
    let req = request();
    assert!(matches!(admit(&filter, &req).await, FilterAction::Continue));

    // The rule is full: each later request clears the hierarchy, then the
    // rule denies it and its hierarchy holds are released.
    for attempt in 1..=2 {
        assert_eq!(
            denied_by_limit(&admit(&filter, &req).await).as_deref(),
            Some("400"),
            "attempt {attempt}: the rule denies, not the user level (holds released)"
        );
    }
}

#[tokio::test]
async fn rule_fail_open_releases_hierarchy_holds() {
    let mut filter = filter("    capacity: 100000\n");
    filter.rules.first_mut().unwrap().backend = Arc::new(UnavailableBackend);
    filter.on_failure = BackendOnFailure::Open;
    let req = request();

    // Each fail-open admission forwards without a rule reservation and
    // releases its hierarchy holds: three 400s would not fit the user's 1000.
    for attempt in 1..=3 {
        assert!(
            matches!(admit(&filter, &req).await, FilterAction::Continue),
            "attempt {attempt}: forwarded, with no hierarchy hold left behind"
        );
    }
}

#[tokio::test]
async fn soft_forward_keeps_hierarchy_holds_and_settles_them_by_the_rule() {
    let filter = filter(
        "    capacity: 400\n\
         \x20   weights:\n\
         \x20     output: 13.0\n\
         \x20   enforcement: soft\n\
         \x20   over_quota:\n\
         \x20     headers:\n\
         \x20       X-Over-Quota: \"true\"\n",
    );
    // Another user fills the rule, leaving alice's user level empty.
    assert!(matches!(
        admit(&filter, &request_from("bob")).await,
        FilterAction::Continue
    ));

    // The rule is full and soft: alice is forwarded without a rule
    // reservation, but her hierarchy holds stay and settle at actual usage,
    // priced by the rule's weights: 50 input + 50 output x 13 = 700.
    let req = request();
    let mut ctx = context(&req);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    complete(
        &filter,
        &mut ctx,
        &[
            (META_TOKEN_INPUT, "50"),
            (META_TOKEN_OUTPUT, "50"),
            (META_TOKEN_TOTAL, "100"),
        ],
    );

    // 700 + 400 exceeds alice's 1000. Settled at the raw total (100) or left
    // at the 400 estimate, her next 400 would fit.
    assert_eq!(
        denied_by_limit(&admit(&filter, &req).await).as_deref(),
        Some("1000"),
        "the soft-forwarded hold settled at its weighted cost, not the raw total or the estimate"
    );
}

/// A rule backend whose reservations fail before dispatch, as when a shared
/// backend is unreachable.
struct UnavailableBackend;

#[async_trait::async_trait]
impl TokenRateLimitStateBackend for UnavailableBackend {
    async fn reserve(&self, _request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        Err(BackendError::Unavailable("unreachable (test)".to_owned()))
    }

    async fn reconcile(&self, _request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        panic!("a fail-open admission holds no rule reservation to settle")
    }

    fn enqueue_reconcile(&self, _request: ReconcileRequest) -> Result<(), BackendError> {
        panic!("a fail-open admission holds no rule reservation to settle")
    }

    fn limit(&self) -> u64 {
        ESTIMATE
    }

    fn snapshot(&self) -> BackendSnapshot {
        BackendSnapshot {
            budget_remaining: 0,
            active_reservations: 0,
            active_keys: 0,
        }
    }

    fn backend_name(&self) -> &'static str {
        "test"
    }

    fn algorithm_name(&self) -> &'static str {
        "test"
    }
}
