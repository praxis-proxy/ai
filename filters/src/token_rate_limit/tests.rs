// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

#![allow(clippy::indexing_slicing, reason = "test fixtures are structurally defined")]

use super::*;

#[tokio::test]
async fn valkey_worker_reconciles_body_usage_and_preserves_idempotency() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        return;
    };
    let mut value = config();
    value["backend"] = serde_yaml::from_str(&format!(
        "kind: valkey\nurl: {url}\nnamespace: praxis-test-{}\n",
        std::process::id()
    ))
    .unwrap();
    set_capacity(&mut value, 14);
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();

    let request = request_with_model("model-a");
    let mut first = crate::test_utils::make_filter_context(&request);
    first.set_metadata("identity.user_id", "valkey-alice");
    assert!(matches!(
        filter.on_request(&mut first).await.unwrap(),
        FilterAction::Continue
    ));
    let reservation_id = first
        .get_metadata(META_RESERVATION_ID)
        .and_then(|value| value.split_once(':'))
        .and_then(|(_, id)| id.parse::<u64>().ok())
        .unwrap();
    let quota_key = first.get_metadata(META_KEY).unwrap().to_owned();
    first.set_metadata(META_TOKEN_TOTAL, "4");
    let mut body = None;
    assert!(matches!(
        filter.on_response_body(&mut first, &mut body, true).unwrap(),
        FilterAction::Continue
    ));

    let mut admitted_after_reconcile = false;
    for _ in 0..20 {
        let mut next = crate::test_utils::make_filter_context(&request);
        next.set_metadata("identity.user_id", "valkey-alice");
        if matches!(filter.on_request(&mut next).await.unwrap(), FilterAction::Continue) {
            admitted_after_reconcile = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(admitted_after_reconcile, "worker settlement did not become visible");
    assert_eq!(
        filter.rules[0]
            .backend
            .reconcile(ReconcileRequest {
                key: quota_key,
                reservation_id,
                actual: Some(4),
                estimate: 10,
                now_ms: filter.now_ms(),
            })
            .await
            .unwrap(),
        BackendSettlement::Noop
    );
}

#[tokio::test]
async fn valkey_is_shared_across_independent_filter_instances() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        return;
    };
    let namespace = format!("praxis-test-{}-shared", std::process::id());
    let mut value = config();
    value["backend"] = serde_yaml::from_str(&format!("kind: valkey\nurl: {url}\nnamespace: {namespace}\n")).unwrap();
    set_capacity(&mut value, 10);
    let first = TokenRateLimitFilter::from_config_inner(&value).unwrap();
    let second = TokenRateLimitFilter::from_config_inner(&value).unwrap();

    let request = request_with_model("model-a");
    let mut first_ctx = crate::test_utils::make_filter_context(&request);
    first_ctx.set_metadata("identity.user_id", "shared-alice");
    let mut second_ctx = crate::test_utils::make_filter_context(&request);
    second_ctx.set_metadata("identity.user_id", "shared-alice");
    let (first_result, second_result) =
        tokio::join!(first.on_request(&mut first_ctx), second.on_request(&mut second_ctx));
    let admitted = [first_result.unwrap(), second_result.unwrap()]
        .into_iter()
        .filter(|result| matches!(result, FilterAction::Continue))
        .count();
    assert_eq!(admitted, 1, "shared Valkey quota admitted concurrent requests twice");
}

#[tokio::test]
async fn valkey_failure_fails_closed_before_route_metadata() {
    let mut value = config();
    value["backend"] =
        serde_yaml::from_str("kind: valkey\nurl: redis://127.0.0.1:6399\nnamespace: praxis-test-unavailable\n")
            .unwrap();
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();
    let request = request_with_model("model-a");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "offline-alice");
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Reject(rejection) if rejection.status == 503));
    assert!(ctx.cluster.is_none());
}

#[tokio::test]
async fn valkey_global_active_bound_and_expiry_are_atomic() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        return;
    };
    let namespace = format!("praxis-test-{}-active-bound", std::process::id());
    let mut value = config();
    value["backend"] = serde_yaml::from_str(&format!("kind: valkey\nurl: {url}\nnamespace: {namespace}\n")).unwrap();
    value["reservationTimeout"] = serde_yaml::Value::String("1s".into());
    value["limits"]["max_active_reservations"] = serde_yaml::Value::Number(1.into());
    let first = TokenRateLimitFilter::from_config_inner(&value).unwrap();
    let second = TokenRateLimitFilter::from_config_inner(&value).unwrap();

    let request = request_with_model("model-a");
    let mut first_ctx = crate::test_utils::make_filter_context(&request);
    first_ctx.set_metadata("identity.user_id", "active-one");
    let mut second_ctx = crate::test_utils::make_filter_context(&request);
    second_ctx.set_metadata("identity.user_id", "active-two");
    let (first_result, second_result) =
        tokio::join!(first.on_request(&mut first_ctx), second.on_request(&mut second_ctx));
    let admitted = [first_result.unwrap(), second_result.unwrap()]
        .into_iter()
        .filter(|result| matches!(result, FilterAction::Continue))
        .count();
    assert_eq!(admitted, 1);

    tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
    let mut after_expiry = crate::test_utils::make_filter_context(&request);
    after_expiry.set_metadata("identity.user_id", "active-three");
    assert!(matches!(
        first.on_request(&mut after_expiry).await.unwrap(),
        FilterAction::Continue
    ));
}

#[tokio::test]
async fn valkey_missing_usage_is_charged_at_estimate() {
    let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
        return;
    };
    let mut value = config();
    value["backend"] = serde_yaml::from_str(&format!(
        "kind: valkey\nurl: {url}\nnamespace: praxis-test-{}-missing\n",
        std::process::id()
    ))
    .unwrap();
    set_capacity(&mut value, 10);
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();
    let request = request_with_model("model-a");
    let mut first = crate::test_utils::make_filter_context(&request);
    first.set_metadata("identity.user_id", "missing-usage");
    assert!(matches!(
        filter.on_request(&mut first).await.unwrap(),
        FilterAction::Continue
    ));
    let mut body = None;
    assert!(matches!(
        filter.on_response_body(&mut first, &mut body, true).unwrap(),
        FilterAction::Continue
    ));

    let mut denied = false;
    for _ in 0..20 {
        let mut next = crate::test_utils::make_filter_context(&request);
        next.set_metadata("identity.user_id", "missing-usage");
        if matches!(filter.on_request(&mut next).await.unwrap(), FilterAction::Reject(rejection) if rejection.status == 429)
        {
            denied = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(denied, "missing usage was not charged conservatively");
}

#[allow(clippy::indexing_slicing, reason = "test fixtures are structurally defined above")]
fn set_capacity(value: &mut serde_yaml::Value, capacity: u64) {
    value["rules"][0]["token_budgets"][0]["capacity"] = serde_yaml::Value::Number(capacity.into());
}

fn config() -> serde_yaml::Value {
    serde_yaml::from_str(
        "key:\n  principal:\n    source: metadata\n    name: identity.user_id\n    onMissing: reject\n  model:\n    source: header\n    name: x-model\n    onMissing: reject\n    allowedModels: [model-a, model-b]\nreservationTimeout: 2m\nlimits:\n  max_keys: 10\n  max_key_length: 256\n  max_active_reservations: 10\nrules:\n  - name: tenant-default\n    estimation:\n      strategy: fixed\n      tokens: 10\n    token_budgets:\n      - window: 1m\n        capacity: 100\n",
    )
    .unwrap()
}

fn request_with_model(model: &'static str) -> praxis_filter::Request {
    let mut request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    request.headers.insert("x-model", model.parse().unwrap());
    request
}

#[test]
fn parses_fixed_configuration() {
    TokenRateLimitFilter::from_config_inner(&config()).unwrap();
}

#[test]
fn valkey_configuration_requires_a_url() {
    let mut value = config();
    value["backend"] = serde_yaml::from_str("kind: valkey\n").unwrap();

    let error = TokenRateLimitFilter::from_config_inner(&value).unwrap_err();
    assert!(error.to_string().contains("backend.url is required for Valkey"));
}

#[test]
fn valkey_configuration_rejects_an_invalid_url() {
    let mut value = config();
    value["backend"] = serde_yaml::from_str("kind: valkey\nurl: not-a-valkey-url\n").unwrap();

    let error = TokenRateLimitFilter::from_config_inner(&value).unwrap_err();
    assert!(error.to_string().contains("shared quota backend unavailable"));
}

#[test]
fn declares_response_body_access_for_reconciliation() {
    let filter = TokenRateLimitFilter::from_config_inner(&config()).unwrap();
    assert_eq!(filter.response_body_access(), BodyAccess::ReadOnly);
}

#[test]
fn rejects_unknown_configuration_fields() {
    let mut value = config();
    value.as_mapping_mut().expect("fixture must be a mapping").insert(
        serde_yaml::Value::String("unexpected".into()),
        serde_yaml::Value::String("reject".into()),
    );
    assert!(TokenRateLimitFilter::from_config_inner(&value).is_err());
}

#[test]
fn rejects_missing_identity_or_model() {
    let filter = TokenRateLimitFilter::from_config_inner(&config()).unwrap();
    let request = request_with_model("model-a");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    let action = futures::executor::block_on(filter.on_request(&mut ctx)).unwrap();
    assert!(matches!(action, FilterAction::Reject(rejection) if rejection.status == 401));

    let request = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "alice");
    let action = futures::executor::block_on(filter.on_request(&mut ctx)).unwrap();
    assert!(matches!(action, FilterAction::Reject(rejection) if rejection.status == 400));

    let request = request_with_model("invented-model");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "alice");
    let action = futures::executor::block_on(filter.on_request(&mut ctx)).unwrap();
    assert!(matches!(action, FilterAction::Reject(rejection) if rejection.status == 404));
}

#[test]
fn rejects_composite_key_before_backend_allocation() {
    let mut value = config();
    value["limits"]["max_key_length"] = serde_yaml::Value::Number(8.into());
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();
    let request = request_with_model("model-a");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "alice");
    let action = futures::executor::block_on(filter.on_request(&mut ctx)).unwrap();
    assert!(matches!(action, FilterAction::Reject(rejection) if rejection.status == 400));
    assert!(ctx.get_metadata(META_RESERVATION_ID).is_none());
}

#[test]
#[allow(clippy::indexing_slicing, reason = "test fixture has a known rules sequence")]
fn rejects_default_rule_before_specific_rule() {
    let mut value = config();
    let rules = value["rules"].as_sequence_mut().unwrap();
    rules.push(serde_yaml::from_str(
        "name: alice\nmatch:\n  metadata:\n    identity.user_id: alice\nestimation:\n  strategy: fixed\n  tokens: 10\ntoken_budgets:\n  - window: 1m\n    capacity: 10\n",
    )
    .unwrap());
    assert!(TokenRateLimitFilter::from_config_inner(&value).is_err());
}

#[test]
fn principal_and_model_buckets_are_independent() {
    let value: serde_yaml::Value = serde_yaml::from_str(
        "key:\n  principal:\n    source: metadata\n    name: identity.user_id\n    onMissing: reject\n  model:\n    source: header\n    name: x-model\n    onMissing: reject\n    allowedModels: [model-a, model-b]\nreservationTimeout: 2m\nlimits:\n  max_keys: 10\n  max_key_length: 256\n  max_active_reservations: 10\nrules:\n  - name: alice\n    match:\n      metadata:\n        identity.user_id: alice\n    estimation:\n      strategy: fixed\n      tokens: 10\n    token_budgets:\n      - window: 1m\n        capacity: 10\n  - name: bob\n    match:\n      metadata:\n        identity.user_id: bob\n    estimation:\n      strategy: fixed\n      tokens: 10\n    token_budgets:\n      - window: 1m\n        capacity: 10\n",
    )
    .unwrap();
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();

    for (user, model) in [("alice", "model-a"), ("alice", "model-b"), ("bob", "model-a")] {
        let request = request_with_model(model);
        let mut ctx = crate::test_utils::make_filter_context(&request);
        ctx.set_metadata("identity.user_id", user);
        let action = futures::executor::block_on(filter.on_request(&mut ctx)).unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }

    let request = request_with_model("model-a");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "alice");
    let action = futures::executor::block_on(filter.on_request(&mut ctx)).unwrap();
    assert!(matches!(action, FilterAction::Reject(rejection) if rejection.status == 429));
}

#[tokio::test]
async fn admitted_request_reconciles_actual_usage_and_allows_refund() {
    let mut value = config();
    set_capacity(&mut value, 14);
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();

    let request = request_with_model("model-a");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "alice");
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.set_metadata(META_TOKEN_TOTAL, "4");
    let mut response = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response);
    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.response_header = None;

    let request = request_with_model("model-a");
    let mut before_body = crate::test_utils::make_filter_context(&request);
    before_body.set_metadata("identity.user_id", "alice");
    assert!(matches!(
        filter.on_request(&mut before_body).await.unwrap(),
        FilterAction::Reject(rejection) if rejection.status == 429
    ));

    let mut body = None;
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));

    let request = request_with_model("model-a");
    let mut ctx = crate::test_utils::make_filter_context(&request);
    ctx.set_metadata("identity.user_id", "alice");
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
}

#[tokio::test]
async fn exhausted_request_is_429_before_any_route_metadata_is_set() {
    let mut value = config();
    set_capacity(&mut value, 10);
    let filter = TokenRateLimitFilter::from_config_inner(&value).unwrap();
    let request = request_with_model("model-a");
    let mut first = crate::test_utils::make_filter_context(&request);
    first.set_metadata("identity.user_id", "alice");
    assert!(matches!(
        filter.on_request(&mut first).await.unwrap(),
        FilterAction::Continue
    ));

    let mut second = crate::test_utils::make_filter_context(&request);
    second.set_metadata("identity.user_id", "alice");
    let action = filter.on_request(&mut second).await.unwrap();
    let FilterAction::Reject(rejection) = action else {
        panic!("expected 429")
    };
    assert_eq!(rejection.status, 429);
    assert!(rejection.headers.iter().any(|(name, _)| name == "Retry-After"));
    assert!(rejection.headers.iter().any(|(name, _)| name == "X-RateLimit-Limit"));
    assert!(
        rejection
            .headers
            .iter()
            .any(|(name, _)| name == "X-RateLimit-Remaining")
    );
    assert!(rejection.headers.iter().any(|(name, _)| name == "X-RateLimit-Reset"));
    assert!(second.cluster.is_none());
}
