// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Hermetic settlement coverage through the real native client and public backend.

use std::time::Duration;

use super::{
    super::{
        ACCOUNTING_GENERATION_VALUE, ValkeyConnection, accounting_generation_key,
        connection::ValkeyTimeouts,
        test_resp::{Reply, ScriptedRespServer, Step, array, bulk},
    },
    ValkeyTokenBucketBackend, ValkeyTokenBucketConfig,
};
use crate::token_rate_limit::{
    AccountingPolicy, CompiledEstimation,
    backend::{BackendError, BackendSettlement, ReconcileRequest, TokenRateLimitStateBackend as _},
    weights::TokenWeights,
};

const NOW: u64 = 2_000;
const TTL: u64 = 101_000;

fn request(actual: Option<u64>) -> ReconcileRequest {
    ReconcileRequest {
        key: "alice".into(),
        reservation_id: 7,
        actual,
        now_ms: NOW,
    }
}

async fn setup(
    script: impl FnOnce(&ValkeyTokenBucketBackend) -> Vec<Step>,
    operation: Duration,
) -> (ValkeyTokenBucketBackend, ScriptedRespServer) {
    let mut backend = ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
        valkey: ValkeyConnection::new("redis://127.0.0.1:1".into()).unwrap(),
        namespace: "unit:settlement".into(),
        rule: "default".into(),
        capacity: 100,
        refill_rate: 1.0,
        reservation_timeout_ms: 1_000,
        max_keys: 8,
        max_active_reservations: 8,
        accounting: AccountingPolicy {
            estimation: CompiledEstimation::Fixed { estimate: 50 },
            weights: TokenWeights::UNITY,
            key_fingerprint: "unit-key".into(),
        },
    })
    .unwrap();
    let server = ScriptedRespServer::start(script(&backend)).await;
    backend.valkey = ValkeyConnection::standalone_with_timeouts_and_name(
        server.url(),
        ValkeyTimeouts {
            connect: Duration::from_secs(1),
            operation,
            discovery: Duration::from_secs(1),
        },
        "shared_kv",
    )
    .unwrap();
    (backend, server)
}

fn marker_reads(backend: &ValkeyTokenBucketBackend) -> Vec<Step> {
    vec![
        Step::reply(
            redis::cmd("GET").arg(accounting_generation_key(&backend.namespace)),
            bulk(Some(ACCOUNTING_GENERATION_VALUE)),
        ),
        Step::reply(
            redis::cmd("GET").arg(backend.accounting_config_key()),
            bulk(Some(&backend.config_fingerprint)),
        ),
    ]
}

fn reservation_reads(backend: &ValkeyTokenBucketBackend, value: Option<&str>, tokens: &str) -> Vec<Step> {
    let bucket = backend.bucket_key(&backend.key_id("alice"));
    let reservation = format!("{bucket}:r:7");
    vec![
        Step::reply(redis::cmd("WATCH").arg(&bucket).arg(&reservation), b"+OK\r\n"),
        Step::reply(redis::cmd("GET").arg(&reservation), bulk(value)),
        Step::reply(
            redis::cmd("HMGET").arg(&bucket).arg("tokens").arg("last_refill_ms"),
            array(&[tokens, "1000"]),
        ),
    ]
}

#[expect(
    clippy::too_many_lines,
    reason = "one transcript spells out atomic settlement and TTL commands"
)]
fn settlement_writes(backend: &ValkeyTokenBucketBackend, balance: f64, exec: Reply) -> Vec<Step> {
    let id = backend.key_id("alice");
    let bucket = backend.bucket_key(&id);
    let keys = backend.keys_key();
    vec![
        Step::reply(&redis::cmd("MULTI"), b"+OK\r\n"),
        Step::reply(
            redis::cmd("HSET")
                .arg(&bucket)
                .arg("tokens")
                .arg(balance)
                .arg("last_refill_ms")
                .arg(NOW),
            b"+QUEUED\r\n",
        ),
        Step::reply(redis::cmd("PEXPIRE").arg(&bucket).arg(TTL).arg("NX"), b"+QUEUED\r\n"),
        Step::reply(redis::cmd("PEXPIRE").arg(&bucket).arg(TTL).arg("GT"), b"+QUEUED\r\n"),
        Step::reply(
            redis::cmd("ZADD").arg(&keys).arg("GT").arg(NOW + TTL).arg(&id),
            b"+QUEUED\r\n",
        ),
        Step::reply(redis::cmd("PEXPIRE").arg(&keys).arg(TTL).arg("NX"), b"+QUEUED\r\n"),
        Step::reply(redis::cmd("PEXPIRE").arg(&keys).arg(TTL).arg("GT"), b"+QUEUED\r\n"),
        Step::reply(redis::cmd("DEL").arg(format!("{bucket}:r:7")), b"+QUEUED\r\n"),
        Step::reply(
            redis::cmd("ZREM")
                .arg(backend.active_index_key())
                .arg(format!("{id}|7")),
            b"+QUEUED\r\n",
        ),
        Step::new(&redis::cmd("EXEC"), exec),
    ]
}

fn committed() -> Reply {
    Reply::Bytes(b"*8\r\n:1\r\n:1\r\n:1\r\n:1\r\n:1\r\n:1\r\n:1\r\n:1\r\n".to_vec())
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "one fixture verifies the original settlement and its duplicate"
)]
async fn hermetic_settlement_refunds_once_and_duplicate_is_a_noop() {
    let (backend, server) = setup(
        |backend| {
            let mut steps = marker_reads(backend);
            steps.extend(reservation_reads(backend, Some("50|1000"), "20"));
            steps.extend(settlement_writes(backend, 51.0, committed()));
            steps.extend(marker_reads(backend));
            steps.extend(reservation_reads(backend, None, "51"));
            steps.push(Step::reply(&redis::cmd("UNWATCH"), b"+OK\r\n"));
            steps
        },
        Duration::from_secs(1),
    )
    .await;
    assert!(
        matches!(
            backend.reconcile(request(Some(20))).await,
            Ok(BackendSettlement::Applied {
                actual: 20,
                refund: 30,
                overage: 0
            })
        ),
        "unused estimate must refund exactly 30 tokens after refill"
    );
    assert!(
        matches!(backend.reconcile(request(Some(20))).await, Ok(BackendSettlement::Noop)),
        "a deleted reservation must not refund twice"
    );
    assert_eq!(
        server.count("EXEC"),
        1,
        "the duplicate must not dispatch another settlement"
    );
    server.finish().await;
}

#[tokio::test]
async fn hermetic_settlement_bounds_overage_refund_and_missing_actual() {
    for (actual, stored, balance, refund, overage) in [
        (Some(80), "20", 0.0, 0, 30),
        (Some(0), "99", 100.0, 50, 0),
        (None, "20", 21.0, 0, 0),
    ] {
        let (backend, server) = setup(
            |backend| {
                let mut steps = marker_reads(backend);
                steps.extend(reservation_reads(backend, Some("50|1000"), stored));
                steps.extend(settlement_writes(backend, balance, committed()));
                steps
            },
            Duration::from_secs(1),
        )
        .await;
        let result = backend.reconcile(request(actual)).await.unwrap();
        assert!(
            matches!(result, BackendSettlement::Applied { actual: applied, refund: credited, overage: debited }
            if applied == actual.unwrap_or(50) && credited == refund && debited == overage),
            "settlement must preserve estimate fallback, refund ceiling, and overage floor"
        );
        server.finish().await;
    }
}

#[tokio::test]
async fn hermetic_settlement_retries_only_a_confirmed_exec_abort() {
    let (backend, server) = setup(
        |backend| {
            let mut steps = marker_reads(backend);
            for exec in [Reply::Bytes(b"*-1\r\n".to_vec()), committed()] {
                steps.extend(reservation_reads(backend, Some("50|1000"), "20"));
                steps.extend(settlement_writes(backend, 51.0, exec));
            }
            steps
        },
        Duration::from_secs(1),
    )
    .await;
    assert!(
        matches!(
            backend.reconcile(request(Some(20))).await,
            Ok(BackendSettlement::Applied { refund: 30, .. })
        ),
        "confirmed abort must retry and settle"
    );
    assert_eq!(
        server.count("WATCH"),
        2,
        "an aborted EXEC must re-read the reservation before retry"
    );
    assert_eq!(
        server.count("EXEC"),
        2,
        "only the known non-applied attempt may be retried"
    );
    server.finish().await;
}

#[tokio::test]
async fn hermetic_settlement_read_faults_remain_retryable_without_mutation() {
    for fault in [Reply::Disconnect, Reply::Stall] {
        let (backend, server) = setup(
            |backend| {
                let mut steps = marker_reads(backend);
                let bucket = backend.bucket_key(&backend.key_id("alice"));
                steps.push(Step::reply(
                    redis::cmd("WATCH").arg(&bucket).arg(format!("{bucket}:r:7")),
                    b"+OK\r\n",
                ));
                steps.push(Step::new(redis::cmd("GET").arg(format!("{bucket}:r:7")), fault));
                steps
            },
            Duration::from_millis(200),
        )
        .await;
        assert!(
            matches!(
                backend.reconcile(request(Some(20))).await,
                Err(BackendError::Unavailable(_))
            ),
            "a lost or timed-out reservation read must stay retryable"
        );
        assert_eq!(server.count("MULTI"), 0, "read faults must not dispatch a settlement");
        server.finish().await;
    }
}

#[tokio::test]
async fn hermetic_settlement_lost_exec_reply_is_unconfirmed_and_never_replayed() {
    for fault in [Reply::Disconnect, Reply::Stall] {
        let (backend, server) = setup(
            |backend| {
                let mut steps = marker_reads(backend);
                steps.extend(reservation_reads(backend, Some("50|1000"), "20"));
                steps.extend(settlement_writes(backend, 51.0, fault));
                steps
            },
            Duration::from_millis(200),
        )
        .await;
        assert!(
            matches!(
                backend.reconcile(request(Some(20))).await,
                Err(BackendError::Unconfirmed(_))
            ),
            "a dispatched settlement with no reply must remain ambiguous"
        );
        assert_eq!(
            server.count("WATCH"),
            1,
            "an ambiguous settlement must not restart its read phase"
        );
        assert_eq!(
            server.count("EXEC"),
            1,
            "a lost EXEC reply must never trigger mutation replay"
        );
        server.finish().await;
    }
}

#[tokio::test]
async fn hermetic_settlement_missing_reservation_cleanup_is_deadline_bounded() {
    for fault in [Reply::Disconnect, Reply::Stall] {
        let (backend, server) = setup(
            |backend| {
                let mut steps = marker_reads(backend);
                steps.extend(reservation_reads(backend, None, "20"));
                steps.push(Step::new(&redis::cmd("UNWATCH"), fault));
                steps
            },
            Duration::from_millis(200),
        )
        .await;
        let result = tokio::time::timeout(Duration::from_secs(1), backend.reconcile(request(Some(20)))).await;
        assert!(
            matches!(result, Ok(Err(BackendError::Unavailable(_)))),
            "failed UNWATCH must return within the operation deadline as retryable: {result:?}"
        );
        assert_eq!(
            server.count("EXEC"),
            0,
            "a missing reservation must not dispatch a mutation"
        );
        server.finish().await;
    }
}

#[tokio::test]
async fn hermetic_settlement_rejects_malformed_reservations_before_mutation() {
    let (backend, server) = setup(
        |backend| {
            let mut steps = marker_reads(backend);
            steps.extend(reservation_reads(backend, Some("not-a-reservation"), "20"));
            steps
        },
        Duration::from_secs(1),
    )
    .await;
    assert!(
        matches!(
            backend.reconcile(request(Some(20))).await,
            Err(BackendError::InvalidResponse)
        ),
        "malformed reservation metadata must fail safely"
    );
    assert_eq!(
        server.count("MULTI"),
        0,
        "invalid reservation metadata must never mutate a bucket"
    );
    server.finish().await;
}
