// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! #532 allocation regressions at the real SQLite store boundary.
//!
//! `allocation-counter` counts Rust heap bytes on the calling thread only. The
//! current-thread runtime keeps polling/JSON decoding here, but SQLite's `SQLx`
//! worker and SQLite's C allocator are excluded. Cross-thread frees can make the
//! net balance negative, so `bytes_max` is a thread-local net high-water mark,
//! NOT process peak live memory or RSS. The synchronous shared-history tests
//! separately measure exact live heap bytes without cross-thread transfers.
//! No timing or throughput claim is made. Initial `ResponsesState` copies (#1618)
//! and the required final wire serialization are outside these store seams.

#![cfg(all(test, feature = "sqlite"))]
#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures and public store calls must succeed"
)]
#![expect(clippy::print_stderr, reason = "retain per-repetition heap evidence in test logs")]

#[cfg(test)]
#[expect(
    clippy::too_many_lines,
    reason = "allocation matrices keep measurement and payload-derived assertions together"
)]
mod tests {
    use allocation_counter::{AllocationInfo, measure};
    use praxis_ai_store::{ConversationItemRecord, ConversationItemStore as _, ConversationRecord, StateOwner};
    use praxis_ai_store_backends::SqliteResponseStore;
    use serde_json::{Value, json};
    use tokio::runtime::{Builder, Runtime};

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    const CONVERSATION: &str = "allocation_history";
    const REPETITIONS: usize = 5;
    const MUTATIONS: usize = 8;

    #[test]
    fn sqlite_single_item_mutations_do_not_rebuild_growing_histories() {
        let runtime = runtime();
        let mut baseline = AllocationInfo::default();
        for (count, payload_bytes) in [
            (1, 32),
            (128, 32),
            (1024, 32),
            (4096, 32),
            (16, 8192),
            (64, 8192),
            (256, 8192),
        ] {
            let (store, owner) = runtime.block_on(seed(count, payload_bytes));
            let delta = item(&owner, 999_999, 32);
            let mutate = || {
                runtime.block_on(async {
                    for _ in 0..MUTATIONS {
                        store
                            .create_items_and_sync_messages(&owner, CONVERSATION, std::slice::from_ref(&delta))
                            .await
                            .unwrap();
                        assert!(
                            store
                                .delete_item_and_sync_messages(&owner, CONVERSATION, &delta.item_id)
                                .await
                                .unwrap(),
                            "single appended item deleted"
                        );
                    }
                });
            };
            mutate();
            for repetition in 0..REPETITIONS {
                let info = measure(mutate);
                eprintln!(
                    "sqlite mutations count={count} payload_bytes={payload_bytes} repetition={repetition} operations={} {info:?}",
                    MUTATIONS * 2
                );
                if count == 1 {
                    baseline.bytes_total = baseline.bytes_total.max(info.bytes_total);
                    baseline.bytes_max = baseline.bytes_max.max(info.bytes_max);
                    baseline.bytes_current = baseline.bytes_current.max(info.bytes_current);
                    continue;
                }
                // Admit scheduling/query overhead smaller than half ONE additional
                // history payload, not one full rebuild per each of 16 mutations.
                let growth_budget = u64::try_from((count * payload_bytes - 32) / 2).unwrap();
                assert!(
                    info.bytes_total <= baseline.bytes_total + growth_budget,
                    "mutation work must depend on delta, not history: {info:?}; baseline={baseline:?}"
                );
                assert!(
                    info.bytes_max <= baseline.bytes_max + growth_budget,
                    "mutation high-water mark must not grow with history: {info:?}"
                );
                // SQLx frees query buffers on its worker: net residual is not a
                // leak count. Bound its growth, rather than falsely require zero.
                assert!(
                    info.bytes_current <= baseline.bytes_current + i64::try_from(growth_budget).unwrap(),
                    "mutation residual must not retain growing history: {info:?}"
                );
            }
            runtime.block_on(async {
                let history = store.conversation_history(&owner, CONVERSATION).await.unwrap().unwrap();
                consume(&history, count, payload_bytes);
                store.close().await;
            });
        }
    }

    #[test]
    fn sqlite_restores_complete_growing_histories_without_extra_payload_copies() {
        let runtime = runtime();
        let (empty_store, empty_owner) = runtime.block_on(seed(0, 0));
        let empty_read = || {
            runtime.block_on(async {
                assert!(
                    empty_store
                        .conversation_history(&empty_owner, CONVERSATION)
                        .await
                        .unwrap()
                        .unwrap()
                        .is_empty(),
                    "empty reference history"
                );
            });
        };
        empty_read();
        let fixed_overhead = measure(empty_read).bytes_total;
        runtime.block_on(empty_store.close());
        for (count, payload_bytes) in [(128, 32), (1024, 32), (4096, 32), (16, 8192), (64, 8192), (256, 8192)] {
            let (store, owner) = runtime.block_on(seed(count, payload_bytes));
            let encoded: Vec<String> = (0..count)
                .map(|index| serde_json::to_string(&item(&owner, index, payload_bytes).item_data).unwrap())
                .collect();
            let wire_bytes: usize = encoded.iter().map(String::len).sum();
            let decoded = measure(|| {
                let mut history = Vec::new();
                for value in &encoded {
                    history.push(serde_json::from_str::<Value>(value).unwrap());
                }
                consume(&history, count, payload_bytes);
            });
            eprintln!("decode control count={count} payload_bytes={payload_bytes} wire_bytes={wire_bytes} {decoded:?}");
            let restore = || {
                runtime.block_on(async {
                    let history = store.conversation_history(&owner, CONVERSATION).await.unwrap().unwrap();
                    consume(&history, count, payload_bytes);
                    drop(history);
                });
            };
            restore();
            for repetition in 0..REPETITIONS {
                let info = measure(restore);
                eprintln!(
                    "sqlite restore count={count} payload_bytes={payload_bytes} repetition={repetition} {info:?}"
                );
                // One JSON decode, one SQL TEXT-to-String copy per row, and fixed
                // query costs. Reserve one additional empty-read envelope for pool
                // scheduling; the reserve does not grow with history size.
                let total_budget = decoded.bytes_total + u64::try_from(wire_bytes).unwrap() + 2 * fixed_overhead;
                let row_bytes = encoded.iter().map(String::len).max().unwrap();
                let peak_budget = decoded.bytes_max + u64::try_from(row_bytes).unwrap() + 2 * fixed_overhead;
                assert!(
                    info.bytes_total <= total_budget,
                    "restore must decode once, without full-history sizing or cloning: {info:?}; budget={total_budget}"
                );
                assert!(
                    info.bytes_max <= peak_budget,
                    "restore must not retain a second payload tree: {info:?}; budget={peak_budget}"
                );
                assert!(
                    info.bytes_current <= i64::try_from(2 * fixed_overhead).unwrap(),
                    "dropping history must leave only fixed query overhead: {info:?}"
                );
            }
            runtime.block_on(store.close());
        }
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn runtime() -> Runtime {
        Builder::new_current_thread().enable_all().build().unwrap()
    }

    async fn seed(count: usize, payload_bytes: usize) -> (SqliteResponseStore, StateOwner) {
        let store = SqliteResponseStore::new(
            "sqlite::memory:",
            "responses",
            "conversations",
            Some("items"),
            None,
            None,
        )
        .await
        .unwrap();
        let owner = StateOwner::from_trusted_parts("tenant", "issuer", "subject").unwrap();
        store
            .upsert_conversation(&ConversationRecord {
                conversation_id: CONVERSATION.into(),
                owner: owner.clone(),
                created_at: 1,
                metadata: json!({}),
                messages: json!([]),
            })
            .await
            .unwrap();
        // Keep setup, payload construction and insert serialization outside measure.
        for index in 0..count {
            store
                .create_items_and_sync_messages(&owner, CONVERSATION, &[item(&owner, index, payload_bytes)])
                .await
                .unwrap();
        }
        (store, owner)
    }

    fn item(owner: &StateOwner, index: usize, payload_bytes: usize) -> ConversationItemRecord {
        ConversationItemRecord {
            item_id: format!("item_{index:08}"),
            owner: owner.clone(),
            conversation_id: CONVERSATION.into(),
            item_data: json!({"index": index, "text": "x".repeat(payload_bytes)}),
            created_at: 1,
            position: 0,
        }
    }

    fn consume(history: &[Value], count: usize, payload_bytes: usize) {
        assert_eq!(history.len(), count, "complete history, not a prefix");
        let mut bytes = 0;
        for (index, value) in history.iter().enumerate() {
            assert_eq!(
                value.get("index").and_then(Value::as_u64),
                Some(index as u64),
                "persisted order"
            );
            let text = value.get("text").and_then(Value::as_str).unwrap();
            assert!(text.bytes().all(|byte| byte == b'x'), "full payload consumed");
            bytes += text.len();
        }
        assert_eq!(
            std::hint::black_box(bytes),
            count * payload_bytes,
            "all payload bytes restored"
        );
    }
}
