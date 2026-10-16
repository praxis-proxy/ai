// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! #532 synchronous shared-history allocation guard. These are Rust heap bytes,
//! not RSS. All allocations and frees occur on this thread, unlike `SQLx` reads.
//! Payload setup is excluded; complete ordered consumption and release are
//! included. Required final wire serialization and initial `ResponsesState`
//! copies (#1618) are deliberately not attributed to shared-history cloning.

#![expect(clippy::unwrap_used, reason = "fixed test fixtures and allocation measurements")]
#![expect(clippy::print_stderr, reason = "retain per-repetition heap evidence in test logs")]

#[cfg(test)]
#[expect(
    clippy::too_many_lines,
    reason = "allocation matrix keeps measurement and payload-derived assertions together"
)]
mod tests {
    use std::{hint::black_box, mem::size_of, sync::Arc};

    use allocation_counter::measure;
    use serde_json::{Value, json};

    use crate::openai::responses::history::MessageHistory;

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    const REPETITIONS: usize = 5;

    #[test]
    fn shared_history_restore_and_sparse_edit_allocate_handles_not_payload_copies() {
        for (count, payload_bytes) in [(128, 32), (1024, 32), (8192, 32), (16, 8192), (64, 8192), (256, 8192)] {
            let original = MessageHistory::from(
                (0..count)
                    .map(|index| json!({"index": index, "text": "x".repeat(payload_bytes)}))
                    .collect::<Vec<_>>(),
            );
            // One-item COW costs depend on serde_json's map layout, not history
            // length. Measure that independently rather than pin allocator internals.
            let item_copy = measure(|| {
                black_box(original.get(0).unwrap().clone());
            });
            let exercise = || {
                let mut replay = original.clone();
                replay
                    .get_mut(count / 2)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert("index".into(), json!(count));
                for (index, value) in replay.iter().enumerate() {
                    let expected = if index == count / 2 { count } else { index };
                    assert_eq!(
                        value.get("index").and_then(Value::as_u64),
                        Some(expected as u64),
                        "ordered history with one edit"
                    );
                    let text = value.get("text").and_then(Value::as_str).unwrap();
                    assert_eq!(text.len(), payload_bytes, "full payload retained");
                    assert!(
                        black_box(text).bytes().all(|byte| byte == b'x'),
                        "full payload consumed"
                    );
                }
                assert_eq!(replay.len(), count, "complete replay");
                assert_eq!(
                    original.get(count / 2).unwrap().get("index").and_then(Value::as_u64),
                    Some((count / 2) as u64),
                    "persisted view remains unchanged"
                );
                drop(replay);
            };
            exercise();
            for repetition in 0..REPETITIONS {
                let info = measure(exercise);
                // One handle per item, one detached Value + Arc header, and the
                // replacement key. No payload-proportional full-history allowance.
                let budget =
                    (count * size_of::<Arc<Value>>() + size_of::<Value>() + 2 * size_of::<usize>() + "index".len())
                        as u64
                        + item_copy.bytes_total;
                eprintln!(
                    "shared restore/edit count={count} payload_bytes={payload_bytes} repetition={repetition} budget_bytes={budget} {info:?}"
                );
                assert!(
                    info.bytes_total <= budget,
                    "no sizing serialization or deep history copy: {info:?}"
                );
                assert!(
                    info.bytes_max <= budget,
                    "only handles and one edited payload may be live: {info:?}"
                );
                assert_eq!(info.bytes_current, 0, "release restored history and edited item");
                assert_eq!(info.count_current, 0, "release every temporary allocation");
            }
        }
    }
}
