// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Benchmark-only wrappers for comparing pre- and post-#1618 state ownership.

use std::hint::black_box;

use serde_json::Value;

use super::{
    history::MessageHistory,
    rehydrate::{build_state, replay_messages_from_stored},
    request_input::{InputForm, input_echo_value},
    state::{ResponsesState, normalize_input_owned},
};

/// Recreate the pre-#1618 state and persistence-capture path for comparison.
///
/// # Errors
///
/// Returns a JSON error if the captured request body cannot be parsed.
pub fn legacy(body: Value, stored: Vec<Value>, raw_body: &[u8]) -> Result<(), serde_json::Error> {
    let messages = MessageHistory::from(
        body.get("input")
            .cloned()
            .map(normalize_input_owned)
            .unwrap_or_default(),
    );
    let mut state = ResponsesState {
        input: messages.clone(),
        input_form: InputForm::classify(body.get("input")),
        messages: messages.clone(),
        persisted_messages: messages,
        request_body: body,
        ..ResponsesState::default()
    };
    let stored = MessageHistory::from(stored);
    state.messages.prepend_shared(replay_messages_from_stored(&stored));
    state.persisted_messages.prepend_shared(stored);
    state.history_rehydrated = true;

    let mut store_body: Value = serde_json::from_slice(raw_body)?;
    let store_input = store_body.as_object_mut().and_then(|object| object.remove("input"));
    black_box((state, store_input));
    Ok(())
}

/// Run current managed rehydration and reconstruct the input for store capture.
pub fn current(body: Value, stored: Vec<Value>) {
    let state = build_state(body, stored, vec![], None);
    let store_input = input_echo_value(&state);
    black_box((state, store_input));
}
