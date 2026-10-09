// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded raw reasoning state and its Responses event lifecycle.

use serde_json::{Value, json};

use super::{ConvertError, EmitState, StreamLimits, emit_event, events};

/// One reasoning output item, allocated at the first non-empty provider delta.
pub(super) struct ReasoningState {
    /// Stable position shared by incremental events and the terminal resource.
    pub(super) output_index: usize,
    /// Round-specific identity shared with the finite translator.
    pub(super) item_id: String,
    /// Bounded semantic text retained for terminal construction and persistence.
    pub(super) text: String,
}

impl ReasoningState {
    /// Announce the item and its single raw reasoning content part.
    pub(super) fn open(
        &self,
        emit: &mut EmitState,
        limits: &StreamLimits,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        let item = json!({
            "id": self.item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
            "content": [],
        });
        emit_event(
            emit,
            limits,
            true,
            events::output_item_added(self.output_index, &item),
            out,
        )?;
        emit_event(
            emit,
            limits,
            true,
            events::content_part_added(
                &self.item_id,
                self.output_index,
                0,
                &json!({"type": "reasoning_text", "text": ""}),
            ),
            out,
        )
    }

    /// Close using the terminal item after the accumulated text has moved into translation.
    pub(super) fn close(
        &self,
        resource: &Value,
        emit: &mut EmitState,
        limits: &StreamLimits,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        let item = self.terminal_item(resource)?;
        let part = item
            .get("content")
            .and_then(|content| content.get(0))
            .ok_or(ConvertError::InvalidTerminalResource)?;
        let text = part
            .get("text")
            .and_then(Value::as_str)
            .ok_or(ConvertError::InvalidTerminalResource)?;
        emit_event(
            emit,
            limits,
            true,
            events::reasoning_text_done(&self.item_id, self.output_index, text),
            out,
        )?;
        emit_event(
            emit,
            limits,
            true,
            events::content_part_done(&self.item_id, self.output_index, 0, part),
            out,
        )?;
        emit_event(
            emit,
            limits,
            true,
            events::output_item_done(self.output_index, item),
            out,
        )
    }

    /// Locate the item at its announced output position after terminal ordering.
    fn terminal_item<'a>(&self, resource: &'a Value) -> Result<&'a Value, ConvertError> {
        resource
            .get("output")
            .and_then(|output| output.get(self.output_index))
            .ok_or(ConvertError::InvalidTerminalResource)
    }
}
