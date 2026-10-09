// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Borrow selected input items until a downstream edit requires ownership.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::needless_raw_string_hashes,
    clippy::needless_raw_strings,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::{borrow::Cow, fmt};

use praxis_filter::FilterError;
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Value, value::RawValue};

use super::{
    ResponsesState, TopLevelField, TopLevelMember, messages_for_backend, provider_owns_conversation,
    selected_body_slice,
};

/// Reconcile live input with borrowed canonical history and agentic results.
pub(super) fn reconcile<'a>(
    body: &[u8],
    members: &[TopLevelMember],
    state: &'a ResponsesState,
    preserve_native_compaction: bool,
) -> Result<Vec<Cow<'a, Value>>, FilterError> {
    let start = if provider_owns_conversation(state) && state.iteration > 0 {
        state.provider_history_len
    } else {
        0
    };
    let canonical: Vec<_> = state.messages.iter().skip(start).map(Cow::Borrowed).collect();
    let Some(member) = members.iter().rev().find(|member| member.name == TopLevelField::Input) else {
        return Ok(canonical);
    };
    let raw: &RawValue = serde_json::from_slice(selected_body_slice(body, member.value_start, member.value_end)?)
        .map_err(input_error)?;
    if raw.get().starts_with('"') {
        return reconcile_shorthand(raw, state, canonical, preserve_native_compaction);
    }
    let raw_items = input_items(raw)?;
    if matches_items(&raw_items, state.input.iter())? {
        return Ok(canonical);
    }
    reconcile_items(&raw_items, state, canonical, start, preserve_native_compaction)
}

/// Extract item references without decoding their payloads.
fn input_items(raw: &RawValue) -> Result<Vec<&RawValue>, FilterError> {
    match raw.get().as_bytes().first() {
        Some(b'[') => serde_json::from_str(raw.get()).map_err(input_error),
        Some(b'{') => Ok(vec![raw]),
        _ => {
            // RawValue defers number validation; retain Value's range errors.
            drop(serde_json::from_str::<Value>(raw.get()).map_err(input_error)?);
            Ok(Vec::new())
        },
    }
}

/// String shorthand equals only the exact normalized user-message shape.
fn reconcile_shorthand<'a>(
    raw: &RawValue,
    state: &'a ResponsesState,
    canonical: Vec<Cow<'a, Value>>,
    native: bool,
) -> Result<Vec<Cow<'a, Value>>, FilterError> {
    if let [Value::Object(message)] = state.input.as_slice()
        && message.len() == 3
        && message.get("type").and_then(Value::as_str) == Some("message")
        && message.get("role").and_then(Value::as_str) == Some("user")
        && let Some(content) = message.get("content")
        && equal(raw, content).map_err(input_error)?
    {
        return Ok(canonical);
    }
    let normalized = super::normalize_input_owned(serde_json::from_str(raw.get()).map_err(input_error)?);
    Ok(merge(
        normalized.into_iter().map(Cow::Owned).collect(),
        state,
        canonical,
        native,
    ))
}

/// Project only individual summaries; ordinary payloads remain borrowed.
fn project<'a>(state: &ResponsesState, item: &'a Value, native: bool) -> Cow<'a, Value> {
    match messages_for_backend(std::slice::from_ref(item), native, &state.provider_compaction_ids) {
        Cow::Borrowed(_) => Cow::Borrowed(item),
        Cow::Owned(items) => Cow::Owned(items.into_iter().next().unwrap_or(Value::Null)),
    }
}

/// Reuse state payloads for unchanged items, owning only live edits.
fn reconcile_items<'a>(
    raw: &[&RawValue],
    state: &'a ResponsesState,
    canonical: Vec<Cow<'a, Value>>,
    start: usize,
    native: bool,
) -> Result<Vec<Cow<'a, Value>>, FilterError> {
    let projected: Vec<_> = state
        .messages
        .iter()
        .skip(start)
        .map(|item| project(state, item, native))
        .collect();
    if matches_items(raw, projected.iter().map(AsRef::as_ref))? {
        return Ok(canonical);
    }
    let live = raw
        .iter()
        .enumerate()
        .map(|(index, raw)| reuse_or_parse(raw, projected.get(index), state.input.get(index)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(merge(live, state, canonical, native))
}

/// Copy a translated summary only when a live edit requires reconciliation.
fn reuse_or_parse<'a>(
    raw: &RawValue,
    projected: Option<&Cow<'a, Value>>,
    input: Option<&'a Value>,
) -> Result<Cow<'a, Value>, FilterError> {
    if let Some(candidate) = projected
        && equal(raw, candidate).map_err(input_error)?
    {
        return Ok(candidate.clone());
    }
    if let Some(candidate) = input
        && equal(raw, candidate).map_err(input_error)?
    {
        return Ok(Cow::Borrowed(candidate));
    }
    Ok(Cow::Owned(serde_json::from_str(raw.get()).map_err(input_error)?))
}

/// Append borrowed results, retaining local history only when it is absent.
fn merge<'a>(
    mut live: Vec<Cow<'a, Value>>,
    state: &'a ResponsesState,
    canonical: Vec<Cow<'a, Value>>,
    native: bool,
) -> Vec<Cow<'a, Value>> {
    let appended_start = if state.input.is_empty() {
        0
    } else {
        state
            .messages
            .rposition_run(&state.input)
            .map_or(state.messages.len(), |start| start + state.input.len())
    };
    let appended: Vec<_> = state.messages.iter().skip(appended_start).map(Cow::Borrowed).collect();
    if state.history_rehydrated {
        let history_len = state.messages.len().saturating_sub(state.input.len() + appended.len());
        if !has_history(&live, state, history_len, native) {
            let mut history: Vec<_> = state.messages.iter().take(history_len).map(Cow::Borrowed).collect();
            history.append(&mut live);
            live = history;
        }
        append(&mut live, appended);
    } else if provider_owns_conversation(state) && state.iteration > 0 {
        append(&mut live, canonical);
    } else {
        append(&mut live, appended);
    }
    live
}

/// Recognize an already-rehydrated prefix without materializing that history.
fn has_history(live: &[Cow<'_, Value>], state: &ResponsesState, history_len: usize, native: bool) -> bool {
    live.len() >= history_len
        && live
            .iter()
            .zip(state.messages.iter().take(history_len))
            .all(|(live, history)| project(state, history, native).as_ref() == live.as_ref())
}

/// Do not duplicate a tool result already included by an earlier rebuild.
fn append<'a>(live: &mut Vec<Cow<'a, Value>>, trailing: Vec<Cow<'a, Value>>) {
    if !live.ends_with(&trailing) {
        live.extend(trailing);
    }
}

/// Attach the selected-input boundary to a consumed deserializer error.
#[expect(clippy::needless_pass_by_value, reason = "map_err consumes its error")]
fn input_error(error: serde_json::Error) -> FilterError {
    format!("openai_responses_proxy: invalid selected input: {error}").into()
}

/// Compare a live array with a borrowed projection without decoding payloads.
fn matches_items<'a>(
    raw: &[&RawValue],
    expected: impl ExactSizeIterator<Item = &'a Value> + 'a,
) -> Result<bool, FilterError> {
    if raw.len() != expected.len() {
        return Ok(false);
    }
    for (raw, expected) in raw.iter().zip(expected) {
        if !equal(raw, expected).map_err(input_error)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Compare JSON semantically without constructing owned message payloads.
fn equal(raw: &RawValue, expected: &Value) -> Result<bool, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_str(raw.get());
    let result = Matches(Some(expected)).deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(result)
}

/// Missing expected values still consume and validate the live JSON value.
struct Matches<'a>(Option<&'a Value>);

impl<'de> DeserializeSeed<'de> for Matches<'_> {
    type Value = bool;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Matches<'_> {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<bool, E> {
        Ok(self.0 == Some(&Value::Null))
    }

    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<bool, E> {
        Ok(self.0 == Some(&Value::Bool(v)))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<bool, E> {
        Ok(self.0 == Some(&Value::from(v)))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<bool, E> {
        Ok(self.0 == Some(&Value::from(v)))
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<bool, E> {
        Ok(self.0 == Some(&Value::from(v)))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<bool, E> {
        Ok(self.0.and_then(Value::as_str) == Some(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<bool, A::Error> {
        let expected = self.0.and_then(Value::as_array);
        let mut index = 0;
        let mut matches = expected.is_some();
        while let Some(equal) = seq.next_element_seed(Matches(expected.and_then(|items| items.get(index))))? {
            matches &= equal;
            index += 1;
        }
        Ok(matches && expected.is_some_and(|items| items.len() == index))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<bool, A::Error> {
        let Some(expected) = self.0.and_then(Value::as_object) else {
            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
            return Ok(false);
        };
        let mut matches = vec![false; expected.len()];
        let mut extra = false;
        while let Some(key) = map.next_key_seed(Text)? {
            if let Some((index, value)) = expected
                .iter()
                .enumerate()
                .find_map(|(index, (name, value))| (name == key.as_ref()).then_some((index, value)))
            {
                // Like Value deserialization, duplicate members use the last
                // occurrence, even if an earlier occurrence did not match.
                if let Some(slot) = matches.get_mut(index) {
                    *slot = map.next_value_seed(Matches(Some(value)))?;
                }
            } else {
                map.next_value::<IgnoredAny>()?;
                extra = true;
            }
        }
        Ok(!extra && matches.into_iter().all(|matches| matches))
    }
}

/// Borrow unescaped strings, owning only JSON escape decoding scratch space.
struct Text;
impl<'de> DeserializeSeed<'de> for Text {
    type Value = Cow<'de, str>;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}
impl<'de> Visitor<'de> for Text {
    type Value = Cow<'de, str>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string")
    }

    fn visit_borrowed_str<E: serde::de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(v))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Cow::Owned(v.to_owned()))
    }
}
