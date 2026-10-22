// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Input item pagination for the `OpenAI` Responses API.

use std::collections::HashSet;

use serde::{Serialize, Serializer};

use crate::{
    openai::include::{IncludeFields, project_item},
    store::{ResponseRecord, StoreError},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default page size for input item list operations (matches `OpenAI` default).
pub(crate) const DEFAULT_PAGE_LIMIT: u32 = 20;

/// Maximum page size for input item list operations (matches `OpenAI` maximum).
pub(crate) const MAX_PAGE_LIMIT: u32 = 100;

/// Prefix for a position-bearing cursor used when an item ID is ambiguous.
const POSITION_CURSOR_PREFIX: &str = "praxis_input_items_offset:";

// -----------------------------------------------------------------------------
// Order
// -----------------------------------------------------------------------------

/// Sort order for input item listing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Order {
    /// Oldest first (natural input order).
    Ascending,

    /// Newest first (reversed input order).
    #[default]
    Descending,
}

// -----------------------------------------------------------------------------
// ListParams
// -----------------------------------------------------------------------------

/// Cursor-based pagination parameters for input item listing.
#[derive(Debug, Clone)]
pub(crate) struct ListParams {
    /// Opaque cursor for the next page. `None` starts from the
    /// beginning.
    pub cursor: Option<String>,

    /// Maximum number of items to return (clamped to
    /// `1..=[MAX_PAGE_LIMIT]`).
    pub limit: u32,

    /// Sort order.
    pub order: Order,
}

impl Default for ListParams {
    fn default() -> Self {
        Self {
            cursor: None,
            limit: DEFAULT_PAGE_LIMIT,
            order: Order::default(),
        }
    }
}

impl ListParams {
    /// Return the effective limit, clamped to `1..=[MAX_PAGE_LIMIT]`.
    fn effective_limit(&self) -> u32 {
        self.limit.clamp(1, MAX_PAGE_LIMIT)
    }
}

// -----------------------------------------------------------------------------
// InputItemPage
// -----------------------------------------------------------------------------

/// A page of input items from an `OpenAI` Responses API response.
pub(crate) struct InputItemPage {
    /// Input items as JSON values (heterogeneous types).
    pub data: Vec<serde_json::Value>,

    /// Cursor for the next page (`None` when no more pages). Falls
    /// back to a numeric offset when the last item has no usable ID.
    pub next_cursor: Option<String>,

    /// Whether more pages exist beyond this one.
    pub has_more: bool,
}

impl InputItemPage {
    /// Return the `first_id` cursor for this page, if present.
    pub fn first_id(&self) -> Option<&str> {
        self.data.first().and_then(|v| v.get("id")).and_then(|v| v.as_str())
    }

    /// Return the `last_id` cursor for this page, falling back to `next_cursor`.
    ///
    /// Items normally carry a synthetic ID (see [`normalize_input_items`]),
    /// but non-object array entries cannot be tagged with one. Fall back
    /// to the page's numeric cursor so `after`-based pagination stays
    /// usable even for that edge case, instead of exposing a `null`
    /// `last_id` clients have no way to resume from.
    pub fn last_id(&self) -> Option<&str> {
        self.data
            .last()
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .or(self.next_cursor.as_deref())
    }

    /// Keep an ambiguous item ID separate from its occurrence cursor so direct
    /// HTTP clients can paginate without changing a reference target.
    fn continuation_cursor(&self) -> Option<&str> {
        let cursor = self.next_cursor.as_deref()?;
        (Some(cursor) != self.last_id()).then_some(cursor)
    }
}

impl Serialize for InputItemPage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        InputItemPageView {
            object: "list",
            data: &self.data,
            has_more: self.has_more,
            first_id: self.first_id(),
            last_id: self.last_id(),
            next_cursor: self.continuation_cursor(),
        }
        .serialize(serializer)
    }
}

/// Borrowed view of an input item page formatted for direct JSON response serialization.
#[derive(Serialize)]
struct InputItemPageView<'a> {
    /// Responses API object type (always `"list"`).
    object: &'static str,

    /// Page window data items.
    data: &'a [serde_json::Value],

    /// Whether additional items exist beyond this page window.
    has_more: bool,

    /// First item ID cursor in the page window.
    first_id: Option<&'a str>,

    /// Last item ID cursor (or fallback cursor) in the page window.
    last_id: Option<&'a str>,

    /// Praxis continuation cursor for pages whose item ID is ambiguous.
    #[serde(skip_serializing_if = "Option::is_none")]
    next_cursor: Option<&'a str>,
}

// -----------------------------------------------------------------------------
// Input Item Pagination
// -----------------------------------------------------------------------------

/// Extract and paginate input items from a [`ResponseRecord`].
///
/// Items are extracted from the stored `input` JSON column and
/// paginated in memory using item ID cursors. Numeric offset cursors
/// remain supported as a defensive fallback for malformed, non-object
/// entries that cannot carry a synthetic ID.
///
/// # Errors
///
/// Returns [`StoreError::InvalidInput`] if the cursor is malformed
/// or overflows while calculating the page window.
#[expect(
    clippy::too_many_lines,
    reason = "pagination logic benefits from single-function locality"
)]
pub(crate) fn list_input_items(
    record: &ResponseRecord,
    params: &ListParams,
    includes: IncludeFields,
) -> Result<InputItemPage, StoreError> {
    let mut items = normalize_input_items(record);
    if params.order == Order::Descending {
        items.reverse();
    }

    let offset = params
        .cursor
        .as_deref()
        .map(|cursor| cursor_offset(&items, cursor))
        .transpose()?
        .unwrap_or(0);

    let limit = usize::try_from(params.effective_limit()).map_err(|e| StoreError::InvalidInput(e.to_string()))?;
    let end = offset
        .checked_add(limit)
        .ok_or_else(|| StoreError::InvalidInput("input_items cursor offset overflow".to_owned()))?
        .min(items.len());
    let has_more = end < items.len();
    let next_cursor = page_next_cursor(&items, end, has_more);

    let data: Vec<serde_json::Value> = items
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|mut item| {
            project_item(&mut item, includes);
            item
        })
        .collect();

    Ok(InputItemPage {
        data,
        next_cursor,
        has_more,
    })
}

/// Normalize raw stored input into a list of Responses API items.
///
/// The stored `input` column preserves the original create-request
/// value verbatim (`"Hello"`, `null`, or an `ItemResource[]` array).
/// The `/v1/responses/{id}/input_items` endpoint returns
/// `ItemResource[]`, so this function applies the same
/// resource shape as the public API: string input becomes a
/// synthetic user message resource, null input yields an empty list,
/// and arrays/objects pass through with a stable, collision-safe synthetic `id`
/// assigned to any item that lacks one (see [`ensure_stable_ids`]). Existing
/// IDs are preserved because an `item_reference.id` identifies its target.
fn normalize_input_items(record: &ResponseRecord) -> Vec<serde_json::Value> {
    let items = match &record.input {
        serde_json::Value::Null => vec![],
        serde_json::Value::String(text) => vec![scalar_input_item(&record.id, text)],
        serde_json::Value::Array(arr) => arr.iter().filter(|item| !is_compaction_item(item)).cloned().collect(),
        other => vec![other.clone()],
    };
    ensure_stable_ids(&record.id, items)
}

/// Returns `true` for internal compaction items that should be hidden
/// from the public `input_items` API.
fn is_compaction_item(item: &serde_json::Value) -> bool {
    item.get("type").and_then(serde_json::Value::as_str) == Some("compaction")
}

/// Ensure every object item has a stable ID keyed by its position in the
/// original stored input order (before any order-based reversal).
///
/// Plain content-part objects — the common shape for array `input` —
/// carry no `id` field. Without a synthetic one, `first_id`/`last_id`
/// in the list response stay `null` and clients have no `after` value
/// to resume pagination past the first page, even though `has_more`
/// reports `true`. Synthetic IDs avoid every explicit ID but repeated explicit
/// IDs remain unchanged because they can be semantic reference targets.
fn ensure_stable_ids(response_id: &str, items: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let mut reserved_ids: HashSet<String> = items.iter().filter_map(item_id).map(str::to_owned).collect();

    items
        .into_iter()
        .enumerate()
        .map(|(index, mut item)| {
            let needs_synthetic_id = item.get("id").and_then(serde_json::Value::as_str).is_none();
            if needs_synthetic_id && let Some(obj) = item.as_object_mut() {
                let mut id = format!("msg_{response_id}_input_{index}");
                while !reserved_ids.insert(id.clone()) {
                    id.push('_');
                }
                obj.insert("id".to_owned(), serde_json::Value::String(id));
            }
            item
        })
        .collect()
}

/// Build a public input message resource for scalar create input.
fn scalar_input_item(response_id: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": format!("msg_{response_id}_input_0"),
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": text
            }
        ]
    })
}

/// Resolve an `after` cursor to the offset where the next page starts.
/// ID-based lookup takes precedence: if an item's `id` field matches
/// the cursor string, the offset is the position after its final occurrence.
/// Position-bearing cursors preserve access to each duplicate occurrence;
/// numeric parsing is a fallback for inputs without item IDs.
fn cursor_offset(items: &[serde_json::Value], cursor: &str) -> Result<usize, StoreError> {
    if let Some(offset) = cursor_id_offset(items, cursor) {
        return Ok(offset);
    }

    if let Some(offset) = position_cursor_offset(cursor)? {
        return Ok(offset);
    }

    cursor
        .parse::<usize>()
        .map_err(|e| StoreError::InvalidInput(format!("invalid input_items cursor: {e}")))
}

/// Return the offset after the final item whose `id` matches the cursor.
///
/// Official SDKs derive `after` only from the returned item ID, so repeated IDs
/// produce indistinguishable requests. Selecting the final duplicate keeps
/// retries deterministic and prevents SDK iteration from repeating forever;
/// lossless callers use the separate position-bearing continuation cursor.
fn cursor_id_offset(items: &[serde_json::Value], cursor: &str) -> Option<usize> {
    items
        .iter()
        .rposition(|item| item_id(item) == Some(cursor))
        .map(|index| index + 1)
}

/// Return the public item ID when the input item has one.
fn item_id(item: &serde_json::Value) -> Option<&str> {
    item.get("id").and_then(serde_json::Value::as_str)
}

/// Return the cursor clients should use to fetch the next page.
fn page_next_cursor(items: &[serde_json::Value], end: usize, has_more: bool) -> Option<String> {
    if !has_more {
        return None;
    }

    let last_item = items.get(end.checked_sub(1)?)?;
    let Some(id) = item_id(last_item) else {
        return Some(end.to_string());
    };

    if items.iter().filter(|item| item_id(item) == Some(id)).take(2).count() == 1 {
        Some(id.to_owned())
    } else {
        Some(unique_position_cursor(items, end))
    }
}

/// Create a position cursor that cannot be mistaken for an item ID.
fn unique_position_cursor(items: &[serde_json::Value], offset: usize) -> String {
    let mut candidate = format!("{POSITION_CURSOR_PREFIX}{offset}:");
    loop {
        candidate.push('x');
        if !items.iter().any(|item| item_id(item) == Some(candidate.as_str())) {
            return candidate;
        }
    }
}

/// Accept position cursors because a repeated reference target cannot identify
/// which occurrence produced the continuation on its own.
fn position_cursor_offset(cursor: &str) -> Result<Option<usize>, StoreError> {
    let Some(encoded) = cursor.strip_prefix(POSITION_CURSOR_PREFIX) else {
        return Ok(None);
    };
    let Some((offset, discriminator)) = encoded.split_once(':') else {
        return Err(StoreError::InvalidInput(
            "invalid input_items position cursor".to_owned(),
        ));
    };
    let offset = offset
        .parse::<usize>()
        .map_err(|e| StoreError::InvalidInput(format!("invalid input_items position cursor: {e}")))?;
    if discriminator.is_empty() || !discriminator.bytes().all(|byte| byte == b'x') {
        return Err(StoreError::InvalidInput(
            "invalid input_items position cursor discriminator".to_owned(),
        ));
    }
    Ok(Some(offset))
}
