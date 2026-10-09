// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Item-level shared ownership for replay and exact persisted history.

#[cfg(test)]
use std::ops::RangeBounds;
use std::{
    ops::{Index, IndexMut},
    sync::Arc,
};

use serde::{Serialize, Serializer};
use serde_json::Value;

/// Ordered JSON history with item-level copy-on-write mutation.
///
/// Cloning this container shares payloads. Editing one item detaches only that
/// item, preserving the independently mutable replay and persistence views.
#[derive(Clone, Debug, Default)]
pub(crate) struct MessageHistory(Vec<Arc<Value>>);

impl MessageHistory {
    /// Number of history items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the history is empty.
    #[cfg(any(feature = "store", feature = "openai-compact", test))]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Borrow items without materializing a second JSON history.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Value> + ExactSizeIterator {
        self.0.iter().map(AsRef::as_ref)
    }

    /// Borrow an item by position.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.0.get(index).map(AsRef::as_ref)
    }

    /// Find the highest position whose contiguous run equals `needle`.
    #[must_use]
    pub fn rposition_run(&self, needle: &[Value]) -> Option<usize> {
        let needle_len = needle.len();
        if needle_len == 0 || needle_len > self.0.len() {
            return None;
        }
        (0..=self.0.len().saturating_sub(needle_len)).rev().find(|&start| {
            (0..needle_len).all(|offset| {
                self.0
                    .get(start + offset)
                    .zip(needle.get(offset))
                    .is_some_and(|(item, want)| item.as_ref() == want)
            })
        })
    }

    /// Mutably borrow one item, copying its payload only if shared.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.0.get_mut(index).map(Arc::make_mut)
    }

    /// Borrow the last item.
    #[cfg(test)]
    #[must_use]
    pub fn last(&self) -> Option<&Value> {
        self.0.last().map(AsRef::as_ref)
    }

    /// Remove every item.
    #[cfg(feature = "openai-compact")]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Append an owned item.
    pub fn push(&mut self, item: Value) {
        self.0.push(Arc::new(item));
    }

    /// Append an already shared item without copying its payload.
    pub(crate) fn push_shared(&mut self, item: Arc<Value>) {
        self.0.push(item);
    }

    /// Borrow the shared handles for immutable projections.
    #[cfg(any(feature = "store", test))]
    pub(crate) fn shared_items(&self) -> impl Iterator<Item = &Arc<Value>> {
        self.0.iter()
    }

    /// Prepend a shared history without copying JSON payloads.
    #[cfg(feature = "store")]
    pub(crate) fn prepend_shared(&mut self, prefix: Self) {
        self.0.splice(0..0, prefix.0);
    }

    /// Append owned items.
    pub fn extend(&mut self, items: impl IntoIterator<Item = Value>) {
        self.0.extend(items.into_iter().map(Arc::new));
    }

    /// Insert an owned item at the given position.
    ///
    /// # Panics
    /// Panics when the position is greater than the history length.
    #[cfg(test)]
    pub fn insert(&mut self, index: usize, item: Value) {
        self.0.insert(index, Arc::new(item));
    }

    /// Replace a range with owned items.
    ///
    /// # Panics
    /// Panics when the range is invalid.
    #[cfg(test)]
    pub fn splice<R: RangeBounds<usize>>(&mut self, range: R, items: impl IntoIterator<Item = Value>) {
        self.0.splice(range, items.into_iter().map(Arc::new));
    }

    /// Retain items satisfying a borrowed predicate.
    #[cfg(feature = "openai-mcp-tools")]
    pub fn retain(&mut self, mut predicate: impl FnMut(&Value) -> bool) {
        self.0.retain(|item| predicate(item));
    }

    /// Split history without copying any item payloads.
    ///
    /// # Panics
    /// Panics when the position is greater than the history length.
    #[cfg(feature = "openai-compact")]
    pub fn split_off(&mut self, at: usize) -> Self {
        Self(self.0.split_off(at))
    }

    /// Append another history by transferring its shared handles.
    #[cfg(feature = "openai-compact")]
    pub fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0);
    }

    /// Test whether two positions share the same immutable JSON allocation.
    #[cfg(test)]
    pub(crate) fn shares_item_with(&self, index: usize, other: &Self, other_index: usize) -> bool {
        self.0
            .get(index)
            .zip(other.0.get(other_index))
            .is_some_and(|(left, right)| Arc::ptr_eq(left, right))
    }

    /// Convert to owned JSON at a boundary that requires independently owned
    /// values. Shared items are copied here, not on rehydration.
    #[cfg(feature = "store")]
    #[must_use]
    pub fn into_values(self) -> Vec<Value> {
        self.into_iter().collect()
    }
}

impl From<Vec<Value>> for MessageHistory {
    fn from(items: Vec<Value>) -> Self {
        Self(items.into_iter().map(Arc::new).collect())
    }
}

impl FromIterator<Value> for MessageHistory {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        Self(iter.into_iter().map(Arc::new).collect())
    }
}

impl IntoIterator for MessageHistory {
    type IntoIter = std::iter::Map<std::vec::IntoIter<Arc<Value>>, fn(Arc<Value>) -> Value>;
    type Item = Value;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(Arc::unwrap_or_clone)
    }
}

impl<'a> IntoIterator for &'a MessageHistory {
    type IntoIter = std::iter::Map<std::slice::Iter<'a, Arc<Value>>, fn(&'a Arc<Value>) -> &'a Value>;
    type Item = &'a Value;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(AsRef::as_ref)
    }
}

impl<'a> IntoIterator for &'a mut MessageHistory {
    type IntoIter = std::iter::Map<std::slice::IterMut<'a, Arc<Value>>, fn(&'a mut Arc<Value>) -> &'a mut Value>;
    type Item = &'a mut Value;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut().map(Arc::make_mut)
    }
}

impl Index<usize> for MessageHistory {
    type Output = Value;

    fn index(&self, index: usize) -> &Value {
        self.0.index(index)
    }
}

impl IndexMut<usize> for MessageHistory {
    fn index_mut(&mut self, index: usize) -> &mut Value {
        Arc::make_mut(self.0.index_mut(index))
    }
}

impl Serialize for MessageHistory {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl PartialEq for MessageHistory {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl PartialEq<Vec<Value>> for MessageHistory {
    fn eq(&self, other: &Vec<Value>) -> bool {
        self.iter().eq(other.iter())
    }
}

impl PartialEq<MessageHistory> for Vec<Value> {
    fn eq(&self, other: &MessageHistory) -> bool {
        self.iter().eq(other.iter())
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "test fixture has two known items and a known id field"
)]
mod tests {
    use super::*;

    #[test]
    fn cloning_shares_payloads_and_mutation_detaches_only_changed_item() {
        let original = MessageHistory::from(vec![serde_json::json!({"id": "a"}), serde_json::json!({"id": "b"})]);
        let mut replay = original.clone();
        assert!(original.shares_item_with(0, &replay, 0), "first payload shared");
        replay[0]["id"] = serde_json::json!("changed");
        assert!(!original.shares_item_with(0, &replay, 0), "edited item detached");
        assert!(original.shares_item_with(1, &replay, 1), "untouched item stays shared");
        assert_eq!(original[0]["id"], "a", "persisted content unchanged");
    }
}
