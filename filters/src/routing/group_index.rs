// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Validated, immutable indexes for producer-defined selection groups.

use std::{
    collections::HashMap,
    sync::{Arc, atomic::AtomicUsize},
};

use praxis_filter::FilterError;

use super::descriptor::{AdmissionState, CapabilityKind, RouteCandidate};

/// One priority group for a capability.
#[derive(Debug)]
pub(crate) struct SelectionGroup {
    /// Producer-assigned priority number.
    pub(crate) number: u32,
    /// Admission state shared by all group members.
    pub(crate) admission_state: AdmissionState,
    /// Indexes into the immutable candidate array. Duplicate producer entries
    /// are retained as separate slots, so duplication intentionally acts as a
    /// producer-controlled selection weight.
    pub(crate) candidate_indexes: Vec<usize>,
    /// State belongs to this snapshot and therefore resets only on a real
    /// semantic snapshot replacement.
    pub(crate) next: AtomicUsize,
}

/// Precomputed capability lookup used by the request path.
pub(crate) type GroupIndex = HashMap<CapabilityKind, HashMap<Arc<str>, Vec<SelectionGroup>>>;

/// Validate selection-group invariants and build the request-time index.
///
/// Metadata is validated independently for every `(kind, name)` capability:
/// grouped and ungrouped candidates cannot be mixed, numbering starts at zero
/// and is contiguous, and all members of a group share an admission state.
#[expect(
    clippy::too_many_lines,
    reason = "validation and index construction must remain atomic"
)]
pub(crate) fn build(candidates: &[RouteCandidate]) -> Result<GroupIndex, FilterError> {
    let mut index: GroupIndex = HashMap::new();
    let mut mode: HashMap<(CapabilityKind, Arc<str>), bool> = HashMap::new();

    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let key = (candidate.kind, Arc::clone(&candidate.name));
        let grouped = candidate.selection_group.is_some();
        if let Some(previous) = mode.insert(key, grouped)
            && previous != grouped
        {
            return Err(format!(
                "routing: candidate {candidate_index}: capability mixes grouped and ungrouped candidates"
            )
            .into());
        }

        let Some(number) = candidate.selection_group else {
            continue;
        };
        let groups = index
            .entry(candidate.kind)
            .or_default()
            .entry(Arc::clone(&candidate.name))
            .or_default();

        match groups.last_mut() {
            None if number != 0 => {
                return Err(format!("routing: candidate {candidate_index}: selection_group must start at 0").into());
            },
            Some(group) if group.number == number => {
                if group.admission_state != candidate.admission_state {
                    return Err(format!(
                        "routing: candidate {candidate_index}: selection_group {number} mixes admission states"
                    )
                    .into());
                }
                group.candidate_indexes.push(candidate_index);
                continue;
            },
            Some(group) if number != group.number.saturating_add(1) => {
                return Err(format!(
                    "routing: candidate {candidate_index}: selection_group must be contiguous and monotonic"
                )
                .into());
            },
            _ => {},
        }

        groups.push(SelectionGroup {
            number,
            admission_state: candidate.admission_state,
            candidate_indexes: vec![candidate_index],
            next: AtomicUsize::new(0),
        });
    }

    Ok(index)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use std::sync::Arc;

    use super::{
        super::descriptor::{AdmissionState, CapabilityKind, RouteCandidate},
        build,
    };

    fn candidate(group: Option<u32>, admission_state: AdmissionState) -> RouteCandidate {
        RouteCandidate {
            admission_state,
            cluster: Arc::from("cluster"),
            credential: None,
            fresh: true,
            kind: CapabilityKind::InferenceModel,
            name: Arc::from("model"),
            rank: None,
            selection_group: group,
            selection_tier: None,
            site: Arc::from("site"),
            stable_id: Arc::from("stable"),
        }
    }

    #[test]
    fn group_numbering_starts_at_zero() {
        let error = build(&[candidate(Some(1), AdmissionState::NewAndExisting)]).unwrap_err();
        assert!(error.to_string().contains("must start at 0"));
    }

    #[test]
    fn group_members_have_uniform_admission() {
        let error = build(&[
            candidate(Some(0), AdmissionState::NewAndExisting),
            candidate(Some(0), AdmissionState::ExistingOnly),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("mixes admission states"));
    }

    #[test]
    fn group_numbers_must_be_contiguous_per_capability() {
        let error = build(&[
            candidate(Some(0), AdmissionState::NewAndExisting),
            candidate(Some(2), AdmissionState::NewAndExisting),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("contiguous and monotonic"));
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "test fixture assertions use known keys")]
    fn interleaved_capabilities_keep_independent_group_indexes() {
        let mut other = candidate(Some(0), AdmissionState::NewAndExisting);
        other.name = Arc::from("other-model");
        let candidates = vec![
            candidate(Some(0), AdmissionState::NewAndExisting),
            other,
            candidate(Some(0), AdmissionState::NewAndExisting),
        ];
        let index = build(&candidates).unwrap();

        let by_name = index.get(&CapabilityKind::InferenceModel).unwrap();
        assert_eq!(by_name.get("model").unwrap().len(), 1);
        assert_eq!(by_name.get("other-model").unwrap().len(), 1);
        assert_eq!(by_name.get("model").unwrap()[0].candidate_indexes, vec![0, 2]);
    }
}
