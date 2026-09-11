// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-time local selection inside a producer-defined priority group.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::RngExt as _;

use super::{
    descriptor::{AdmissionState, CapabilityKind, RouteCandidate},
    group_index::{GroupIndex, SelectionGroup},
    overlay::PickerPolicy,
};

/// Select a candidate from the lowest viable producer-defined group.
pub(crate) fn select_candidate<'a>(
    candidates: &'a [RouteCandidate],
    groups: &GroupIndex,
    kind: CapabilityKind,
    name: &str,
    policy: PickerPolicy,
) -> Option<(&'a RouteCandidate, Option<u32>)> {
    if let Some(capability_groups) = groups.get(&kind).and_then(|by_name| by_name.get(name)) {
        for group in capability_groups {
            if let Some(candidate) = select_from_group(candidates, group, policy) {
                return Some((candidate, Some(group.number)));
            }
        }
        return None;
    }
    select_legacy(candidates, kind, name).map(|candidate| (candidate, None))
}

/// Preserve the exact ordered behavior for overlays without group metadata.
fn select_legacy<'a>(candidates: &'a [RouteCandidate], kind: CapabilityKind, name: &str) -> Option<&'a RouteCandidate> {
    candidates.iter().find(|candidate| {
        candidate.kind == kind
            && &*candidate.name == name
            && candidate.admission_state == AdmissionState::NewAndExisting
    })
}

/// Select one member of a prevalidated, uniformly admitted group.
fn select_from_group<'a>(
    candidates: &'a [RouteCandidate],
    group: &SelectionGroup,
    policy: PickerPolicy,
) -> Option<&'a RouteCandidate> {
    if group.admission_state != AdmissionState::NewAndExisting {
        return None;
    }
    let index = choose_candidate_index(policy, group, &group.next)?;
    candidates.get(index)
}

/// Resolve a policy to an index inside a non-empty selection group.
fn choose_candidate_index(policy: PickerPolicy, group: &SelectionGroup, counter: &AtomicUsize) -> Option<usize> {
    let len = group.candidate_indexes.len();
    if len == 0 {
        return None;
    }
    match policy {
        PickerPolicy::Deterministic => group.candidate_indexes.first().copied(),
        PickerPolicy::RoundRobin => group
            .candidate_indexes
            .get(counter.fetch_add(1, Ordering::Relaxed) % len)
            .copied(),
        PickerPolicy::Random => group.candidate_indexes.get(rand::rng().random_range(0..len)).copied(),
        PickerPolicy::WeightedRandom => {
            if group.total_weight == 0 {
                return None;
            }
            let draw = rand::rng().random_range(0..group.total_weight);
            weighted_index_for_draw(group, draw)
        },
    }
}

/// Resolve a prevalidated weighted draw to its candidate index.
fn weighted_index_for_draw(group: &SelectionGroup, draw: u64) -> Option<usize> {
    if draw >= group.total_weight {
        return None;
    }
    let bucket = group
        .weighted_entries
        .partition_point(|entry| entry.cumulative_upper_bound <= draw);
    group.weighted_entries.get(bucket).map(|entry| entry.candidate_index)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use std::sync::{Arc, atomic::AtomicUsize};

    use super::{
        super::{
            descriptor::{AdmissionState, CapabilityKind, RouteCandidate},
            group_index::{self, SelectionGroup},
            overlay::PickerPolicy,
        },
        choose_candidate_index, select_candidate, weighted_index_for_draw,
    };

    fn candidate(cluster: &str, group: Option<u32>, admission: AdmissionState) -> RouteCandidate {
        RouteCandidate {
            admission_state: admission,
            cluster: Arc::from(cluster),
            credential: None,
            fresh: true,
            kind: CapabilityKind::InferenceModel,
            name: Arc::from("model"),
            rank: None,
            selection_group: group,
            traffic_weight: None,
            selection_tier: None,
            site: Arc::from("site"),
            stable_id: Arc::from(cluster),
        }
    }

    #[test]
    fn round_robin_distributes_only_inside_best_group() {
        let candidates = vec![
            candidate("a", Some(0), AdmissionState::NewAndExisting),
            candidate("b", Some(0), AdmissionState::NewAndExisting),
            candidate("fallback", Some(1), AdmissionState::NewAndExisting),
        ];
        let groups = group_index::build(&candidates).unwrap();
        let selected: Vec<_> = (0..4)
            .map(|_| {
                select_candidate(
                    &candidates,
                    &groups,
                    CapabilityKind::InferenceModel,
                    "model",
                    PickerPolicy::RoundRobin,
                )
                .map(|(candidate, _)| candidate.cluster.to_string())
                .unwrap_or_default()
            })
            .collect();
        assert_eq!(selected, ["a", "b", "a", "b"]);
    }

    #[test]
    fn unavailable_best_group_falls_through() {
        let candidates = vec![
            candidate("draining", Some(0), AdmissionState::ExistingOnly),
            candidate("fallback", Some(1), AdmissionState::NewAndExisting),
        ];
        let groups = group_index::build(&candidates).unwrap();
        let selected = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::RoundRobin,
        );
        assert_eq!(selected.map(|(candidate, _)| &*candidate.cluster), Some("fallback"));
    }

    #[test]
    fn all_inadmissible_groups_return_no_candidate() {
        let candidates = vec![
            candidate("draining", Some(0), AdmissionState::ExistingOnly),
            candidate("excluded", Some(1), AdmissionState::Excluded),
        ];
        let groups = group_index::build(&candidates).unwrap();

        let selected = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::RoundRobin,
        );

        assert!(selected.is_none());
    }

    #[test]
    fn ungrouped_overlay_preserves_first_candidate_selection() {
        let candidates = vec![
            candidate("a", None, AdmissionState::NewAndExisting),
            candidate("b", None, AdmissionState::NewAndExisting),
        ];
        let groups = group_index::build(&candidates).unwrap();
        let selected = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::RoundRobin,
        );
        assert_eq!(selected.map(|(candidate, _)| &*candidate.cluster), Some("a"));
    }

    #[test]
    fn deterministic_always_selects_first_candidate_in_best_group() {
        let candidates = vec![
            candidate("a", Some(0), AdmissionState::NewAndExisting),
            candidate("b", Some(0), AdmissionState::NewAndExisting),
        ];
        let groups = group_index::build(&candidates).unwrap();

        for _ in 0..8 {
            let selected = select_candidate(
                &candidates,
                &groups,
                CapabilityKind::InferenceModel,
                "model",
                PickerPolicy::Deterministic,
            );
            assert_eq!(selected.map(|(candidate, _)| &*candidate.cluster), Some("a"));
        }
    }

    #[test]
    fn random_stays_inside_the_best_group() {
        let candidates = vec![
            candidate("a", Some(0), AdmissionState::NewAndExisting),
            candidate("b", Some(0), AdmissionState::NewAndExisting),
            candidate("fallback", Some(1), AdmissionState::NewAndExisting),
        ];
        let groups = group_index::build(&candidates).unwrap();

        for _ in 0..128 {
            let selected = select_candidate(
                &candidates,
                &groups,
                CapabilityKind::InferenceModel,
                "model",
                PickerPolicy::Random,
            )
            .unwrap();

            assert!(matches!(selected.0.cluster.as_ref(), "a" | "b"));
            assert_eq!(selected.1, Some(0));
        }
    }

    #[test]
    fn round_robin_counter_wraps_without_leaving_group() {
        let counter = AtomicUsize::new(usize::MAX);
        let group = SelectionGroup {
            number: 0,
            admission_state: AdmissionState::NewAndExisting,
            candidate_indexes: vec![0, 1],
            weighted_entries: Vec::new(),
            total_weight: 0,
            next: AtomicUsize::new(0),
        };
        assert_eq!(
            choose_candidate_index(PickerPolicy::RoundRobin, &group, &counter),
            Some(usize::MAX % 2)
        );
        assert_eq!(
            choose_candidate_index(PickerPolicy::RoundRobin, &group, &counter),
            Some(0)
        );
    }

    #[test]
    fn weighted_draw_boundaries_are_proportional_and_exclusive() {
        let mut candidates = vec![
            candidate("a", Some(0), AdmissionState::NewAndExisting),
            candidate("b", Some(0), AdmissionState::NewAndExisting),
        ];
        candidates.first_mut().unwrap().traffic_weight = Some(70);
        candidates.get_mut(1).unwrap().traffic_weight = Some(30);
        let groups = group_index::build(&candidates).unwrap();
        let group = groups
            .get(&CapabilityKind::InferenceModel)
            .and_then(|by_name| by_name.get("model"))
            .and_then(|groups| groups.first())
            .unwrap();

        assert_eq!(weighted_index_for_draw(group, 0), Some(0));
        assert_eq!(weighted_index_for_draw(group, 69), Some(0));
        assert_eq!(weighted_index_for_draw(group, 70), Some(1));
        assert_eq!(weighted_index_for_draw(group, 99), Some(1));
        assert_eq!(weighted_index_for_draw(group, 100), None);
        let counts = (0..group.total_weight).fold([0_u64; 2], |mut counts, draw| {
            let index = weighted_index_for_draw(group, draw).unwrap();
            *counts.get_mut(index).unwrap() += 1;
            counts
        });
        assert_eq!(counts, [70, 30]);
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "independent counter assertions")]
    fn counters_are_isolated_per_capability() {
        let mut other = candidate("other-a", Some(0), AdmissionState::NewAndExisting);
        other.name = Arc::from("other-model");
        let candidates = vec![
            candidate("model-a", Some(0), AdmissionState::NewAndExisting),
            candidate("model-b", Some(0), AdmissionState::NewAndExisting),
            other,
        ];
        let groups = group_index::build(&candidates).unwrap();

        let first_model = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::RoundRobin,
        )
        .unwrap();
        let first_other = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "other-model",
            PickerPolicy::RoundRobin,
        )
        .unwrap();
        let second_model = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::RoundRobin,
        )
        .unwrap();

        assert_eq!(&*first_model.0.cluster, "model-a");
        assert_eq!(&*first_other.0.cluster, "other-a");
        assert_eq!(&*second_model.0.cluster, "model-b");
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "concurrent selection assertion")]
    fn concurrent_round_robin_selection_is_safe_and_complete() {
        let candidates = Arc::new(vec![
            candidate("a", Some(0), AdmissionState::NewAndExisting),
            candidate("b", Some(0), AdmissionState::NewAndExisting),
        ]);
        let groups = Arc::new(group_index::build(&candidates).unwrap());

        let counts = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let candidates = Arc::clone(&candidates);
                    let groups = Arc::clone(&groups);
                    scope.spawn(move || {
                        let mut counts = [0_usize; 2];
                        for _ in 0..128 {
                            let selected = select_candidate(
                                &candidates,
                                &groups,
                                CapabilityKind::InferenceModel,
                                "model",
                                PickerPolicy::RoundRobin,
                            )
                            .unwrap();
                            let cluster = selected.0.cluster.as_ref();
                            assert!(matches!(cluster, "a" | "b"));
                            if cluster == "a" {
                                counts[0] += 1;
                            } else {
                                counts[1] += 1;
                            }
                        }
                        counts
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .fold([0_usize; 2], |mut total, counts| {
                    total[0] += counts[0];
                    total[1] += counts[1];
                    total
                })
        });

        assert_eq!(counts, [512, 512]);
    }
}
