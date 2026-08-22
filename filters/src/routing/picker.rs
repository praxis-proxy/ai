// SPDX-License-Identifier: MIT
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
    let ordinal = choose_ordinal(policy, group.candidate_indexes.len(), &group.next);
    select_from_group_at(candidates, group, ordinal)
}

/// Select a specific member of a prevalidated group.
///
/// This small ordinal seam keeps policy tests deterministic without changing
/// production randomness or making the request path injectable at runtime.
fn select_from_group_at<'a>(
    candidates: &'a [RouteCandidate],
    group: &SelectionGroup,
    ordinal: usize,
) -> Option<&'a RouteCandidate> {
    group
        .candidate_indexes
        .get(ordinal)
        .and_then(|&index| candidates.get(index))
}

/// Resolve a policy to an index inside a non-empty selection group.
fn choose_ordinal(policy: PickerPolicy, len: usize, counter: &AtomicUsize) -> usize {
    match policy {
        PickerPolicy::Deterministic => 0,
        PickerPolicy::RoundRobin => counter.fetch_add(1, Ordering::Relaxed) % len,
        PickerPolicy::Random => rand::rng().random_range(0..len),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use std::sync::{Arc, atomic::AtomicUsize};

    use super::{
        super::{
            descriptor::{AdmissionState, CapabilityKind, RouteCandidate},
            group_index,
            overlay::PickerPolicy,
        },
        choose_ordinal, select_candidate, select_from_group_at,
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

        let best_group = groups
            .get(&CapabilityKind::InferenceModel)
            .unwrap()
            .get("model")
            .unwrap()
            .first()
            .unwrap();
        let selected_a = select_from_group_at(&candidates, best_group, 0).unwrap();
        let selected_b = select_from_group_at(&candidates, best_group, 1).unwrap();

        assert_eq!(&*selected_a.cluster, "a");
        assert_eq!(&*selected_b.cluster, "b");
        assert_eq!(best_group.number, 0);
    }

    #[test]
    fn round_robin_counter_wraps_without_leaving_group() {
        let counter = AtomicUsize::new(usize::MAX);
        assert_eq!(choose_ordinal(PickerPolicy::RoundRobin, 2, &counter), usize::MAX % 2);
        assert_eq!(choose_ordinal(PickerPolicy::RoundRobin, 2, &counter), 0);
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
