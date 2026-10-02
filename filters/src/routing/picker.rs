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

/// Per-request candidate eligibility derived from claim gates.
///
/// A gate resolves the caller's claims to a set of required label values; only
/// candidates carrying every one stay eligible. With no gate configured every
/// candidate is eligible and the selection path is unchanged.
pub(crate) enum Eligibility {
    /// No gate; every candidate is eligible.
    All,
    /// Keep only candidates whose labels match every `(label, value)` pair.
    Gated(Vec<(String, String)>),
}

impl Eligibility {
    /// Whether `candidate` satisfies every gate. A candidate lacking a gated
    /// label is not eligible: an unlabeled candidate is never assumed in-scope.
    pub(crate) fn allows(&self, candidate: &RouteCandidate) -> bool {
        match self {
            Self::All => true,
            Self::Gated(required) => required
                .iter()
                .all(|(label, value)| candidate.label(label) == Some(value.as_str())),
        }
    }

    /// Whether the gate is inert, so the ungated selection path applies.
    fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }
}

/// Select a candidate from the lowest viable producer-defined group.
#[expect(
    clippy::too_many_arguments,
    reason = "selection needs the full request context plus eligibility"
)]
pub(crate) fn select_candidate<'a>(
    candidates: &'a [RouteCandidate],
    groups: &GroupIndex,
    kind: CapabilityKind,
    name: &str,
    policy: PickerPolicy,
    eligible: &Eligibility,
) -> Option<(&'a RouteCandidate, Option<u32>)> {
    if let Some(capability_groups) = groups.get(&kind).and_then(|by_name| by_name.get(name)) {
        for group in capability_groups {
            if let Some(candidate) = select_from_group(candidates, group, policy, eligible) {
                return Some((candidate, Some(group.number)));
            }
        }
        return None;
    }
    select_legacy(candidates, kind, name, eligible).map(|candidate| (candidate, None))
}

/// Preserve the exact ordered behavior for overlays without group metadata.
fn select_legacy<'a>(
    candidates: &'a [RouteCandidate],
    kind: CapabilityKind,
    name: &str,
    eligible: &Eligibility,
) -> Option<&'a RouteCandidate> {
    candidates.iter().find(|candidate| {
        candidate.kind == kind
            && &*candidate.name == name
            && candidate.admission_state == AdmissionState::NewAndExisting
            && eligible.allows(candidate)
    })
}

/// Select one member of a prevalidated, uniformly admitted group.
fn select_from_group<'a>(
    candidates: &'a [RouteCandidate],
    group: &SelectionGroup,
    policy: PickerPolicy,
    eligible: &Eligibility,
) -> Option<&'a RouteCandidate> {
    if group.admission_state != AdmissionState::NewAndExisting {
        return None;
    }
    // Ungated requests keep the precomputed fast path byte-for-byte; a gate
    // restricts selection to the eligible members of the group.
    let index = if eligible.is_all() {
        choose_candidate_index(policy, group, &group.next)?
    } else {
        choose_eligible_index(policy, group, candidates, eligible)?
    };
    candidates.get(index)
}

/// Resolve a policy to an index among the eligible members of a group.
///
/// Mirrors [`choose_candidate_index`] over the subset of `candidate_indexes`
/// the gate admits, recomputing the weighted total so a drawn weight always
/// lands on an eligible member.
fn choose_eligible_index(
    policy: PickerPolicy,
    group: &SelectionGroup,
    candidates: &[RouteCandidate],
    eligible: &Eligibility,
) -> Option<usize> {
    let admitted: Vec<usize> = group
        .candidate_indexes
        .iter()
        .copied()
        .filter(|&i| candidates.get(i).is_some_and(|c| eligible.allows(c)))
        .collect();
    if admitted.is_empty() {
        return None;
    }
    match policy {
        PickerPolicy::Deterministic => admitted.first().copied(),
        PickerPolicy::RoundRobin => {
            let draw = group.next.fetch_add(1, Ordering::Relaxed) % admitted.len();
            admitted.get(draw).copied()
        },
        PickerPolicy::Random => admitted.get(rand::rng().random_range(0..admitted.len())).copied(),
        PickerPolicy::WeightedRandom => draw_weighted_index(&admitted, candidates),
    }
}

/// Draw one index from `admitted` proportional to each candidate's weight.
/// A candidate with no `traffic_weight` counts as zero and is never drawn.
fn draw_weighted_index(admitted: &[usize], candidates: &[RouteCandidate]) -> Option<usize> {
    let weight = |i: usize| -> u64 { candidates.get(i).and_then(|c| c.traffic_weight).map_or(0, u64::from) };
    let total: u64 = admitted.iter().map(|&i| weight(i)).sum();
    if total == 0 {
        return None;
    }
    let mut draw = rand::rng().random_range(0..total);
    for &i in admitted {
        let w = weight(i);
        if draw < w {
            return Some(i);
        }
        draw -= w;
    }
    None
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
        Eligibility, choose_candidate_index, select_candidate, weighted_index_for_draw,
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
            labels: std::collections::BTreeMap::new(),
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
                    &Eligibility::All,
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
            &Eligibility::All,
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
            &Eligibility::All,
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
            &Eligibility::All,
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
                &Eligibility::All,
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
                &Eligibility::All,
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
            &Eligibility::All,
        )
        .unwrap();
        let first_other = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "other-model",
            PickerPolicy::RoundRobin,
            &Eligibility::All,
        )
        .unwrap();
        let second_model = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::RoundRobin,
            &Eligibility::All,
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
                                &Eligibility::All,
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

    fn with_label(mut c: RouteCandidate, key: &str, value: &str) -> RouteCandidate {
        c.labels.insert(key.to_owned(), value.to_owned());
        c
    }

    #[test]
    fn eligibility_all_allows_every_candidate() {
        let c = candidate("a", None, AdmissionState::NewAndExisting);
        assert!(Eligibility::All.allows(&c));
    }

    #[test]
    fn gated_eligibility_matches_label_and_excludes_unlabeled() {
        let gate = Eligibility::Gated(vec![("region".to_owned(), "eu-west-1".to_owned())]);
        let in_region = with_label(
            candidate("a", None, AdmissionState::NewAndExisting),
            "region",
            "eu-west-1",
        );
        let other_region = with_label(
            candidate("b", None, AdmissionState::NewAndExisting),
            "region",
            "us-east-1",
        );
        let unlabeled = candidate("c", None, AdmissionState::NewAndExisting);
        assert!(gate.allows(&in_region), "matching region is eligible");
        assert!(!gate.allows(&other_region), "other region is fenced out");
        assert!(
            !gate.allows(&unlabeled),
            "an unlabeled candidate is never assumed in-scope"
        );
    }

    #[test]
    fn select_candidate_gate_narrows_to_the_matching_region() {
        let candidates = vec![
            with_label(
                candidate("us", Some(0), AdmissionState::NewAndExisting),
                "region",
                "us-east-1",
            ),
            with_label(
                candidate("eu", Some(0), AdmissionState::NewAndExisting),
                "region",
                "eu-west-1",
            ),
        ];
        let groups = group_index::build(&candidates).unwrap();
        let gate = Eligibility::Gated(vec![("region".to_owned(), "eu-west-1".to_owned())]);
        let selected = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::Deterministic,
            &gate,
        );
        assert_eq!(selected.map(|(c, _)| c.cluster.to_string()), Some("eu".to_owned()));
    }

    #[test]
    fn select_candidate_gate_with_no_eligible_member_returns_none() {
        let candidates = vec![with_label(
            candidate("us", Some(0), AdmissionState::NewAndExisting),
            "region",
            "us-east-1",
        )];
        let groups = group_index::build(&candidates).unwrap();
        let gate = Eligibility::Gated(vec![("region".to_owned(), "eu-west-1".to_owned())]);
        let selected = select_candidate(
            &candidates,
            &groups,
            CapabilityKind::InferenceModel,
            "model",
            PickerPolicy::Deterministic,
            &gate,
        );
        assert!(
            selected.is_none(),
            "no in-region candidate means no selection, never an out-of-region fallback"
        );
    }

    #[test]
    fn weighted_selection_draws_only_eligible_members() {
        let mut eu = candidate("eu", Some(0), AdmissionState::NewAndExisting);
        eu.traffic_weight = Some(1);
        let mut us = candidate("us", Some(0), AdmissionState::NewAndExisting);
        us.traffic_weight = Some(1000);
        let candidates = vec![
            with_label(eu, "region", "eu-west-1"),
            with_label(us, "region", "us-east-1"),
        ];
        let groups = group_index::build(&candidates).unwrap();
        let gate = Eligibility::Gated(vec![("region".to_owned(), "eu-west-1".to_owned())]);
        // Despite the heavy out-of-region weight, every draw must land in-region.
        for _ in 0..256 {
            let selected = select_candidate(
                &candidates,
                &groups,
                CapabilityKind::InferenceModel,
                "model",
                PickerPolicy::WeightedRandom,
                &gate,
            );
            assert_eq!(selected.map(|(c, _)| c.cluster.to_string()), Some("eu".to_owned()));
        }
    }
}
