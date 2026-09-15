// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Candidate scoring, shaped after the endpoint picker's scheduling framework.
//!
//! A [`Scorer`] rates every candidate on one signal, normalised so higher is
//! better and comparable across signals. Scores combine by weight and a picker
//! takes the best. A new signal is a new [`Scorer`] and changes nothing else.
//!
//! One difference from the endpoint picker is deliberate: it drops endpoints
//! that report no metrics, since a pooled endpoint is expected to. A route
//! candidate differs, because a third-party API is routable yet exposes nothing.
//! Reachability would then depend on observability, so an incompletely covered
//! set stays in the order the overlay rendered.

use std::sync::Arc;

use super::{
    descriptor::RouteCandidate,
    load::{LoadStore, SignalConfig, SignalScale},
};

/// Rates candidates on one signal.
///
/// Returns `None` for a candidate the signal says nothing about, which is not
/// the same as rating it badly.
pub(crate) trait Scorer: Send + Sync {
    /// Raw values per candidate, in the order given.
    ///
    /// Normalisation is applied by [`score_all`], so an implementation reports
    /// the quantity it measures rather than a rating.
    fn measure(&self, candidates: &[&RouteCandidate], now_ms: i64) -> Vec<Option<f64>>;

    /// Whether a lower measurement is better.
    fn lower_is_better(&self) -> bool;

    /// Spread below which every candidate rates the same on this signal.
    ///
    /// In the signal's own units. Zero means every difference counts.
    fn deadband(&self) -> f64 {
        0.0
    }

    /// Whether the measurement is already a 0.0 to 1.0 rating.
    ///
    /// Min-max scaling makes a signal relative to the candidates in hand,
    /// which is right for an unbounded quantity like queue depth and wrong for
    /// a ratio: two pools at 0.10 and 0.12 utilisation would score 1.0 and 0.0
    /// and the gateway would treat a trivial difference as a decisive one.
    fn already_normalised(&self) -> bool {
        false
    }

    /// Relative weight in the combined score.
    fn weight(&self) -> f64 {
        1.0
    }
}

/// Min and max of the present readings, in one pass with no allocation.
fn min_max(raw: &[Option<f64>]) -> Option<(f64, f64)> {
    raw.iter().flatten().copied().fold(None, |acc, v| {
        Some(acc.map_or((v, v), |(min, max)| (min.min(v), max.max(v))))
    })
}

/// Whether the candidates differ by less than this signal can meaningfully
/// distinguish.
fn within_deadband(raw: &[Option<f64>], deadband: f64) -> bool {
    if deadband <= 0.0 {
        return false;
    }
    let Some((min, max)) = min_max(raw) else {
        return false;
    };
    max - min <= deadband
}

/// Rate a measurement against the other candidates, so the best reading in
/// hand becomes 1.0 and the worst 0.0.
///
/// This is what the endpoint picker's queue scorer does:
/// `(max - v) / (max - min)`, and 1.0 for everyone when they are all equal.
fn normalise(raw: &[Option<f64>], lower_is_better: bool) -> Vec<Option<f64>> {
    let Some((min, max)) = min_max(raw) else {
        return vec![None; raw.len()];
    };
    let span = max - min;
    raw.iter()
        .map(|value| {
            value.map(|v| {
                if span <= f64::EPSILON {
                    return 1.0;
                }
                let scaled = (v - min) / span;
                if lower_is_better { 1.0 - scaled } else { scaled }
            })
        })
        .collect()
}

/// Take a measurement that is already a rating, inverting it when lower is
/// better and clamping what a misreporting provider might send.
fn rate_directly(raw: &[Option<f64>], lower_is_better: bool) -> Vec<Option<f64>> {
    raw.iter()
        .map(|value| {
            value.map(|v| {
                let clamped = v.clamp(0.0, 1.0);
                if lower_is_better { 1.0 - clamped } else { clamped }
            })
        })
        .collect()
}

/// Floor for the scoring window, so the aggregate always spans a stable view
/// even when the configured freshness horizon is shorter.
const MIN_WINDOW_MS: i64 = 15_000;

/// One metric, read from the live load store.
///
/// The signal it rates is named by configuration, so a grid that publishes a
/// signal this code has never heard of scores on it without a change here.
pub(crate) struct MetricScorer {
    /// Windowed signals keyed by `"site/cluster"`.
    pub store: Arc<LoadStore>,
    /// Metric that carries this signal.
    pub metric: Box<str>,
    /// Age past which a sample is ignored.
    pub max_age_ms: i64,
    /// Relative weight in the combined score.
    pub weight: f64,
    /// Whether a lower reading is the better one.
    pub lower_is_better: bool,
    /// Whether the reading is already a 0.0 to 1.0 rating.
    pub already_normalised: bool,
    /// Spread below which candidates rate the same, in the signal's units.
    pub deadband: f64,
}

impl Scorer for MetricScorer {
    fn measure(&self, candidates: &[&RouteCandidate], now_ms: i64) -> Vec<Option<f64>> {
        // Aggregate over a window rather than taking the last value, so a site
        // that drained a burst does not instantly snap to idle. The window is
        // the configured freshness horizon, floored at MIN_WINDOW_MS.
        let window_ms = self.max_age_ms.max(MIN_WINDOW_MS);
        candidates
            .iter()
            .map(|c| {
                self.store
                    .window_worst(&c.load_key, &self.metric, now_ms, window_ms, self.lower_is_better)
            })
            .collect()
    }

    fn lower_is_better(&self) -> bool {
        self.lower_is_better
    }

    fn already_normalised(&self) -> bool {
        self.already_normalised
    }

    fn deadband(&self) -> f64 {
        self.deadband
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

/// Combined score per candidate, or `None` where no scorer had anything to say.
pub(crate) fn score_all(scorers: &[Box<dyn Scorer>], candidates: &[&RouteCandidate], now_ms: i64) -> Vec<Option<f64>> {
    let mut totals = vec![0.0; candidates.len()];
    let mut weights = vec![0.0; candidates.len()];
    for scorer in scorers {
        let weight = scorer.weight();
        let measured = scorer.measure(candidates, now_ms);
        // A spread too small to mean anything expresses no preference at all,
        // rather than a small one. Relative scaling has no small preference to
        // give: it maps whatever spread exists onto the whole range.
        let rated = if within_deadband(&measured, scorer.deadband()) {
            measured.iter().map(|m| m.map(|_| 1.0)).collect()
        } else if scorer.already_normalised() {
            rate_directly(&measured, scorer.lower_is_better())
        } else {
            normalise(&measured, scorer.lower_is_better())
        };
        for (index, score) in rated.into_iter().enumerate() {
            if let (Some(score), Some(total), Some(sum)) = (score, totals.get_mut(index), weights.get_mut(index)) {
                *total += score * weight;
                *sum += weight;
            }
        }
    }
    totals
        .into_iter()
        .zip(weights)
        .map(|(total, sum)| (sum > 0.0).then(|| total / sum))
        .collect()
}

/// Pick the highest-scoring candidate, or `None` to keep the overlay's order.
///
/// Unscored candidates are skipped rather than disqualifying the whole set, so a
/// group that mixes observable providers with one that exposes no metrics still
/// balances across the observable ones. An unscored provider is therefore
/// eligible but non-winning: it is never chosen over a provider whose load is
/// known, and it is still reached when it is the only candidate, because a set
/// with no scores at all returns `None` and falls back to the declared order.
///
/// This deliberately replaces the earlier all-or-nothing rule. That rule avoided
/// starving an unobservable provider, but on this grid the unobservable provider
/// is a paid cloud fallback, where losing load-balanced requests is the wanted
/// behaviour and the starvation concern does not apply.
pub(crate) fn pick<'a>(candidates: &[&'a RouteCandidate], scores: &[Option<f64>]) -> Option<&'a RouteCandidate> {
    if candidates.is_empty() || scores.len() != candidates.len() {
        return None;
    }
    let mut best: Option<(&RouteCandidate, f64)> = None;
    for (candidate, score) in candidates
        .iter()
        .zip(scores.iter())
        .filter_map(|(c, s)| s.as_ref().map(|s| (c, s)))
    {
        // Strictly greater, so an earlier candidate wins a tie and the overlay's
        // order still decides where the signal does not.
        if best.is_none_or(|(_, high)| *score > high) {
            best = Some((candidate, *score));
        }
    }
    best.map(|(c, _)| c)
}

/// Build one [`MetricScorer`] per configured signal, all reading `store`.
///
/// Empty when no signal is configured, which leaves selection to the picker.
pub(crate) fn scorers_from(store: &Arc<LoadStore>, signals: &[SignalConfig], max_age_ms: i64) -> Vec<Box<dyn Scorer>> {
    signals
        .iter()
        .map(|s| -> Box<dyn Scorer> {
            Box::new(MetricScorer {
                store: Arc::clone(store),
                metric: s.key.as_str().into(),
                max_age_ms,
                weight: s.weight,
                lower_is_better: s.lower_is_better,
                already_normalised: s.scale == SignalScale::Ratio,
                deadband: s.deadband,
            })
        })
        .collect()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use super::{
        super::descriptor::{AdmissionState, CapabilityKind},
        *,
    };

    pub(super) fn candidate(site: &str, cluster: &str) -> RouteCandidate {
        RouteCandidate {
            admission_state: AdmissionState::NewAndExisting,
            cluster: Arc::from(cluster),
            credential: None,
            fresh: true,
            kind: CapabilityKind::InferenceModel,
            name: Arc::from("llama"),
            rank: None,
            selection_group: None,
            traffic_weight: None,
            selection_tier: None,
            site: Arc::from(site),
            stable_id: Arc::from(format!("inference_model/llama/{site}/{cluster}")),
            load_key: format!("{site}/{cluster}").into(),
        }
    }

    /// Reports the values it was built with, standing in for a live signal.
    struct Fixed {
        values: Vec<Option<f64>>,
        lower_is_better: bool,
        weight: f64,
    }

    impl Scorer for Fixed {
        fn measure(&self, _candidates: &[&RouteCandidate], _now_ms: i64) -> Vec<Option<f64>> {
            self.values.clone()
        }

        fn lower_is_better(&self) -> bool {
            self.lower_is_better
        }

        fn weight(&self) -> f64 {
            self.weight
        }
    }

    fn scorers(values: Vec<Option<f64>>, lower_is_better: bool) -> Vec<Box<dyn Scorer>> {
        vec![Box::new(Fixed {
            values,
            lower_is_better,
            weight: 1.0,
        })]
    }

    #[test]
    fn the_lightest_candidate_wins() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scores = score_all(&scorers(vec![Some(9.0), Some(1.0)], true), &set, 0);
        assert_eq!(
            pick(&set, &scores).map(|c| &*c.cluster),
            Some("b"),
            "lower queue is better"
        );
    }

    #[test]
    fn direction_is_respected() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scores = score_all(&scorers(vec![Some(9.0), Some(1.0)], false), &set, 0);
        assert_eq!(
            pick(&set, &scores).map(|c| &*c.cluster),
            Some("a"),
            "higher is better here"
        );
    }

    #[test]
    fn an_unscored_candidate_is_skipped_not_disqualifying() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scores = score_all(&scorers(vec![None, Some(1.0)], true), &set, 0);
        assert_eq!(
            pick(&set, &scores).map(|c| &*c.cluster),
            Some("b"),
            "one unscored member must not stop the observable ones being balanced"
        );
    }

    #[test]
    fn an_all_unscored_set_keeps_the_overlay_order() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scores = score_all(&scorers(vec![None, None], true), &set, 0);
        assert!(
            pick(&set, &scores).is_none(),
            "with nothing observable there is nothing to rank on, so the declared order stands"
        );
    }

    #[test]
    fn a_lone_unscored_candidate_is_still_reachable() {
        let a = candidate("east", "a");
        let set = [&a];
        let scores = score_all(&scorers(vec![None], true), &set, 0);
        assert!(
            pick(&set, &scores).is_none(),
            "a sole unscored candidate falls back to declared order and is still served"
        );
    }

    #[test]
    fn nothing_observed_leaves_the_overlay_order_alone() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scores = score_all(&scorers(vec![None, None], true), &set, 0);
        assert!(pick(&set, &scores).is_none(), "no signal, no change");
    }

    #[test]
    fn equal_measurements_keep_the_first_candidate() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scores = score_all(&scorers(vec![Some(4.0), Some(4.0)], true), &set, 0);
        assert_eq!(
            pick(&set, &scores).map(|c| &*c.cluster),
            Some("a"),
            "a tie is decided by the overlay, not arbitrarily"
        );
    }

    #[test]
    fn weights_decide_between_disagreeing_signals() {
        let (a, b) = (candidate("east", "a"), candidate("west", "b"));
        let set = [&a, &b];
        let scorers: Vec<Box<dyn Scorer>> = vec![
            Box::new(Fixed {
                values: vec![Some(1.0), Some(9.0)],
                lower_is_better: true,
                weight: 3.0,
            }),
            Box::new(Fixed {
                values: vec![Some(9.0), Some(1.0)],
                lower_is_better: true,
                weight: 1.0,
            }),
        ];
        let scores = score_all(&scorers, &set, 0);
        assert_eq!(
            pick(&set, &scores).map(|c| &*c.cluster),
            Some("a"),
            "the heavier signal decides"
        );
    }

    #[test]
    fn a_score_set_of_the_wrong_length_is_refused() {
        let a = candidate("east", "a");
        let set = [&a];
        assert!(
            pick(&set, &[Some(1.0), Some(2.0)]).is_none(),
            "mismatched lengths cannot be trusted"
        );
    }
}

#[cfg(test)]
mod upstream_parity_tests {
    use super::{normalise, rate_directly};

    // The endpoint picker's multicluster queue scorer computes
    // (maxQ - q) / (maxQ - minQ), and 1.0 when every candidate is equal.
    // Ours has to agree, or a grid routes differently from a pool.
    #[test]
    fn queue_depth_matches_the_endpoint_pickers_min_max_rating() {
        let measured = [Some(2.0), Some(6.0), Some(10.0)];
        let rated = normalise(&measured, true);
        let (min_q, max_q) = (2.0_f64, 10.0_f64);
        for (i, raw) in measured.iter().enumerate() {
            let expected = raw.map(|q| (max_q - q) / (max_q - min_q));
            assert_eq!(rated.get(i).copied().flatten(), expected, "candidate {i}");
        }
    }

    #[test]
    fn an_equal_queue_across_candidates_rates_every_one_at_the_top() {
        let rated = normalise(&[Some(4.0), Some(4.0)], true);
        assert_eq!(rated, vec![Some(1.0), Some(1.0)]);
    }

    // Their KV scorer rates 1 - utilisation, absolute. Min-max scaling would
    // turn a trivial spread into a decisive one.
    #[test]
    fn kv_cache_rates_one_minus_utilisation_rather_than_a_spread() {
        let rated = rate_directly(&[Some(0.10), Some(0.12)], true);
        assert_eq!(rated, vec![Some(0.90), Some(0.88)]);

        let stretched = normalise(&[Some(0.10), Some(0.12)], true);
        assert_eq!(
            stretched,
            vec![Some(1.0), Some(0.0)],
            "min-max would have made two similar pools look opposite"
        );
    }

    #[test]
    fn a_utilisation_outside_the_ratio_is_clamped_not_trusted() {
        assert_eq!(
            rate_directly(&[Some(1.4), Some(-0.2)], true),
            vec![Some(0.0), Some(1.0)]
        );
    }

    #[test]
    fn a_candidate_the_signal_says_nothing_about_stays_unrated() {
        assert_eq!(rate_directly(&[None, Some(0.25)], true), vec![None, Some(0.75)]);
    }
}

#[cfg(test)]
mod deadband_tests {
    use super::{Scorer, score_all, within_deadband};
    use crate::routing::descriptor::RouteCandidate;

    struct Fixed {
        values: Vec<Option<f64>>,
        deadband: f64,
    }

    impl Scorer for Fixed {
        fn measure(&self, _: &[&RouteCandidate], _: i64) -> Vec<Option<f64>> {
            self.values.clone()
        }

        fn lower_is_better(&self) -> bool {
            true
        }

        fn deadband(&self) -> f64 {
            self.deadband
        }
    }

    // The case from a real run: one pool idle at half a queued request, the
    // other at none. Without a deadband that is a total preference for the
    // remote pool, which is a region away and emptier by nothing.
    #[test]
    fn half_a_queued_request_is_not_a_reason_to_leave_the_local_site() {
        assert!(within_deadband(&[Some(0.5), Some(0.0)], 1.0));
    }

    #[test]
    fn a_real_queue_difference_still_decides() {
        assert!(!within_deadband(&[Some(16.5), Some(0.0)], 1.0));
    }

    fn scored(values: Vec<Option<f64>>, deadband: f64) -> Vec<Option<f64>> {
        let local = super::tests::candidate("pool-a", "a");
        let remote = super::tests::candidate("pool-b", "b");
        let candidates = [&local, &remote];
        let scorers: Vec<Box<dyn Scorer>> = vec![Box::new(Fixed { values, deadband })];
        score_all(&scorers, &candidates, 0)
    }

    #[test]
    fn inside_the_deadband_every_candidate_rates_the_same() {
        assert_eq!(
            scored(vec![Some(0.5), Some(0.0)], 1.0),
            vec![Some(1.0), Some(1.0)],
            "no preference, so the overlay order decides and that order is by locality"
        );
    }

    #[test]
    fn outside_it_the_emptier_candidate_wins_outright() {
        assert_eq!(scored(vec![Some(16.5), Some(0.0)], 1.0), vec![Some(0.0), Some(1.0)]);
    }

    #[test]
    fn a_signal_nobody_reports_is_not_a_narrow_spread() {
        assert!(!within_deadband(&[None, None], 1.0));
    }

    #[test]
    fn zero_means_every_difference_counts() {
        assert!(!within_deadband(&[Some(0.5), Some(0.0)], 0.0));
    }
}

/// Replaying a recorded trace of signal scrapes through the real scoring path.
///
/// A tuning question is which weight or deadband produces which decisions, and
/// a multi-cluster run answers it in twenty minutes. The scrapes are the only
/// input scoring has, so a recorded trace answers the same question in
/// milliseconds, and answers it against the parser and store the gateway uses
/// rather than a model of them.
#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod replay {
    use std::sync::Arc;

    use super::{MetricScorer, Scorer, pick, score_all, tests::candidate};
    use crate::routing::load::LoadStore;

    const QUEUE: &str = "llm_d_epp_average_queue_size";
    const KV: &str = "llm_d_epp_average_kv_cache_utilization";

    /// One recorded scrape: the body the endpoint returned, and when.
    struct Scrape {
        at_ms: i64,
        body: String,
    }

    fn trace() -> Vec<Scrape> {
        let raw = include_str!("../../tests/fixtures/signals-trace.jsonl");
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("fixture line is valid json");
                Scrape {
                    at_ms: v.get("at_ms").and_then(serde_json::Value::as_i64).unwrap_or_default(),
                    body: v
                        .get("body")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }
            })
            .collect()
    }

    /// Scorers for one tuning.
    fn tuned(store: &Arc<LoadStore>, queue_deadband: f64, kv_weight: f64, kv_deadband: f64) -> Vec<Box<dyn Scorer>> {
        vec![
            Box::new(MetricScorer {
                store: Arc::clone(store),
                metric: QUEUE.into(),
                max_age_ms: 15_000,
                weight: 1.0,
                lower_is_better: true,
                already_normalised: false,
                deadband: queue_deadband,
            }),
            Box::new(MetricScorer {
                store: Arc::clone(store),
                metric: KV.into(),
                max_age_ms: 15_000,
                weight: kv_weight,
                lower_is_better: true,
                already_normalised: true,
                deadband: kv_deadband,
            }),
        ]
    }

    /// Sites chosen across the whole trace under one tuning.
    fn decisions(queue_deadband: f64, kv_weight: f64, kv_deadband: f64) -> (usize, usize, usize) {
        let store = Arc::new(LoadStore::new(std::time::Duration::from_secs(60)));
        let scorers = tuned(&store, queue_deadband, kv_weight, kv_deadband);
        let local = candidate("pool-a", "llmd-pool-a-provider");
        let remote = candidate("pool-b", "llmd-pool-b-provider");
        let candidates = [&local, &remote];

        let (mut a, mut b, mut none) = (0, 0, 0);
        for scrape in trace() {
            store.ingest(&scrape.body);
            let scores = score_all(&scorers, &candidates, scrape.at_ms);
            match pick(&candidates, &scores) {
                Some(c) if c.site.as_ref() == "pool-a" => a += 1,
                Some(_) => b += 1,
                None => none += 1,
            }
        }
        (a, b, none)
    }

    #[test]
    fn the_trace_replays_through_the_real_parser_and_store() {
        let (a, b, none) = decisions(1.0, 1.0, 0.05);
        assert!(
            a + b > 150,
            "the trace should decide most of its scrapes: {a} local, {b} remote, {none} undecided"
        );
    }

    #[test]
    fn the_deadband_keeps_work_local_that_nothing_justified_moving() {
        let (without, ..) = decisions(0.0, 1.0, 0.05);
        let (with, ..) = decisions(1.0, 1.0, 0.05);
        assert!(
            with > without,
            "a deadband should keep requests local, not move them: {without} local without, {with} with"
        );
    }

    /// Not an assertion. Prints the decision split across a grid of tunings so
    /// a weight can be chosen against a recorded run rather than argued.
    ///
    ///   `cargo test -p praxis-ai-filters --features praxis-main tuning_grid -- --nocapture --ignored`
    #[test]
    #[ignore = "prints a tuning table rather than asserting"]
    #[expect(clippy::print_stdout, reason = "the table is the point of this one")]
    fn tuning_grid() {
        println!("\n  queue_db  kv_weight  kv_db   local  remote  undecided");
        for deadband in [0.0, 1.0, 2.0] {
            for kv_weight in [0.0, 1.0] {
                for kv_deadband in [0.0, 0.02, 0.05] {
                    let (a, b, none) = decisions(deadband, kv_weight, kv_deadband);
                    println!("  {deadband:>8.1} {kv_weight:>10.1} {kv_deadband:>6.2} {a:>7} {b:>7} {none:>10}");
                }
            }
        }
    }

    /// Not an assertion. Prints a per-scrape scorecard so the pool-a to pool-b to
    /// pool-a routing transition is visible in the terminal, driven by the real
    /// scorer over the recorded signal trace.
    ///
    ///   `cargo test -p praxis-ai-filters --features praxis-main replay_scorecard -- --nocapture --ignored`
    #[test]
    #[ignore = "prints a routing scorecard rather than asserting"]
    #[expect(clippy::print_stdout, reason = "the scorecard is the point of this one")]
    fn replay_scorecard() {
        let store = Arc::new(LoadStore::new(std::time::Duration::from_secs(60)));
        let scorers = tuned(&store, 1.0, 1.0, 0.05);
        let local = candidate("pool-a", "llmd-pool-a-provider");
        let remote = candidate("pool-b", "llmd-pool-b-provider");
        let candidates = [&local, &remote];

        println!("\n  LLM-D LOAD-BASED ROUTING  (real scorer, replayed signals)\n");
        println!(
            "  {:>5}   {:>12}  {:>12}     route",
            "t(s)", "pool-a queue", "pool-b queue"
        );
        println!("  {}", "-".repeat(52));

        let start = trace().first().map_or(0, |s| s.at_ms);
        let mut last = String::new();
        for scrape in trace() {
            store.ingest(&scrape.body);
            let scores = score_all(&scorers, &candidates, scrape.at_ms);
            let route = pick(&candidates, &scores).map_or_else(|| "none".to_owned(), |c| c.site.to_string());
            let qa = queue_of(&scrape.body, "pool-a");
            let qb = queue_of(&scrape.body, "pool-b");
            let secs = (scrape.at_ms - start) / 1000;
            let flip = if !last.is_empty() && route != last {
                "   <== flip"
            } else {
                ""
            };
            println!("  {secs:>5}   {qa:>12.1}  {qb:>12.1}     -> {route}{flip}");
            last = route;
        }
        println!();
    }

    /// Read one site's queue-size gauge out of a recorded scrape body, for display.
    fn queue_of(body: &str, site: &str) -> f64 {
        let needle = format!("grid_site=\"{site}\"");
        body.lines()
            .find(|line| line.starts_with(QUEUE) && line.contains(&needle))
            .and_then(|line| line.split_whitespace().nth_back(1))
            .and_then(|value| value.parse().ok())
            .unwrap_or(f64::NAN)
    }
}
