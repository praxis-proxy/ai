// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conservative request-wide retained-payload accounting for Responses.
//!
//! The shared ledger covers input, output, streaming, Store, and owner-specific
//! expansions across inference rounds. Owners reserve capacity before making
//! independent payload copies and settle transient reservations after use.

/// Copies and parsing capacity held while classifying and validating input.
pub(super) const INPUT_WIRE_MULTIPLIER: usize = 32;
/// The router buffer is reserved at one eighth of the configured ceiling.
pub(super) const IRR_RESPONSE_DIVISOR: usize = 8;
/// Copies and serialization capacity held while processing model output.
const OUTPUT_WIRE_MULTIPLIER: usize = 64;
/// Reserve for each JSON value/key, including collection spare capacity.
const JSON_NODE_RESERVE: usize = 256;
/// Parsed provider values may own a four-slot Vec or map for only a few wire
/// bytes. Leave room for those allocations and their response-state copies.
const OUTPUT_NODE_RESERVE: usize = 8_192;
/// Retained projections of each wire chunk across the logical stream.
const STREAM_WIRE_MULTIPLIER: usize = 32;
/// Additional retained replay and response projections when Store is active.
const STREAM_STORE_WIRE_MULTIPLIER: usize = 32;
/// Simultaneously live SSE frame and parsed-event wire projections.
const STREAM_PEAK_WIRE_MULTIPLIER: usize = 16;
/// Retained parsed JSON nodes from completed SSE frames and response copies.
const STREAM_FRAME_NODE_RESERVE: usize = 512;
/// Store builds another response-object projection from parsed terminal state.
const STREAM_STORE_FRAME_NODE_RESERVE: usize = 256;
/// Parsed output items a completed round leaves live in `accumulated_output` and
/// the replayed message history. Measured on the serialized items, so it covers
/// the accumulator copy, the next round's history copy, and their `serde_json`
/// expansion. Smaller than the round-local provider charge so releasing a round
/// nets headroom back while the kept output still accumulates across rounds.
#[cfg(any(test, feature = "openai-agentic-loop"))]
const RETAINED_OUTPUT_MULTIPLIER: usize = 32;
/// Store keeps one more persisted projection of that retained output.
#[cfg(any(test, feature = "openai-agentic-loop"))]
const RETAINED_STORE_OUTPUT_MULTIPLIER: usize = 16;

/// Request-scoped charge that survives every inference round.
///
/// The ledger separates two lifetimes. **Retained** owners live for the whole
/// request and accumulate across rounds: the request body, restored history,
/// tool results, and the output items kept in `accumulated_output`. **Round-local**
/// owners exist only while one inference round is parsed — the buffered provider
/// body, raw SSE transport, and parsed frame nodes — and are released at the
/// round boundary once their payload has been reduced to retained state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimpleBudget {
    /// The smallest configured limit of every reachable loop instance.
    limit: usize,
    /// Input charge retained throughout execution.
    input_charge: usize,
    /// Additional request-side owners admitted before their payload is copied.
    additional_input_charge: usize,
    /// Additional allowance for the response Store's request input snapshot.
    store_input_charge: usize,
    /// Retained output kept across rounds: items held in `accumulated_output` and
    /// the replayed history, plus the terminal response projection. Grows as each
    /// round's surviving output is retained; it is never released.
    output_charge: usize,
    /// The current round's transient provider representations: the buffered
    /// response body, raw SSE transport, and parsed frame nodes. Their owners are
    /// dropped when the round completes, so this is released at the round boundary
    /// by [`SimpleBudget::retain_round_output`] instead of accumulating.
    round_local_charge: usize,
    /// Separate allowance for Store's persisted response and history copies.
    store_output_charge: usize,
    /// Whether Store persists this create response and owns another output projection.
    store_response: bool,
    /// Whether this response enters IRR's separately buffered transport.
    irr_response: bool,
}

impl SimpleBudget {
    /// Headroom after all request-wide reserves, including the transport.
    pub(crate) fn remaining_bytes(self) -> Option<usize> {
        self.limit.checked_sub(self.charge()?)
    }

    /// Admit a request after the allocation-free ingress scan.
    #[cfg(test)]
    pub(crate) fn new(limit: usize, input_charge: usize) -> Option<Self> {
        Self::new_with_store(limit, input_charge, false)
    }

    /// Admit a create that will retain Store input and response projections.
    pub(crate) fn new_with_store(limit: usize, input_charge: usize, store_response: bool) -> Option<Self> {
        Self::with_transport(limit, input_charge, store_response, true)
    }

    /// Admit a locally returned response that never enters IRR's buffer.
    #[cfg(feature = "openai-compact")]
    pub(crate) fn new_for_local_response(limit: usize, input_charge: usize) -> Option<Self> {
        Self::with_transport(limit, input_charge, false, false)
    }

    /// Initialize the same ledger for routed and locally returned responses.
    fn with_transport(limit: usize, input_charge: usize, store_response: bool, irr_response: bool) -> Option<Self> {
        let store_input_charge = if store_response { input_charge } else { 0 };
        // Core's response Vec may keep spare capacity alongside the live raw
        // body. Reserve three times its validated `limit / 8` transport cap
        // only when this request actually enters the router.
        if input_charge
            .checked_add(store_input_charge)?
            .checked_add(if irr_response { response_reserve(limit)? } else { 0 })?
            > limit
        {
            return None;
        }
        Some(Self {
            limit,
            input_charge,
            additional_input_charge: 0,
            store_input_charge,
            output_charge: 0,
            round_local_charge: 0,
            store_output_charge: 0,
            store_response,
            irr_response,
        })
    }

    /// Lower the effective limit when another loop instance is reached.
    #[cfg(any(test, feature = "openai-agentic-loop"))]
    pub(crate) fn lower_limit(&mut self, limit: usize) -> bool {
        self.limit = self.limit.min(limit);
        self.charge().is_some_and(|charge| charge <= self.limit)
    }

    /// Reserve an independently owned request-side payload before allocating it.
    /// Failed reservations leave the previous charge intact.
    pub(crate) fn reserve_additional_input(&mut self, charge: usize) -> bool {
        let Some(next) = self.additional_input_charge.checked_add(charge) else {
            return false;
        };
        let Some(total) = self.charge().and_then(|total| total.checked_add(charge)) else {
            return false;
        };
        if total > self.limit {
            return false;
        }
        self.additional_input_charge = next;
        true
    }

    /// Release transient admission after its owners are dropped, leaving the
    /// measured independently retained portion charged to this request.
    pub(crate) fn settle_additional_input(&mut self, reserved: usize, retained: usize) -> bool {
        let Some(released) = reserved.checked_sub(retained) else {
            return false;
        };
        let Some(next) = self.additional_input_charge.checked_sub(released) else {
            return false;
        };
        self.additional_input_charge = next;
        true
    }

    /// Reserve the live item clones, event envelopes, and encoded SSE bytes
    /// before a local tool lifecycle is synthesized. One item can appear in
    /// opening, progress, done, and Store event-log owners at the same time.
    pub(crate) fn admit_stream_synthesis(&mut self, item_bytes: usize, event_count: usize) -> bool {
        let Some(factor) = 16_usize.checked_add(if self.store_response { 8 } else { 0 }) else {
            return false;
        };
        let Some(charge) = item_bytes
            .checked_mul(factor)
            .and_then(|bytes| event_count.checked_mul(1_024)?.checked_add(bytes))
        else {
            return false;
        };
        self.reserve_additional_input(charge)
    }

    /// Preflight one buffered provider response before any response parser sees
    /// it. The raw body, its parse scratch, and the per-round Store projection are
    /// dropped once the round is reduced to retained state, so the charge is
    /// round-local and released at the round boundary by
    /// [`SimpleBudget::retain_round_output`].
    #[cfg(any(test, feature = "openai-agentic-loop"))]
    pub(crate) fn admit_output(&mut self, bytes: &[u8]) -> bool {
        let Some(additional) = output_charge(bytes) else {
            return false;
        };
        let Some(round_delta) = additional.checked_add(if self.store_response { additional } else { 0 }) else {
            return false;
        };
        let Some(next_round_local) = self.round_local_charge.checked_add(round_delta) else {
            return false;
        };
        let Some(total) = self
            .retained_charge()
            .and_then(|charge| charge.checked_add(next_round_local))
        else {
            return false;
        };
        if total > self.limit {
            return false;
        }
        self.round_local_charge = next_round_local;
        true
    }

    /// Reserve one raw SSE chunk before frame assembly and event parsing.
    ///
    /// Streaming owns the raw chunk, partial frame, parsed frame/event, logical
    /// output, and (when enabled) Store projections at overlapping points.
    /// Retained wire projections accumulate across chunks and rounds. Parser
    /// and event wire copies live only through this chunk, so their reserve
    /// is checked as a transient peak. Parsed nodes are charged after the
    /// complete frame is assembled, independent of transport chunk splits.
    /// A rejected chunk leaves the charge unchanged; the stream owner must
    /// poison the logical stream.
    #[expect(
        clippy::too_many_lines,
        reason = "one checked reservation covers retained and peak stream owners"
    )]
    pub(crate) fn admit_stream_chunk(&mut self, bytes: &[u8]) -> bool {
        let Some(peak) = stream_peak_charge(bytes) else {
            return false;
        };
        let Some(wire_delta) = bytes.len().checked_mul(STREAM_WIRE_MULTIPLIER) else {
            return false;
        };
        let Some(store_delta) = bytes.len().checked_mul(if self.store_response {
            STREAM_STORE_WIRE_MULTIPLIER
        } else {
            0
        }) else {
            return false;
        };
        // The raw transport chunk and its per-round Store replay projection are
        // round-local; the parser and event wire copies live only through this
        // chunk, so their reserve is checked as a transient peak on top.
        let Some(next_round_local) = self
            .round_local_charge
            .checked_add(wire_delta)
            .and_then(|charge| charge.checked_add(store_delta))
        else {
            return false;
        };
        let Some(total) = self
            .retained_charge()
            .and_then(|charge| charge.checked_add(next_round_local))
            .and_then(|charge| charge.checked_add(peak))
        else {
            return false;
        };
        if total > self.limit {
            return false;
        }
        self.round_local_charge = next_round_local;
        true
    }

    /// Preflight a complete reassembled frame before JSON deserialization.
    /// Chunk-local lexical scans cannot see a compact tree split over many
    /// chunks, so only the complete `data:` payload can bound parsed nodes.
    pub(crate) fn admit_stream_frame(&mut self, data: &[u8]) -> bool {
        let Some(nodes) = json_node_count(data) else {
            return false;
        };
        let Some(per_node) = STREAM_FRAME_NODE_RESERVE.checked_add(if self.store_response {
            STREAM_STORE_FRAME_NODE_RESERVE
        } else {
            0
        }) else {
            return false;
        };
        let Some(delta) = nodes.checked_mul(per_node) else {
            return false;
        };
        // Parsed frame nodes are reduced to retained output items at the round
        // boundary, so they are charged round-local here.
        let Some(next_round_local) = self.round_local_charge.checked_add(delta) else {
            return false;
        };
        if self
            .retained_charge()
            .and_then(|charge| charge.checked_add(next_round_local))
            .is_none_or(|total| total > self.limit)
        {
            return false;
        }
        self.round_local_charge = next_round_local;
        true
    }

    /// Reserve terminal wire encoding and Store copy capacity before the
    /// accumulated output moves into the canonical response object.
    pub(crate) fn admit_stream_terminal(&mut self, serialized_output_bytes: usize) -> bool {
        let Some(output_delta) = serialized_output_bytes.checked_mul(4) else {
            return false;
        };
        let Some(store_delta) = serialized_output_bytes.checked_mul(if self.store_response { 2 } else { 0 }) else {
            return false;
        };
        let Some(next_output) = self.output_charge.checked_add(output_delta) else {
            return false;
        };
        let Some(next_store) = self.store_output_charge.checked_add(store_delta) else {
            return false;
        };
        let Some(total) = self
            .charge()
            .and_then(|charge| charge.checked_add(output_delta))
            .and_then(|charge| charge.checked_add(store_delta))
        else {
            return false;
        };
        if total > self.limit {
            return false;
        }
        self.output_charge = next_output;
        self.store_output_charge = next_store;
        true
    }

    /// Retain a completed round's surviving output and release its round-local
    /// provider representations.
    ///
    /// `output_bytes` is the serialized size of the items the round appended to
    /// `accumulated_output`; those stay live in the accumulator and the replayed
    /// history, so they move into the retained ledger. The buffered body, raw SSE
    /// transport, and parsed frame nodes that produced them are now dropped, so
    /// the round-local charge is released. A later round reuses that headroom
    /// while the retained output still accumulates toward the limit.
    ///
    /// Returns `false` on arithmetic overflow or if the accumulated retained
    /// output no longer fits the limit — the caller treats that as exhaustion.
    /// The ledger is left unchanged on failure.
    #[cfg(any(test, feature = "openai-agentic-loop"))]
    pub(crate) fn retain_round_output(&mut self, output_bytes: usize) -> bool {
        let Some(retained) = output_bytes.checked_mul(RETAINED_OUTPUT_MULTIPLIER) else {
            return false;
        };
        let Some(store_retained) = output_bytes.checked_mul(if self.store_response {
            RETAINED_STORE_OUTPUT_MULTIPLIER
        } else {
            0
        }) else {
            return false;
        };
        let Some(next_output) = self.output_charge.checked_add(retained) else {
            return false;
        };
        let Some(next_store) = self.store_output_charge.checked_add(store_retained) else {
            return false;
        };
        let candidate = Self {
            output_charge: next_output,
            store_output_charge: next_store,
            round_local_charge: 0,
            ..*self
        };
        if candidate.charge().is_none_or(|total| total > self.limit) {
            return false;
        }
        *self = candidate;
        true
    }

    /// Return the current charge including the live round-local reserve.
    fn charge(self) -> Option<usize> {
        self.retained_charge()?.checked_add(self.round_local_charge)
    }

    /// Return the request-wide retained charge, excluding the round-local reserve
    /// that `retain_round_output` releases at each round boundary.
    fn retained_charge(self) -> Option<usize> {
        self.output_charge
            .checked_add(self.store_output_charge)?
            .checked_add(self.input_charge)?
            .checked_add(self.additional_input_charge)?
            .checked_add(self.store_input_charge)?
            .checked_add(self.transport_reserve()?)
    }

    /// Core holds an IRR response buffer only for routed requests.
    fn transport_reserve(self) -> Option<usize> {
        if self.irr_response {
            response_reserve(self.limit)
        } else {
            Some(0)
        }
    }
}

/// Peak temporary copies while one chunk is framed and parsed. This remains
/// separate from the cumulative retained wire charge above.
fn stream_peak_charge(bytes: &[u8]) -> Option<usize> {
    bytes.len().checked_mul(STREAM_PEAK_WIRE_MULTIPLIER)
}

/// Account for core's buffered response capacity before parsing or copying.
fn response_reserve(limit: usize) -> Option<usize> {
    (limit / IRR_RESPONSE_DIVISOR).checked_mul(3)
}

/// Bound the simultaneously live raw and parsed create-body projections.
/// This scans borrowed bytes and allocates no request payload.
pub(crate) fn input_charge(bytes: &[u8]) -> Option<usize> {
    bytes
        .len()
        .checked_mul(INPUT_WIRE_MULTIPLIER)?
        .checked_add(json_node_count(bytes)?.checked_mul(JSON_NODE_RESERVE)?)
}

/// Charge provider structure before its first JSON parser allocates.
pub(crate) fn output_charge(bytes: &[u8]) -> Option<usize> {
    bytes
        .len()
        .checked_mul(OUTPUT_WIRE_MULTIPLIER)?
        .checked_add(json_node_count(bytes)?.checked_mul(OUTPUT_NODE_RESERVE)?)
}

/// Count syntactic JSON nodes without allocating or trusting the payload.
#[expect(
    clippy::too_many_lines,
    reason = "the lexical scan keeps string and number state without allocating"
)]
fn json_node_count(bytes: &[u8]) -> Option<usize> {
    let mut nodes = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut in_number = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => {
                nodes = nodes.checked_add(1)?;
                in_string = true;
                in_number = false;
            },
            b'{' | b'[' | b't' | b'f' | b'n' => {
                nodes = nodes.checked_add(1)?;
                in_number = false;
            },
            b'-' | b'0'..=b'9' if !in_number => {
                nodes = nodes.checked_add(1)?;
                in_number = true;
            },
            b'-' | b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' if in_number => {},
            _ => in_number = false,
        }
    }
    Some(nodes)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "unit tests construct validated budgets")]
mod tests {
    use super::*;

    #[test]
    fn cumulative_output_and_lower_limit() {
        assert!(input_charge(br#"{"input":"hello","store":false}"#).is_some());
        let mut budget = SimpleBudget::new(4_096, 192).unwrap();
        assert!(budget.admit_output(&[b' '; 37]));
        assert!(!budget.admit_output(&[b' '; 1]));
        assert!(budget.lower_limit(4_096));
        assert!(!budget.lower_limit(4_092));
    }

    #[test]
    fn restored_owner_reduces_output_headroom() {
        let mut budget = SimpleBudget::new(8_388_608, 1_000).unwrap();
        let before = budget.remaining_bytes().unwrap();
        assert!(budget.reserve_additional_input(128_000));
        assert_eq!(budget.remaining_bytes(), Some(before - 128_000));
    }

    #[test]
    #[cfg(feature = "openai-compact")]
    fn local_response_omits_only_irr_transport_reserve() {
        let limit = 4_096;
        let input = 100;
        let mut routed = SimpleBudget::new_with_store(limit, input, false).unwrap();
        let mut local = SimpleBudget::new_for_local_response(limit, input).unwrap();
        assert_eq!(
            local.remaining_bytes().unwrap() - routed.remaining_bytes().unwrap(),
            response_reserve(limit).unwrap(),
            "local responses do not use IRR's buffered response"
        );
        assert!(
            local.reserve_additional_input(3_000),
            "local owner must use freed headroom"
        );
        assert!(
            !routed.reserve_additional_input(3_000),
            "routed responses must retain their transport reserve"
        );
    }

    #[test]
    fn store_reserves_independent_input_and_output_projections() {
        assert!(SimpleBudget::new(4_096, 1_500).is_some());
        assert!(SimpleBudget::new_with_store(4_096, 1_500, true).is_none());

        let mut plain = SimpleBudget::new(4_096, 100).unwrap();
        let mut stored = SimpleBudget::new_with_store(4_096, 100, true).unwrap();
        assert!(plain.admit_output(&[b' '; 37]));
        assert!(!stored.admit_output(&[b' '; 37]));
    }

    #[test]
    fn additional_input_reservation_is_cumulative_and_atomic() {
        let mut budget = SimpleBudget::new(4_096, 100).unwrap();
        assert!(budget.reserve_additional_input(2_000));
        assert!(!budget.reserve_additional_input(1_000));
        assert!(budget.reserve_additional_input(400));
        assert!(!budget.lower_limit(3_996));
        assert!(!budget.reserve_additional_input(usize::MAX));
    }

    #[test]
    fn split_stream_chunks_are_cumulative_and_failed_charge_is_atomic() {
        let mut budget = SimpleBudget::new(65_536, 100).unwrap();
        assert!(budget.admit_stream_chunk(b"data: first\n\n"));
        let accepted = budget.charge();
        assert!(!budget.admit_stream_chunk(&[b'x'; 1_000]));
        assert_eq!(budget.charge(), accepted);
        assert!(budget.admit_stream_chunk(b"x"));
    }

    #[test]
    fn output_frame_fails_after_committing_created_event_with_ledger_charge() {
        const LIMIT: usize = 131_072;
        const REQUEST: &[u8] = br#"{"model":"gpt-4.1","input":"Reply briefly.","stream":true}"#;
        const CREATED_WIRE: &[u8] = b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-4.1\",\"status\":\"in_progress\",\"output\":[]}}\n\n";
        const CREATED_DATA: &[u8] = b"{\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-4.1\",\"status\":\"in_progress\",\"output\":[]}}";

        let mut budget = SimpleBudget::new(LIMIT, input_charge(REQUEST).unwrap()).unwrap();
        assert!(budget.admit_stream_chunk(CREATED_WIRE));
        assert!(budget.admit_stream_frame(CREATED_DATA));
        let committed_charge = budget.charge();

        let delta_wire = format!(
            "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"delta\":\"{}\"}}\n\n",
            "x".repeat(16_384)
        );
        assert!(
            delta_wire.len() < LIMIT,
            "the raw event remains below the configured ceiling"
        );
        assert!(
            !budget.admit_stream_chunk(delta_wire.as_bytes()),
            "weighted wire and transient parsing charges must reject the output frame"
        );
        assert_eq!(
            budget.charge(),
            committed_charge,
            "a rejected frame must leave the previously committed charge unchanged"
        );
    }

    #[test]
    fn local_sse_synthesis_admits_at_ceiling_and_fails_atomically_above_it() {
        let mut budget = SimpleBudget::new(65_536, 0).unwrap();
        assert_eq!(budget.remaining_bytes(), Some(40_960));
        assert!(budget.admit_stream_synthesis(0, 40));
        assert_eq!(budget.remaining_bytes(), Some(0));
        assert!(!budget.admit_stream_synthesis(1, 0));
        assert_eq!(budget.remaining_bytes(), Some(0));
    }

    #[test]
    fn stream_chunk_at_peak_ceiling_is_admitted_but_next_frame_is_not() {
        let wire = b"data: {}\n\n";
        let limit = 65_536;
        let wire_charge = wire.len() * STREAM_WIRE_MULTIPLIER;
        let input = limit - response_reserve(limit).unwrap() - stream_peak_charge(wire).unwrap() - wire_charge;
        let mut budget = SimpleBudget::new(limit, input).unwrap();
        assert!(budget.admit_stream_chunk(wire));
        assert_eq!(budget.charge(), Some(limit - stream_peak_charge(wire).unwrap()));
        assert!(!budget.admit_stream_chunk(wire));
    }

    #[test]
    fn many_small_stream_events_do_not_retain_each_parsers_node_peak() {
        let mut budget = SimpleBudget::new(67_108_864, 1_000).unwrap();
        let event =
            b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n";
        let frame = b"{\"type\":\"response.output_text.delta\",\"delta\":\"x\"}";
        for _ in 0..1_000 {
            assert!(budget.admit_stream_chunk(event));
            assert!(budget.admit_stream_frame(frame));
        }
        assert!(budget.charge().unwrap() < 40_000_000);

        let batch = event.repeat(1_000);
        let mut batch_budget = SimpleBudget::new(67_108_864, 1_000).unwrap();
        assert!(batch_budget.admit_stream_chunk(&batch));
        for _ in 0..1_000 {
            assert!(batch_budget.admit_stream_frame(frame));
        }
    }

    #[test]
    fn stored_stream_charges_replay_projection_separately() {
        let wire = vec![b'x'; 600];
        let mut plain = SimpleBudget::new(65_536, 100).unwrap();
        let mut stored = SimpleBudget::new_with_store(65_536, 100, true).unwrap();
        assert!(plain.admit_stream_chunk(&wire));
        assert!(!stored.admit_stream_chunk(&wire));
    }

    #[test]
    fn additional_input_owner_reduces_available_stream_budget() {
        let wire = vec![b'x'; 500];
        let mut plain = SimpleBudget::new(65_536, 100).unwrap();
        let mut translated = SimpleBudget::new(65_536, 100).unwrap();
        assert!(plain.admit_stream_chunk(&wire));
        assert!(translated.reserve_additional_input(30_000));
        assert!(!translated.admit_stream_chunk(&wire));
    }

    #[test]
    fn terminal_copy_reservation_fails_atomically() {
        let mut budget = SimpleBudget::new_with_store(65_536, 100, true).unwrap();
        let prior = budget.charge();
        assert!(!budget.admit_stream_terminal(100_000));
        assert_eq!(budget.charge(), prior);
        assert!(budget.admit_stream_terminal(100));
        assert_eq!(budget.charge(), Some(prior.unwrap() + 600));
    }

    #[test]
    fn complete_frame_preflight_catches_nested_tree_split_across_chunks() {
        let item = format!("{}0{}", "[".repeat(20), "]".repeat(20));
        let data = format!(
            r#"{{"output":[{}]}}"#,
            std::iter::repeat_n(item.as_str(), 12_000).collect::<Vec<_>>().join(",")
        );
        let mut budget = SimpleBudget::new(67_108_864, 2_272).unwrap();
        for chunk in data.as_bytes().chunks(4_096) {
            assert!(budget.admit_stream_chunk(chunk));
        }
        let prior = budget.charge();
        assert!(!budget.admit_stream_frame(data.as_bytes()));
        assert_eq!(budget.charge(), prior);
    }

    #[test]
    fn frame_split_inside_json_string_has_same_admission_as_whole_frame() {
        let data = format!(
            r#"{{"type":"response.output_text.delta","delta":"{}"}}"#,
            "n".repeat(32_768)
        );
        let wire = format!("event: response.output_text.delta\ndata: {data}\n\n");
        let mut whole = SimpleBudget::new(67_108_864, 1_000).unwrap();
        assert!(whole.admit_stream_chunk(wire.as_bytes()));
        assert!(whole.admit_stream_frame(data.as_bytes()));

        let mut split = SimpleBudget::new(67_108_864, 1_000).unwrap();
        for chunk in wire.as_bytes().chunks(83) {
            assert!(split.admit_stream_chunk(chunk));
        }
        assert!(split.admit_stream_frame(data.as_bytes()));
        assert_eq!(split.charge(), whole.charge());
    }

    #[test]
    fn checked_arithmetic_fails_closed() {
        let mut budget = SimpleBudget::new(usize::MAX, 0).unwrap();
        budget.output_charge = usize::MAX;
        assert!(!budget.admit_output(b" "));
    }

    #[test]
    fn compact_nested_provider_output_is_rejected_before_parse() {
        let nested = format!("{}0{}", "[".repeat(20), "]".repeat(20));
        let values = std::iter::repeat_n(nested.as_str(), 16_000)
            .collect::<Vec<_>>()
            .join(",");
        let body = format!(
            r#"{{"object":"response","status":"completed","output":[{{"type":"message","content":[{{"type":"output_text","text":"hello"}}],"extra":[{values}]}}]}}"#
        );
        let mut budget = SimpleBudget::new(67_108_864, 2_272).unwrap();
        assert!(body.len() * OUTPUT_WIRE_MULTIPLIER < budget.limit);
        assert!(!budget.admit_output(body.as_bytes()));
    }

    #[test]
    fn released_buffered_round_reuses_headroom() {
        // Each round admits a large provider body, then retains only the small
        // output that survives into `accumulated_output` and releases the body
        // charge. Ten modest rounds must all admit because the body is reclaimed.
        let limit = 8_388_608;
        let body = vec![b'x'; 40 * 1024];
        let mut budget = SimpleBudget::new(limit, 1_000).unwrap();
        for _ in 0..10 {
            assert!(budget.admit_output(&body), "round body admits");
            assert!(budget.retain_round_output(1_024), "round retained and released");
        }
        assert!(budget.remaining_bytes().unwrap() > 0, "released rounds leave headroom");

        // Without the release, the accumulated per-round body charges exhaust the
        // same budget well before ten rounds.
        let mut no_release = SimpleBudget::new(limit, 1_000).unwrap();
        let admitted = (0..10).take_while(|_| no_release.admit_output(&body)).count();
        assert!(admitted < 10, "accumulating every round's body exhausts the budget");
    }

    #[test]
    fn released_streaming_round_reuses_headroom() {
        // The streaming analogue: raw SSE transport and parsed frames are
        // round-local and reclaimed each round once the output is retained.
        let limit = 8_388_608;
        let chunk = vec![b'x'; 32 * 1024];
        let mut budget = SimpleBudget::new(limit, 1_000).unwrap();
        for _ in 0..10 {
            assert!(budget.admit_stream_chunk(&chunk), "chunk admits");
            assert!(budget.admit_stream_frame(br#"{"delta":"x"}"#), "frame admits");
            assert!(budget.retain_round_output(1_024), "round retained and released");
        }
        assert!(
            budget.remaining_bytes().unwrap() > 0,
            "released stream rounds leave headroom"
        );
    }

    #[test]
    fn retained_output_accumulates_to_ceiling() {
        // Retained output is never released, so repeated rounds that keep real
        // output eventually exhaust the budget even though each body is reclaimed.
        let limit = 2_097_152;
        let body = vec![b'x'; 8 * 1024];
        let mut budget = SimpleBudget::new(limit, 1_000).unwrap();
        let mut rounds = 0;
        while rounds <= 1_000 {
            if !budget.admit_output(&body) || !budget.retain_round_output(8 * 1024) {
                break;
            }
            rounds += 1;
        }
        assert!(rounds > 0, "some rounds admit");
        assert!(rounds <= 1_000, "retained output eventually hits the ceiling");
    }

    #[test]
    fn store_round_reuses_headroom_and_retains_both_projections() {
        // Store doubles both the round-local body charge and the retained output,
        // so headroom reuse still holds and the stored projection accumulates.
        let limit = 16_777_216;
        let body = vec![b'x'; 40 * 1024];
        let mut stored = SimpleBudget::new_with_store(limit, 1_000, true).unwrap();
        let mut plain = SimpleBudget::new_with_store(limit, 1_000, false).unwrap();
        for _ in 0..8 {
            assert!(stored.admit_output(&body) && stored.retain_round_output(1_024));
            assert!(plain.admit_output(&body) && plain.retain_round_output(1_024));
        }
        assert!(
            stored.remaining_bytes().unwrap() < plain.remaining_bytes().unwrap(),
            "the stored projection accumulates additional retained output"
        );
    }

    #[test]
    fn aborted_round_keeps_round_local_until_retained() {
        // A round that never reaches the retain boundary (aborted or terminal)
        // keeps its round-local charge; only an explicit retain releases it, so the
        // ledger never silently drops a still-live owner.
        let mut budget = SimpleBudget::new(8_388_608, 1_000).unwrap();
        let body = vec![b'x'; 16 * 1024];
        assert!(budget.admit_output(&body));
        let during_round = budget.remaining_bytes().unwrap();
        assert_eq!(budget.remaining_bytes().unwrap(), during_round, "no implicit release");
        assert!(budget.retain_round_output(1_024));
        assert!(
            budget.remaining_bytes().unwrap() > during_round,
            "retaining the round releases the round-local body charge"
        );
    }

    #[test]
    fn lower_limit_sees_retained_and_round_local_charges() {
        let body = vec![b'x'; 16 * 1024];
        // Mid-round, the live charge a later loop instance checks includes the
        // round-local provider body. (`lower_limit` mutates the limit even when it
        // rejects, so each case uses a fresh ledger.)
        let mut mid_round = SimpleBudget::new(8_388_608, 1_000).unwrap();
        assert!(mid_round.admit_output(&body));
        assert!(mid_round.lower_limit(8_388_608), "the same limit still fits");

        let mut mid_round_reject = SimpleBudget::new(8_388_608, 1_000).unwrap();
        assert!(mid_round_reject.admit_output(&body));
        assert!(
            !mid_round_reject.lower_limit(1_000),
            "a limit below the live round-local charge is rejected"
        );

        // After the round is retained, the round-local body is released, but the
        // retained output still exceeds a tiny limit.
        let mut retained = SimpleBudget::new(8_388_608, 1_000).unwrap();
        assert!(retained.admit_output(&body));
        assert!(retained.retain_round_output(16 * 1024));
        assert!(
            !retained.lower_limit(1_000),
            "retained output still exceeds a tiny limit"
        );
    }
}
