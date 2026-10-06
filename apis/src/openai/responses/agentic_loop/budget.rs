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

/// Request-scoped charge that survives every inference round.
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
    /// Conservative sum of provider payload charges from all rounds.
    output_charge: usize,
    /// Cumulative raw SSE transport charged without retaining transient nodes
    /// from every already-parsed event.
    stream_wire_charge: usize,
    /// Parsed frame structure retained in response state and terminal copies.
    stream_frame_node_charge: usize,
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
            stream_wire_charge: 0,
            stream_frame_node_charge: 0,
            store_output_charge: 0,
            store_response,
            irr_response,
        })
    }

    /// Lower the effective limit when another loop instance is reached.
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

    /// Preflight one more provider chunk before any response parser sees it.
    pub(crate) fn admit_output(&mut self, bytes: &[u8]) -> bool {
        let Some(additional) = output_charge(bytes) else {
            return false;
        };
        let Some(next_output) = self.output_charge.checked_add(additional) else {
            return false;
        };
        let Some(next_store) = self
            .store_output_charge
            .checked_add(if self.store_response { additional } else { 0 })
        else {
            return false;
        };
        let Some(total) = next_output
            .checked_add(self.stream_wire_charge)
            .and_then(|charge| charge.checked_add(self.stream_frame_node_charge))
            .and_then(|charge| charge.checked_add(next_store))
            .and_then(|charge| charge.checked_add(self.input_charge))
            .and_then(|charge| charge.checked_add(self.additional_input_charge))
            .and_then(|charge| charge.checked_add(self.store_input_charge))
            .and_then(|charge| charge.checked_add(self.transport_reserve()?))
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
        let Some(next_wire) = self.stream_wire_charge.checked_add(wire_delta) else {
            return false;
        };
        let Some(store_delta) = bytes.len().checked_mul(if self.store_response {
            STREAM_STORE_WIRE_MULTIPLIER
        } else {
            0
        }) else {
            return false;
        };
        let Some(next_store) = self.store_output_charge.checked_add(store_delta) else {
            return false;
        };
        let Some(total) = self
            .charge()
            .and_then(|charge| charge.checked_add(wire_delta))
            .and_then(|charge| charge.checked_add(store_delta))
            .and_then(|charge| charge.checked_add(peak))
        else {
            return false;
        };
        if total > self.limit {
            return false;
        }
        self.stream_wire_charge = next_wire;
        self.store_output_charge = next_store;
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
        let Some(next) = self.stream_frame_node_charge.checked_add(delta) else {
            return false;
        };
        if self
            .charge()
            .and_then(|charge| charge.checked_add(delta))
            .is_none_or(|total| total > self.limit)
        {
            return false;
        }
        self.stream_frame_node_charge = next;
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

    /// Return the current charge including any IRR transport reserve.
    fn charge(self) -> Option<usize> {
        self.output_charge
            .checked_add(self.stream_wire_charge)?
            .checked_add(self.stream_frame_node_charge)?
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
}
