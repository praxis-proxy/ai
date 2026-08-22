// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::borrow::Cow;

use http::{HeaderMap, HeaderValue, Method};

use super::*;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Request ID resolution
// -----------------------------------------------------------------------------

#[test]
fn prefers_client_supplied_request_id() {
    let req = request_with(&[("x-request-id", "client-abc")]);
    let ctx = make_filter_context(&req);

    let map = applied(&Correlation::from_filter_context(&ctx));

    assert_eq!(
        header(&map, "x-request-id"),
        "client-abc",
        "client-supplied request ID should be forwarded unchanged"
    );
}

#[test]
fn uses_id_injected_by_request_id_filter() {
    let req = request_with(&[]);
    let mut ctx = make_filter_context(&req);
    // The request_id core builtin injects here, not into
    // ctx.request.headers — the case a downstream-headers-only
    // lookup would miss.
    ctx.extra_request_headers
        .push((Cow::Borrowed("X-Request-ID"), "generated-by-filter".to_owned()));

    let map = applied(&Correlation::from_filter_context(&ctx));

    assert_eq!(
        header(&map, "x-request-id"),
        "generated-by-filter",
        "injected request ID should be picked up from extra_request_headers"
    );
}

#[test]
fn generates_request_id_when_no_source_available() {
    let req = request_with(&[]);
    let ctx = make_filter_context(&req);

    let map = applied(&Correlation::from_filter_context(&ctx));
    let id = header(&map, "x-request-id");

    assert_eq!(id.len(), 32, "generated request ID should be 32 hex chars: {id}");
    assert!(is_lower_hex(&id), "generated request ID should be lowercase hex: {id}");
}

// -----------------------------------------------------------------------------
// Traceparent
// -----------------------------------------------------------------------------

#[test]
fn continues_valid_inbound_trace_under_new_span() {
    let inbound = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let req = request_with(&[("traceparent", inbound)]);
    let ctx = make_filter_context(&req);

    let map = applied(&Correlation::from_filter_context(&ctx));
    let outbound = header(&map, "traceparent");
    let parts: Vec<&str> = outbound.split('-').collect();

    assert_eq!(parts.len(), 4, "traceparent should have four fields: {outbound}");
    assert_eq!(
        parts[1], "4bf92f3577b34da6a3ce929d0e0e4736",
        "trace-id should be carried forward"
    );
    assert_eq!(parts[3], "01", "inbound trace flags should be preserved");
    assert_ne!(
        parts[2], "00f067aa0ba902b7",
        "delegation hop should emit its own span-id"
    );
    assert_eq!(parts[2].len(), 16, "span-id should be 16 hex chars: {outbound}");
}

#[test]
fn starts_new_trace_when_absent() {
    let req = request_with(&[]);
    let ctx = make_filter_context(&req);

    let map = applied(&Correlation::from_filter_context(&ctx));
    let outbound = header(&map, "traceparent");
    let parts: Vec<&str> = outbound.split('-').collect();

    assert_eq!(parts.len(), 4, "traceparent should have four fields: {outbound}");
    assert_eq!(parts[0], "00", "version should be 00");
    assert_eq!(parts[1].len(), 32, "trace-id should be 32 hex chars: {outbound}");
    assert_eq!(parts[2].len(), 16, "span-id should be 16 hex chars: {outbound}");
    assert_eq!(parts[3], "01", "new traces should be marked sampled");
    assert!(!is_all_zero(parts[1]), "trace-id must not be all zeros");
    assert!(!is_all_zero(parts[2]), "span-id must not be all zeros");
}

#[test]
fn discards_malformed_inbound_traceparent() {
    // Client-controlled input must not reach the telemetry backend
    // unchecked.
    let malformed = [
        "not-a-traceparent",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
        "00-zzf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    ];

    for value in malformed {
        let req = request_with(&[("traceparent", value)]);
        let ctx = make_filter_context(&req);

        let map = applied(&Correlation::from_filter_context(&ctx));
        let outbound = header(&map, "traceparent");

        assert_ne!(
            outbound, value,
            "malformed traceparent should not be forwarded: {value}"
        );
        let parts: Vec<&str> = outbound.split('-').collect();
        assert_eq!(parts.len(), 4, "replacement should be well-formed: {outbound}");
        assert_eq!(parts[1].len(), 32, "replacement trace-id should be 32 hex: {outbound}");
    }
}

#[test]
fn parse_traceparent_accepts_unsampled_flags() {
    let parsed = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
        .expect("unsampled but well-formed traceparent should parse");

    assert_eq!(parsed.flags, "00", "unsampled flags should be preserved");
}

#[test]
fn parse_traceparent_accepts_future_versions() {
    let parsed = parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        .expect("future version should still yield a usable trace-id");

    assert_eq!(
        parsed.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736",
        "trace-id should be extracted from a future version"
    );
}

#[test]
fn parse_traceparent_ignores_future_version_extension_fields() {
    // A higher version may append fields after the flags. Dropping the
    // trace because of them would restart traces during an upstream
    // version rollout.
    let parsed = parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extension")
        .expect("extension fields of a future version should be ignored");

    assert_eq!(
        parsed.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736",
        "base fields of a future version should still be continued"
    );
    assert_eq!(parsed.flags, "01", "base flags should survive the extension field");
}

#[test]
fn masks_trace_flags_this_version_does_not_define() {
    // Only the sampled bit is specified; the rest are reserved and
    // must not be re-emitted with an invented meaning.
    for (inbound, expected) in [("ff", "01"), ("fe", "00"), ("03", "01"), ("02", "00")] {
        let value = format!("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-{inbound}");
        let parsed = parse_traceparent(&value).expect("well-formed flags should parse");

        assert_eq!(
            parsed.flags, expected,
            "flags {inbound} should be masked to the sampled bit"
        );
    }
}

#[test]
fn masked_flags_reach_the_outbound_header() {
    let req = request_with(&[("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-ff")]);
    let ctx = make_filter_context(&req);

    let map = applied(&Correlation::from_filter_context(&ctx));
    let outbound = header(&map, "traceparent");

    assert!(
        outbound.ends_with("-01"),
        "reserved flag bits must not be forwarded: {outbound}"
    );
}

// -----------------------------------------------------------------------------
// Defensive ID sanitization
// -----------------------------------------------------------------------------

#[test]
fn sanitized_ids_are_never_all_zero() {
    // Inputs a future core generator could plausibly produce, none of
    // which may yield the all-zero ID W3C section 2.2.2 forbids.
    let degenerate = ["", "0", "00000000000000000000000000000000", "----", "zzzz"];

    for id in degenerate {
        let trace_id = sanitize_trace_id(id);
        assert_eq!(trace_id.len(), TRACE_ID_LEN, "trace-id should be 32 chars: {id:?}");
        assert!(is_lower_hex(&trace_id), "trace-id should be lowercase hex: {trace_id}");
        assert!(!is_all_zero(&trace_id), "trace-id must not be all zeros: {id:?}");

        let span_id = span_id_from(&trace_id);
        assert_eq!(span_id.len(), SPAN_ID_LEN, "span-id should be 16 chars: {id:?}");
        assert!(!is_all_zero(&span_id), "span-id must not be all zeros: {id:?}");
    }
}

#[test]
fn sanitize_preserves_a_well_formed_generated_id() {
    let id = "4bf92f3577b34da6a3ce929d0e0e4736";

    assert_eq!(sanitize_trace_id(id), id, "a valid generated ID should pass through");
}

#[test]
fn span_id_falls_back_when_the_trace_id_tail_is_zero() {
    // A trace-id that is itself valid can still end in 16 zeros.
    let trace_id = "4bf92f3577b34da60000000000000000";

    let span_id = span_id_from(trace_id);

    assert!(!is_all_zero(&span_id), "span-id must not be all zeros: {span_id}");
    assert_eq!(span_id.len(), SPAN_ID_LEN, "span-id should be 16 chars: {span_id}");
}

// -----------------------------------------------------------------------------
// Forwarded and delegated legs share one trace
// -----------------------------------------------------------------------------

#[test]
fn delegated_call_joins_trace_injected_for_the_forwarded_request() {
    // What the trace_context filter injects for the upstream hop.
    let req = request_with(&[]);
    let mut ctx = make_filter_context(&req);
    let forwarded = TraceContext::from_filter_context(&ctx);
    let forwarded_traceparent = forwarded.traceparent_for_hop(&ctx);
    ctx.extra_request_headers
        .push((Cow::Borrowed("traceparent"), forwarded_traceparent.clone()));
    ctx.extra_request_headers
        .push((Cow::Borrowed("X-Request-ID"), forwarded.request_id().to_owned()));

    // What a delegated call resolves later in the same request.
    let map = applied(&Correlation::from_filter_context(&ctx));
    let delegated_traceparent = header(&map, "traceparent");

    let forwarded_parts: Vec<&str> = forwarded_traceparent.split('-').collect();
    let delegated_parts: Vec<&str> = delegated_traceparent.split('-').collect();

    assert_eq!(
        forwarded_parts[1], delegated_parts[1],
        "forwarded and delegated legs must share a trace-id"
    );
    assert_ne!(
        forwarded_parts[2], delegated_parts[2],
        "each leg must be its own span so their latencies stay separable"
    );
    assert_eq!(
        header(&map, "x-request-id"),
        forwarded.request_id(),
        "both legs must carry the same request ID"
    );
}

#[test]
fn init_makes_repeated_resolutions_share_one_trace() {
    let req = request_with(&[]);
    let mut ctx = make_filter_context(&req);
    drop(TraceContext::get_or_init(&mut ctx));

    // Two resolution phases of one filter, e.g. current input and
    // rehydrated history.
    let first = applied(&Correlation::from_filter_context(&ctx));
    let second = applied(&Correlation::from_filter_context(&ctx));

    let first_parts: Vec<String> = header(&first, "traceparent").split('-').map(str::to_owned).collect();
    let second_parts: Vec<String> = header(&second, "traceparent").split('-').map(str::to_owned).collect();

    assert_eq!(
        first_parts[1], second_parts[1],
        "callouts of one request must share a trace-id once initialized"
    );
    assert_eq!(
        header(&first, "x-request-id"),
        header(&second, "x-request-id"),
        "callouts of one request must share a request ID"
    );
}

#[test]
fn client_trace_reaches_both_legs_unchanged() {
    let inbound = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let req = request_with(&[("traceparent", inbound)]);
    let ctx = make_filter_context(&req);

    let forwarded = TraceContext::from_filter_context(&ctx);
    let forwarded_traceparent = forwarded.traceparent_for_hop(&ctx);
    let map = applied(&Correlation::from_filter_context(&ctx));

    for value in [&forwarded_traceparent, &header(&map, "traceparent")] {
        let parts: Vec<&str> = value.split('-').collect();
        assert_eq!(
            parts[1], "4bf92f3577b34da6a3ce929d0e0e4736",
            "client trace-id should reach both legs: {value}"
        );
    }
}

// -----------------------------------------------------------------------------
// Application onto the callout map
// -----------------------------------------------------------------------------

#[test]
fn correlation_overwrites_forwarded_value_of_same_name() {
    let req = request_with(&[("x-request-id", "client-abc")]);
    let ctx = make_filter_context(&req);
    let correlation = Correlation::from_filter_context(&ctx);

    // Simulate an operator listing x-request-id in forward_headers
    // and a stale value already present in the callout map.
    let mut map = HeaderMap::new();
    map.insert(
        HeaderName::from_static("x-request-id"),
        HeaderValue::from_static("stale"),
    );
    correlation.apply(&mut map);

    assert_eq!(
        header(&map, "x-request-id"),
        "client-abc",
        "correlation should overwrite, not append"
    );
    assert_eq!(
        map.get_all("x-request-id").iter().count(),
        1,
        "correlation should not leave duplicate header values"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Apply correlation to an empty map and return it.
fn applied(correlation: &Correlation) -> HeaderMap {
    let mut map = HeaderMap::new();
    correlation.apply(&mut map);
    map
}

/// Read a header as a string.
fn header(map: &HeaderMap, name: &str) -> String {
    map.get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

/// Build a request carrying the given headers.
fn request_with(headers: &[(&'static str, &str)]) -> praxis_filter::Request {
    let mut req = make_request(Method::POST, "/v1/responses");
    for (name, value) in headers {
        req.headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_str(value).expect("valid test header value"),
        );
    }
    req
}
