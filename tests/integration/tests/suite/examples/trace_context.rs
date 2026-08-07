// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `trace_context` example config.
//!
//! Verifies the acceptance criterion of the correlation work: one
//! client request is traceable across both legs it produces — the
//! delegated Files API callout the proxy originates itself, and the
//! forwarded inference request — under a single trace-id with
//! distinct span-ids per leg.

use std::{
    collections::{HashMap, HashSet},
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    time::Duration,
};

use praxis_test_utils::{free_port, http_send, json_post, load_example_config, parse_status, start_proxy};

/// Client trace the proxy is expected to continue.
const CLIENT_TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

/// Trace-id embedded in [`CLIENT_TRACEPARENT`].
const CLIENT_TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

/// Span-id embedded in [`CLIENT_TRACEPARENT`].
const CLIENT_SPAN_ID: &str = "00f067aa0ba902b7";

/// Responses request referencing a file, which triggers a delegated
/// Files API callout before the request is forwarded upstream.
const REQUEST_BODY: &str = r#"{
    "model": "gpt-4.1",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file-abc"}]
        }
    ]
}"#;

/// File metadata returned by the stub.
const FILE_METADATA: &str = r#"{"id":"file-abc","object":"file","bytes":13,"created_at":1750000000,"filename":"test.txt","purpose":"user_data"}"#;

/// File content returned by the stub.
const FILE_CONTENT: &str = "Hello, world!";

/// One request seen by the stub backend.
#[derive(Clone)]
struct SeenRequest {
    /// Request line target, e.g. `/v1/files/file-abc`.
    path: String,
    /// Lowercased header names mapped to their values.
    headers: HashMap<String, String>,
}

impl SeenRequest {
    /// Read a header value by case-insensitive name.
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
    }

    /// The `traceparent` this leg carried, or a panic naming the leg.
    fn traceparent(&self) -> &str {
        self.header("traceparent")
            .unwrap_or_else(|| panic!("leg {} must carry traceparent", self.path))
    }
}

/// Extract field `index` of a `traceparent` value.
fn field(traceparent: &str, index: usize) -> &str {
    traceparent.split('-').nth(index).unwrap_or_default()
}

/// A backend that answers Files API and inference paths while
/// recording every request it sees.
///
/// Path-aware rather than sequential, because the proxy readiness
/// probe issues its own request before the traced one.
fn start_recording_backend() -> (u16, Arc<Mutex<Vec<SeenRequest>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("stub should bind");
    let port = listener.local_addr().expect("stub should have an address").port();
    let seen: Arc<Mutex<Vec<SeenRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let recorder = Arc::clone(&recorder);
            std::thread::spawn(move || handle_request(stream, &recorder));
        }
    });

    (port, seen)
}

/// Serve one request and record it.
fn handle_request(mut stream: TcpStream, seen: &Arc<Mutex<Vec<SeenRequest>>>) {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("read timeout should apply");

    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        if data.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    let raw = String::from_utf8_lossy(&data);
    let path = raw
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_owned();

    let headers = raw
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();

    // The readiness probe is not part of the traced request.
    if path != "/" {
        seen.lock()
            .expect("recorder mutex should not be poisoned")
            .push(SeenRequest {
                path: path.clone(),
                headers,
            });
    }

    let (content_type, body) = if path.ends_with("/content") {
        ("text/plain", FILE_CONTENT)
    } else if path.starts_with("/v1/files") {
        ("application/json", FILE_METADATA)
    } else {
        ("application/json", r#"{"id":"resp_1","object":"response","output":[]}"#)
    };

    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _sent = stream.write_all(header.as_bytes());
    let _sent = stream.write_all(body.as_bytes());
}

/// Send one file-referencing request through the example config and
/// return the legs the backend saw.
fn capture_legs(client_traceparent: Option<&str>) -> Vec<SeenRequest> {
    let (backend_port, seen) = start_recording_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "trace-context.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8321", backend_port)]),
    );
    let proxy = start_proxy(&config);

    let mut request = json_post("/v1/responses", REQUEST_BODY);
    if let Some(traceparent) = client_traceparent {
        request = request.replace("\r\n\r\n", &format!("\r\ntraceparent: {traceparent}\r\n\r\n"));
    }

    let raw = http_send(proxy.addr(), &request);
    assert_eq!(parse_status(&raw), 200, "traced request should succeed");

    let legs = seen.lock().expect("recorder mutex should not be poisoned").clone();
    assert!(
        legs.iter().any(|leg| leg.path.starts_with("/v1/files")),
        "expected a delegated Files API callout, saw {:?}",
        legs.iter().map(|leg| &leg.path).collect::<Vec<_>>()
    );
    assert!(
        legs.iter().any(|leg| leg.path == "/v1/responses"),
        "expected the forwarded inference request, saw {:?}",
        legs.iter().map(|leg| &leg.path).collect::<Vec<_>>()
    );
    legs
}

#[test]
fn example_config_correlates_delegated_and_forwarded_legs() {
    let legs = capture_legs(None);

    let paths: Vec<&str> = legs.iter().map(|leg| leg.path.as_str()).collect();

    let trace_ids: HashSet<&str> = legs.iter().map(|leg| field(leg.traceparent(), 1)).collect();
    assert_eq!(
        trace_ids.len(),
        1,
        "all legs must share one trace-id, saw {trace_ids:?} across {paths:?}"
    );
    assert!(
        !trace_ids.iter().next().is_some_and(|id| id.is_empty()),
        "trace-id should be populated"
    );

    // The delegated callouts of one request share a span: correlation
    // is resolved once at the filter boundary, so file resolution is
    // one delegation hop regardless of how many files it fetches. The
    // forwarded request must be a distinct span, which is what keeps
    // delegation latency separable from inference latency.
    let delegated_spans: HashSet<&str> = legs
        .iter()
        .filter(|leg| leg.path.starts_with("/v1/files"))
        .map(|leg| field(leg.traceparent(), 2))
        .collect();
    let forwarded_spans: HashSet<&str> = legs
        .iter()
        .filter(|leg| leg.path == "/v1/responses")
        .map(|leg| field(leg.traceparent(), 2))
        .collect();

    assert_eq!(
        delegated_spans.len(),
        1,
        "delegated callouts of one request should share a span, saw {delegated_spans:?} across {paths:?}"
    );
    assert!(
        delegated_spans.is_disjoint(&forwarded_spans),
        "forwarded request must be its own span so delegation latency stays separable: \
         delegated={delegated_spans:?} forwarded={forwarded_spans:?}"
    );

    let request_ids: HashSet<&str> = legs
        .iter()
        .map(|leg| {
            leg.header("x-request-id")
                .unwrap_or_else(|| panic!("leg {} must carry x-request-id", leg.path))
        })
        .collect();
    assert_eq!(
        request_ids.len(),
        1,
        "all legs must share one request ID, saw {request_ids:?} across {paths:?}"
    );
}

#[test]
fn example_config_continues_client_supplied_trace() {
    let legs = capture_legs(Some(CLIENT_TRACEPARENT));

    for leg in &legs {
        let traceparent = leg.traceparent();
        assert_eq!(
            field(traceparent, 1),
            CLIENT_TRACE_ID,
            "client trace-id should be continued on leg {}",
            leg.path
        );
        assert_ne!(
            field(traceparent, 2),
            CLIENT_SPAN_ID,
            "leg {} must mint its own span-id, not reuse the client's",
            leg.path
        );
    }
}
