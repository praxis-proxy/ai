// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Loopback capture-relay for the Tavily web-search provider.
//!
//! This is **not a mock**. The relay binds a loopback HTTP listener, forwards
//! every POST verbatim to the real Tavily upstream (default
//! [`TAVILY_UPSTREAM`]) over a public-root HTTPS client, and streams the real
//! upstream status and body back to the caller. On the way through it records
//! `{request, status, response}` for each exchange so a test can prove a live
//! search actually happened — the managed `anthropic_web_search` loop suppresses
//! the `WebSearch` `tool_use` and falls back to an `is_error` tool result on
//! provider failure, so a client-only "got an answer" assertion would pass even
//! against a broken integration. Observing the real call closes that gap.
//!
//! Point a filter's `base_url` at [`TavilyRelayGuard::base_url`] and give it the
//! real Tavily API key; the relay carries the request (key and all) to Tavily
//! and hands back exactly what Tavily returned.

use std::{
    convert::Infallible,
    net::TcpListener,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use http::{Request, Response, StatusCode, header};
use http_body_util::{BodyExt as _, Full};
use hyper::{body::Incoming, server::conn::http1, service::service_fn};
use hyper_util::rt::TokioIo;
use serde_json::Value;

/// The production Tavily search endpoint the relay forwards to by default.
pub const TAVILY_UPSTREAM: &str = "https://api.tavily.com/search";

/// Maximum time the relay waits for the real Tavily upstream to respond.
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(30);

/// One observed search exchange: the forwarded request, the upstream status,
/// and the upstream response, each parsed best-effort as JSON (`Value::Null`
/// when a body is absent or not JSON).
#[derive(Clone, Debug)]
pub struct TavilySearchCapture {
    /// The request body the proxy sent to the relay (forwarded to Tavily).
    pub request: Value,
    /// The HTTP status the real Tavily upstream returned (`0` on a transport
    /// failure reaching Tavily).
    pub status: u16,
    /// The response body the real Tavily upstream returned.
    pub response: Value,
}

/// Handle to a running Tavily capture-relay.
///
/// The relay runs on a detached runtime thread that lives until the process
/// exits (mirroring the loopback backends in [`crate::net`]); dropping the
/// guard only drops the shared capture log, not the listener.
pub struct TavilyRelayGuard {
    /// The loopback port the relay listens on.
    port: u16,
    /// Shared log of every observed exchange, oldest first.
    captures: Arc<Mutex<Vec<TavilySearchCapture>>>,
}

impl TavilyRelayGuard {
    /// The loopback port the relay listens on.
    #[must_use]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The `http://127.0.0.1:<port>` base URL to point a filter `base_url` at.
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// A snapshot of every exchange observed so far, oldest first.
    ///
    /// # Panics
    ///
    /// Panics if the capture lock is poisoned by a panicking relay task.
    #[must_use]
    pub fn captures(&self) -> Vec<TavilySearchCapture> {
        self.captures.lock().expect("tavily relay capture lock").clone()
    }

    /// Drops every recorded exchange (useful between phases of one test).
    ///
    /// # Panics
    ///
    /// Panics if the capture lock is poisoned by a panicking relay task.
    pub fn clear(&self) {
        self.captures.lock().expect("tavily relay capture lock").clear();
    }
}

/// Starts a capture-relay that forwards to the real Tavily endpoint.
///
/// See [`start_tavily_relay_to`] to target a different upstream (its own tests
/// point it at a loopback stub).
///
/// # Panics
///
/// Panics if the loopback listener cannot be bound or the egress client cannot
/// be built.
#[must_use]
pub fn start_tavily_relay() -> TavilyRelayGuard {
    start_tavily_relay_to(TAVILY_UPSTREAM)
}

/// Starts a capture-relay that forwards to an explicit upstream URL.
///
/// # Panics
///
/// Panics if the loopback listener cannot be bound or the egress client cannot
/// be built.
#[must_use]
pub fn start_tavily_relay_to(upstream_url: &str) -> TavilyRelayGuard {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind tavily relay listener");
    let port = listener.local_addr().expect("tavily relay port").port();

    // Public-root HTTPS client on the installed crypto provider, so the real
    // Tavily certificate chain validates exactly as the production binary's
    // would. `no_proxy` keeps CI HTTP(S)_PROXY settings from hijacking egress.
    let client = crate::inference_fixture::http_client_builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(UPSTREAM_TIMEOUT)
        .build()
        .expect("build tavily relay egress client");

    let captures: Arc<Mutex<Vec<TavilySearchCapture>>> = Arc::new(Mutex::new(Vec::new()));
    let thread_captures = Arc::clone(&captures);
    let upstream: Arc<str> = Arc::from(upstream_url);

    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime for tavily relay");
        runtime.block_on(accept_loop(listener, Arc::new(client), upstream, thread_captures));
    });

    TavilyRelayGuard { port, captures }
}

/// Accepts loopback connections and serves each on the current-thread runtime.
#[expect(clippy::infinite_loop, reason = "server accept loop runs until task cancellation")]
async fn accept_loop(
    listener: TcpListener,
    client: Arc<reqwest::Client>,
    upstream: Arc<str>,
    captures: Arc<Mutex<Vec<TavilySearchCapture>>>,
) {
    listener.set_nonblocking(true).expect("tavily relay nonblocking");
    let listener = tokio::net::TcpListener::from_std(listener).expect("tavily relay tokio listener");
    loop {
        let Ok((stream, _peer)) = listener.accept().await else {
            continue;
        };
        let client = Arc::clone(&client);
        let upstream = Arc::clone(&upstream);
        let captures = Arc::clone(&captures);
        tokio::spawn(async move {
            let service = service_fn(move |request: Request<Incoming>| {
                let client = Arc::clone(&client);
                let upstream = Arc::clone(&upstream);
                let captures = Arc::clone(&captures);
                async move { forward(request, client, upstream, captures).await }
            });
            // One request per connection keeps the forward-and-capture path
            // simple; the proxy opens a fresh connection per search anyway.
            let mut builder = http1::Builder::new();
            builder.keep_alive(false);
            let _ = builder.serve_connection(TokioIo::new(stream), service).await;
        });
    }
}

/// Forwards one request to the real upstream and records the exchange.
async fn forward(
    request: Request<Incoming>,
    client: Arc<reqwest::Client>,
    upstream: Arc<str>,
    captures: Arc<Mutex<Vec<TavilySearchCapture>>>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let request_bytes = match request.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => Bytes::new(),
    };
    let request_json = serde_json::from_slice::<Value>(&request_bytes).unwrap_or(Value::Null);

    // Carry the original body straight to Tavily; no clone of the payload.
    let forwarded = client
        .post(&*upstream)
        .header(header::CONTENT_TYPE, "application/json")
        .body(request_bytes)
        .send()
        .await;

    Ok(record_and_respond(&captures, request_json, forwarded).await)
}

/// Records one exchange against the capture log and builds the client response.
#[expect(
    clippy::too_many_lines,
    reason = "two symmetric success/error arms each record then build a response"
)]
async fn record_and_respond(
    captures: &Mutex<Vec<TavilySearchCapture>>,
    request_json: Value,
    forwarded: Result<reqwest::Response, reqwest::Error>,
) -> Response<Full<Bytes>> {
    match forwarded {
        Ok(response) => {
            let status = response.status();
            let response_bytes = response.bytes().await.unwrap_or_default();
            let response_json = serde_json::from_slice::<Value>(&response_bytes).unwrap_or(Value::Null);
            captures
                .lock()
                .expect("tavily relay capture lock")
                .push(TavilySearchCapture {
                    request: request_json,
                    status: status.as_u16(),
                    response: response_json,
                });
            Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Full::new(response_bytes))
                .expect("build tavily relay response")
        },
        Err(error) => {
            let message = format!("tavily relay upstream error: {error}");
            captures
                .lock()
                .expect("tavily relay capture lock")
                .push(TavilySearchCapture {
                    request: request_json,
                    status: 0,
                    response: Value::String(message.clone()),
                });
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Full::new(Bytes::from(message)))
                .expect("build tavily relay error response")
        },
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};

    use super::*;

    /// Minimal raw-TCP HTTP/1.1 stub that answers every request with a fixed
    /// JSON body. Stands in for Tavily so the relay's forward-and-capture path
    /// is exercised offline.
    fn spawn_json_stub(body: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let port = listener.local_addr().expect("stub port").port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.expect("accept stub connection");
                let mut scratch = [0_u8; 4096];
                let _ = stream.read(&mut scratch);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    #[test]
    fn relay_forwards_and_captures_the_exchange() {
        let stub_port = spawn_json_stub(r#"{"results":[{"url":"https://example.com/a"}]}"#);
        let relay = start_tavily_relay_to(&format!("http://127.0.0.1:{stub_port}/search"));

        let client = crate::inference_fixture::http_client();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("client runtime");
        let response = runtime.block_on(async {
            client
                .post(relay.base_url())
                .header(header::CONTENT_TYPE, "application/json")
                .body(r#"{"api_key":"tvly-test","query":"praxis proxy"}"#)
                .send()
                .await
                .expect("relay request")
        });

        assert_eq!(response.status(), StatusCode::OK);
        let body = runtime.block_on(async { response.text().await.expect("relay body") });
        assert!(body.contains("example.com"), "relay returned upstream body: {body}");

        let captures = relay.captures();
        assert_eq!(captures.len(), 1, "exactly one exchange recorded");
        let exchange = &captures[0];
        assert_eq!(exchange.status, 200);
        assert_eq!(exchange.request["api_key"], "tvly-test");
        assert_eq!(exchange.request["query"], "praxis proxy");
        assert_eq!(exchange.response["results"][0]["url"], "https://example.com/a");

        relay.clear();
        assert!(relay.captures().is_empty(), "clear() drops recorded exchanges");
    }
}
