// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Echo backends that reflect request data back in
//! the response.

use std::{
    net::TcpStream,
    sync::{Arc, Mutex},
    time::Duration,
};

use super::specialized::{
    BackendGuard, parse_content_length, read_until_headers_complete, spawn_tcp_server_with_shutdown,
    write_http_response,
};

// -----------------------------------------------------------------------------
// Echo Backends
// -----------------------------------------------------------------------------

/// Start a mock backend that echoes the request body back
/// as the response body.
///
/// Returns a [`BackendGuard`] that shuts down the listener
/// thread when dropped.
///
/// # Panics
///
/// Panics if the server fails to bind or accept connections.
pub fn start_echo_backend() -> BackendGuard {
    spawn_tcp_server_with_shutdown(|mut stream| {
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let body = read_request_body(&mut stream);
        let _sent = write_http_response(&mut stream, &body);
    })
}

/// Backend guard that captures the last non-empty request body.
pub struct CapturingBackendGuard {
    /// Inner backend guard that shuts down the listener on drop.
    guard: BackendGuard,
    /// Captured request body.
    body: Arc<Mutex<Option<String>>>,
}

impl CapturingBackendGuard {
    /// The allocated port number.
    pub fn port(&self) -> u16 {
        self.guard.port()
    }

    /// Return the captured request body.
    ///
    /// # Panics
    ///
    /// Panics when no request body has been captured.
    pub fn body(&self) -> String {
        self.body
            .lock()
            .expect("captured body mutex should not be poisoned")
            .clone()
            .expect("backend should capture one request body")
    }
}

/// Start a backend that captures request bodies and returns a fixed
/// JSON response body.
///
/// Returns a guard that shuts down the listener thread when dropped.
///
/// # Panics
///
/// Panics if the server fails to bind or accept connections.
pub fn start_capturing_backend(response_body: &str) -> CapturingBackendGuard {
    let body = Arc::new(Mutex::new(None));
    let captured = Arc::clone(&body);
    let response_body = response_body.to_owned();
    let guard = spawn_tcp_server_with_shutdown(move |mut stream| {
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let request_body = read_request_body(&mut stream);
        if !request_body.is_empty() {
            *captured.lock().expect("captured body mutex should not be poisoned") = Some(request_body);
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        let _sent = std::io::Write::write_all(&mut stream, response.as_bytes());
    });

    CapturingBackendGuard { guard, body }
}

/// Start a backend that echoes the request URI (path and query)
/// as the response body.
///
/// Returns a [`BackendGuard`] that shuts down the listener
/// thread when dropped.
///
/// # Panics
///
/// Panics if the server fails to bind or accept connections.
pub fn start_uri_echo_backend() -> BackendGuard {
    spawn_tcp_server_with_shutdown(|mut stream| {
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let raw = read_until_headers_complete(&mut stream);
        let uri = raw
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/")
            .to_owned();
        let _sent = write_http_response(&mut stream, &uri);
    })
}

/// Start a backend that echoes request headers as the
/// response body (one per line).
///
/// Returns a [`BackendGuard`] that shuts down the listener
/// thread when dropped.
///
/// # Panics
///
/// Panics if the server fails to bind or accept connections.
pub fn start_header_echo_backend() -> BackendGuard {
    spawn_tcp_server_with_shutdown(|mut stream| {
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let raw = read_until_headers_complete(&mut stream);

        let headers: String = raw
            .lines()
            .skip(1)
            .take_while(|l| !l.is_empty())
            .fold(String::new(), |mut acc, line| {
                if !acc.is_empty() {
                    acc.push('\n');
                }
                acc.push_str(line);
                acc
            });

        let _sent = write_http_response(&mut stream, &headers);
    })
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Read a complete HTTP request body from a raw TCP stream,
/// using Content-Length to determine when all bytes have arrived.
fn read_request_body(stream: &mut TcpStream) -> String {
    use std::io::Read as _;

    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];

    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }

        let raw = String::from_utf8_lossy(&data);
        if let Some(header_section) = raw.split("\r\n\r\n").next() {
            let content_length = parse_content_length(header_section);
            let header_len = header_section.len() + 4;
            if data.len() >= header_len + content_length {
                break;
            }
        }
    }

    let raw = String::from_utf8_lossy(&data);
    raw.split("\r\n\r\n").nth(1).unwrap_or("").to_owned()
}
