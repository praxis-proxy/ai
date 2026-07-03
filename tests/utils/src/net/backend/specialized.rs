// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Shared TCP server utilities for test backends.

use std::{
    io::{Read as _, Write as _},
    net::TcpStream,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// Spawn a raw TCP server that calls `handler` in a new
/// thread for each accepted connection. Returns the port.
pub(crate) fn spawn_tcp_server(handler: impl Fn(TcpStream) + Send + Clone + 'static) -> u16 {
    let (listener, port) = crate::net::port::bind_unique_port();

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let handler = handler.clone();
            std::thread::spawn(move || handler(stream));
        }
    });

    port
}

/// RAII guard that shuts down a backend spawned by
/// `spawn_tcp_server_with_shutdown` when dropped.
pub struct BackendGuard {
    /// The port the backend is listening on.
    port: u16,

    /// Shared flag signalling the listener loop to exit.
    shutdown: Arc<AtomicBool>,
}

impl BackendGuard {
    /// The allocated port number.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        _ = TcpStream::connect(format!("127.0.0.1:{}", self.port));
    }
}

/// Spawn a raw TCP server with a shutdown guard. The
/// listener loop exits when the returned [`BackendGuard`]
/// is dropped.
pub(crate) fn spawn_tcp_server_with_shutdown(handler: impl Fn(TcpStream) + Send + Clone + 'static) -> BackendGuard {
    let (listener, port) = crate::net::port::bind_unique_port();
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&shutdown);

    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            if flag.load(Ordering::Acquire) {
                break;
            }
            let handler = handler.clone();
            std::thread::spawn(move || handler(stream));
        }
    });

    BackendGuard { port, shutdown }
}

/// Read from a TCP stream until the HTTP header terminator
/// (`\r\n\r\n`) is received. Returns the raw request as a
/// string. Prevents partial-read flakiness under load.
pub(crate) fn read_until_headers_complete(stream: &mut TcpStream) -> String {
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

    String::from_utf8_lossy(&data).into_owned()
}

/// Extract Content-Length from raw HTTP headers.
pub(crate) fn parse_content_length(headers: &str) -> usize {
    headers
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':').map(|(_, v)| v))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// Write a minimal HTTP 200 response with the given body.
pub(crate) fn write_http_response(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes())
}
