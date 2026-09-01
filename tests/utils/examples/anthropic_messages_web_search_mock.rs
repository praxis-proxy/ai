// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Deterministic native Anthropic Messages backend for the web-search example.

use std::{
    io::{self, Read as _, Write as _},
    net::{TcpListener, TcpStream},
};

use serde_json::{Value, json};

/// Identifier returned by the deterministic tool-use response.
const TOOL_USE_ID: &str = "toolu_web_search_01";

#[expect(
    clippy::print_stderr,
    reason = "standalone mock reports its listener and connection failures"
)]
fn main() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8000")?;
    eprintln!("Anthropic Messages web-search mock listening on 127.0.0.1:8000");
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = handle_connection(&mut stream) {
                    eprintln!("mock request failed: {error}");
                }
            },
            Err(error) => eprintln!("mock accept failed: {error}"),
        }
    }
    Ok(())
}

/// Handle one HTTP connection to the deterministic Messages backend.
fn handle_connection(stream: &mut TcpStream) -> io::Result<()> {
    let (method, path, body) = read_request(stream)?;
    if method != "POST" || path != "/v1/messages" {
        return write_json_response(
            stream,
            404,
            "Not Found",
            &json!({"error": {"type": "not_found_error", "message": "not found"}}),
        );
    }

    let Ok(request) = serde_json::from_slice(&body) else {
        return write_json_response(
            stream,
            400,
            "Bad Request",
            &json!({"error": {"type": "invalid_request_error", "message": "invalid JSON"}}),
        );
    };
    write_json_response(stream, 200, "OK", &response_for_request(&request))
}

/// Read one HTTP request, using its `Content-Length` to delimit the JSON body.
#[expect(
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "bounded reads make each validated request boundary explicit"
)]
fn read_request(stream: &mut TcpStream) -> io::Result<(String, String, Vec<u8>)> {
    let mut request = Vec::new();
    let header_end = loop {
        if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break end;
        }
        let mut buffer = [0_u8; 4096];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before headers",
            ));
        }
        request.extend_from_slice(&buffer[..count]);
    };

    let headers = std::str::from_utf8(&request[..header_end])
        .map_err(|_error| io::Error::new(io::ErrorKind::InvalidData, "request headers are not UTF-8"))?;
    let (method, path, content_length) = parse_request_headers(headers)?;

    let body_start = header_end + 4;
    while request.len() < body_start + content_length {
        let mut buffer = [0_u8; 4096];
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before body",
            ));
        }
        request.extend_from_slice(&buffer[..count]);
    }
    Ok((method, path, request[body_start..body_start + content_length].to_vec()))
}

/// Parse the request line and `Content-Length` from a complete HTTP header block.
fn parse_request_headers(headers: &str) -> io::Result<(String, String, usize)> {
    let mut request_line = headers.lines().next().unwrap_or_default().split_whitespace();
    let method = request_line
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "request has no method"))?
        .to_owned();
    let path = request_line
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "request has no path"))?
        .to_owned();
    let content_length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>())
        .transpose()
        .map_err(|_error| io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Length"))?
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length"))?;
    Ok((method, path, content_length))
}

/// Serialize and send one JSON response with exact HTTP framing.
fn write_json_response(stream: &mut TcpStream, status: u16, reason: &str, response: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(response)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, format!("serialize response: {error}")))?;
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)
}

/// Select the model response from the presence of the matching tool result.
#[expect(
    clippy::too_many_lines,
    reason = "the fixed deterministic response shapes are easiest to audit inline"
)]
fn response_for_request(request: &Value) -> Value {
    if has_matching_tool_result(request) {
        json!({
            "id": "msg_web_search_02",
            "type": "message",
            "role": "assistant",
            "model": "openai/gpt-oss-20b",
            "content": [{
                "type": "text",
                "text": "Potato is a starchy underground tuber native to the Americas and now eaten worldwide."
            }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 74, "output_tokens": 18}
        })
    } else {
        json!({
            "id": "msg_web_search_01",
            "type": "message",
            "role": "assistant",
            "model": "openai/gpt-oss-20b",
            "content": [{
                "type": "tool_use",
                "id": TOOL_USE_ID,
                "name": "WebSearch",
                "input": {"query": "potato"}
            }],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {"input_tokens": 20, "output_tokens": 8}
        })
    }
}

/// Return whether a Messages turn contains the mock's tool-result identifier.
fn has_matching_tool_result(request: &Value) -> bool {
    request
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .any(|block| {
            block.get("type").and_then(Value::as_str) == Some("tool_result")
                && block.get("tool_use_id").and_then(Value::as_str) == Some(TOOL_USE_ID)
        })
}

#[test]
fn request_without_tool_result_gets_tool_use() {
    let response = response_for_request(&json!({
        "messages": [{"role": "user", "content": "search potato"}]
    }));
    assert_eq!(response["stop_reason"], "tool_use");
    assert_eq!(response["content"][0]["name"], "WebSearch");
}

#[test]
fn matching_tool_result_gets_final_text() {
    let response = response_for_request(&json!({"messages": [{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_web_search_01",
            "content": "[1] Potato - Wikipedia"
        }]
    }]}));
    assert_eq!(response["stop_reason"], "end_turn");
    assert_eq!(response["content"][0]["type"], "text");
}

#[test]
fn content_length_is_read_after_other_headers() {
    let (_, _, content_length) =
        parse_request_headers("POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Length: 17")
            .expect("parse request headers");
    assert_eq!(content_length, 17);
}
