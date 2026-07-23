// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Responses API full-flow example config.

use std::{collections::HashMap, time::Duration};

use futures::{SinkExt as _, StreamExt as _};
use praxis_test_utils::{
    Backend, CapturedWsMessage, TempSqlite, WsBackendEvent, WsServerAction, example_config_path, free_port, http_send,
    json_post, load_example_config, parse_body, parse_status, patch_yaml, start_backend_with_shutdown,
    start_echo_backend, start_proxy, start_scripted_websocket_backend,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{
        Error, Message,
        client::IntoClientRequest as _,
        handshake::client::Response,
        protocol::{CloseFrame, frame::coding::CloseCode},
    },
};

use super::openai_file_resolve::start_files_api_stub;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Backend response for the first turn — stored by response_store.
const FIRST_RESPONSE_JSON: &str = r#"{"id":"resp_first","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"Hello","output":[{"type":"message","content":[{"type":"output_text","text":"Hi there"}]}]}"#;

/// Maximum time allowed for a test client to complete a WebSocket handshake.
const WEBSOCKET_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_resolves_rehydrated_files_before_proxy() {
    let files_api_port = start_files_api_stub();
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();
    let db = TempSqlite::new("full_flow_file_resolve");

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/full-flow.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([
            ("127.0.0.1:9999", files_api_port),
            ("127.0.0.1:3001", backend_guard.port()),
        ]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let create_raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/conversations",
            r#"{
                "metadata": {},
                "items": [{
                    "id": "item_file",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "test-file-123"}]
                }]
            }"#,
        ),
    );
    assert_eq!(parse_status(&create_raw), 200, "conversation creation should succeed");
    let created: serde_json::Value =
        serde_json::from_str(&parse_body(&create_raw)).expect("conversation response should be valid JSON");
    let conversation_id = created["id"]
        .as_str()
        .expect("conversation response should contain an id");

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Summarize the file",
        "conversation": conversation_id,
    });
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            &serde_json::to_string(&request).expect("request should serialize"),
        ),
    );
    assert_eq!(parse_status(&raw), 200, "response request should reach the backend");

    let echoed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("echoed backend request should be valid JSON");
    assert!(
        echoed.get("conversation").is_none(),
        "conversation should be stripped after rehydration"
    );

    let input = echoed["input"]
        .as_array()
        .expect("proxy should rebuild input from rehydrated history");
    let text_part = input
        .iter()
        .filter_map(|item| item.get("content").and_then(serde_json::Value::as_array))
        .flatten()
        .find(|part| part.get("type").and_then(serde_json::Value::as_str) == Some("input_text"))
        .expect("rehydrated file should be extracted to input_text by doc_extract");

    let text = text_part["text"]
        .as_str()
        .expect("extracted input_text should have a text field");
    assert!(
        text.contains("[Source: test.txt]"),
        "extracted text should include filename prefix: {text}"
    );
    assert!(
        text.contains("Hello, world!"),
        "extracted text should include file content: {text}"
    );
}

#[test]
fn full_flow_stateful_valid_request_reaches_backend() {
    let backend_guard = start_backend_with_shutdown("inference-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/full-flow.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello, world!"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "stateful request should pass validation and reach the backend"
    );
    assert_eq!(
        parse_body(&raw),
        "inference-backend",
        "stateful request should route to the shared inference backend"
    );
}

#[test]
fn full_flow_stateless_valid_request_reaches_same_backend() {
    let backend_guard = start_backend_with_shutdown("inference-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/full-flow.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello","store":false}"#),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "stateless request should pass validation and reach the backend"
    );
    assert_eq!(
        parse_body(&raw),
        "inference-backend",
        "stateless request should route to the shared inference backend"
    );
}

#[test]
fn full_flow_chat_completions_body_on_responses_path_does_not_reach_backend() {
    let backend_guard = start_backend_with_shutdown("inference-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/full-flow.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#,
        ),
    );

    assert_eq!(
        parse_status(&raw),
        404,
        "chat completions bodies should not match the Responses-only route"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_previous_response_id_rebuilds_body_with_history() {
    let backend_guard = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("full_flow_prev");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/full-flow.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );
    assert_eq!(parse_status(&raw), 200, "first request should succeed");

    drop(backend_guard);

    let echo_backend = start_echo_backend();
    let patched2 = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", echo_backend.port())]),
    );
    let config2 = praxis_core::config::Config::from_yaml(&patched2).expect("second patched config should parse");
    drop(proxy);

    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"What next?","previous_response_id":"resp_first"}"#,
        ),
    );
    let status2 = parse_status(&raw2);
    let body2 = parse_body(&raw2);
    assert_eq!(
        status2, 200,
        "second request with previous_response_id should succeed, body: {body2}"
    );

    let echoed: serde_json::Value = serde_json::from_str(&body2).expect("echoed request body should be valid JSON");

    assert_eq!(echoed["model"], "gpt-4.1", "model should be preserved");

    let input = echoed["input"]
        .as_array()
        .expect("input should be an array after body rebuild");
    assert!(
        input.len() >= 2,
        "input should contain stored history + new message, got {input_len}",
        input_len = input.len()
    );

    let last = input.last().expect("input should not be empty");
    assert_eq!(last["content"], "What next?", "last message should be the new input");

    assert!(
        echoed.get("previous_response_id").is_none(),
        "previous_response_id should be stripped from outbound body"
    );

    drop(proxy2);
}

/// Bound a stalled opening handshake so the integration suite cannot hang.
#[tokio::test]
async fn websocket_handshake_timeout_is_bounded() {
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    let stalled_server = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    });

    let error =
        connect_websocket_with_timeout(format!("ws://127.0.0.1:{port}/v1/responses"), Duration::from_millis(25))
            .await
            .expect_err("stalled WebSocket handshake should time out");

    assert!(
        matches!(&error, Error::Io(error) if error.kind() == std::io::ErrorKind::TimedOut),
        "expected timed-out I/O error, got {error:?}"
    );
    stalled_server.abort();
}

/// Preserve handshake metadata, ordered text frames, and the close frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_websocket_upgrade_preserves_handshake_and_ordered_text() {
    let first = r#"{"type":"response.created","sequence_number":0}"#;
    let second = r#"{"type":"response.output_text.delta","sequence_number":1,"delta":"PONG"}"#;
    let mut backend = start_scripted_websocket_backend(vec![
        WsServerAction::Text(first.to_owned()),
        WsServerAction::Text(second.to_owned()),
        WsServerAction::Close {
            code: 1000,
            reason: "complete".to_owned(),
        },
    ])
    .await;
    let proxy_port = free_port();
    let (config, _db) = isolated_full_flow_config(
        "websocket_upgrade",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let _proxy = start_proxy(&config);

    let mut request = format!("ws://127.0.0.1:{proxy_port}/v1/responses")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert(http::header::AUTHORIZATION, "Bearer test-token".parse().unwrap());
    let (mut socket, response) = connect_websocket(request)
        .await
        .expect("WebSocket handshake should succeed");
    assert_eq!(
        response.status(),
        http::StatusCode::SWITCHING_PROTOCOLS,
        "the full-flow backend should accept the opening handshake"
    );

    let create = r#"{"type":"response.create","response":{"input":"PING"}}"#;
    socket.send(Message::Text(create.into())).await.unwrap();
    assert_eq!(
        next_ws_message(&mut socket).await.into_text().unwrap(),
        first,
        "the first server frame should preserve its payload and order"
    );
    assert_eq!(
        next_ws_message(&mut socket).await.into_text().unwrap(),
        second,
        "the second server frame should preserve its payload and order"
    );
    let close = next_ws_message(&mut socket).await;
    let Message::Close(Some(frame)) = close else {
        panic!("expected close frame, got {close:?}");
    };
    assert_eq!(
        u16::from(frame.code),
        1000,
        "the close status should pass through unchanged"
    );
    assert_eq!(
        frame.reason, "complete",
        "the close reason should pass through unchanged"
    );

    let handshake = next_backend_event(&mut backend).await;
    let WsBackendEvent::Handshake { headers, method, path } = handshake else {
        panic!("expected handshake event, got {handshake:?}");
    };
    assert_eq!(method, http::Method::GET, "the backend should receive the opening GET");
    assert_eq!(
        path, "/v1/responses",
        "the backend should receive the Responses endpoint"
    );
    assert_eq!(
        headers.get(http::header::AUTHORIZATION).unwrap(),
        "Bearer test-token",
        "the authorization header should reach the backend"
    );
    assert_eq!(
        next_backend_event(&mut backend).await,
        WsBackendEvent::ClientMessage(CapturedWsMessage::Text(create.to_owned())),
        "the client text frame should pass through unchanged"
    );
}

/// Preserve arbitrary binary payloads in both tunnel directions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_websocket_preserves_binary_frames_bidirectionally() {
    let server_payload = bytes::Bytes::from_static(&[0x00, 0x7F, 0x80, 0xFF]);
    let mut backend = start_scripted_websocket_backend(vec![WsServerAction::Binary(server_payload.clone())]).await;
    let proxy_port = free_port();
    let (config, _db) = isolated_full_flow_config(
        "websocket_binary",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let _proxy = start_proxy(&config);
    let url = format!("ws://127.0.0.1:{proxy_port}/v1/responses");
    let (mut socket, _) = connect_websocket(url)
        .await
        .expect("binary-frame WebSocket handshake should succeed");
    let client_payload = bytes::Bytes::from_static(&[0xFF, 0x80, 0x7F, 0x00]);

    socket
        .send(Message::Binary(client_payload.clone()))
        .await
        .expect("client binary frame should be sent");

    assert_eq!(
        next_ws_message(&mut socket).await,
        Message::Binary(server_payload),
        "server binary payload should pass through unchanged"
    );
    let handshake = next_backend_event(&mut backend).await;
    assert!(
        matches!(handshake, WsBackendEvent::Handshake { .. }),
        "backend should observe the WebSocket handshake before data frames"
    );
    assert_eq!(
        next_backend_event(&mut backend).await,
        WsBackendEvent::ClientMessage(CapturedWsMessage::Binary(client_payload)),
        "client binary payload should pass through unchanged"
    );
}

/// Keep an idle upgraded connection alive while relaying control frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_websocket_survives_idle_and_relays_ping_pong() {
    let mut backend = start_scripted_websocket_backend(vec![
        WsServerAction::Ping(vec![7, 8, 9].into()),
        WsServerAction::Delay(Duration::from_millis(750)),
        WsServerAction::Text("after-idle".to_owned()),
    ])
    .await;
    let proxy_port = free_port();
    let (config, _db) = isolated_full_flow_config(
        "websocket_idle",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let _proxy = start_proxy(&config);
    let url = format!("ws://127.0.0.1:{proxy_port}/v1/responses");
    let (mut socket, _) = connect_websocket(url).await.unwrap();

    socket.send(Message::Text("start".into())).await.unwrap();
    assert_eq!(
        next_ws_message(&mut socket).await,
        Message::Ping(vec![7, 8, 9].into()),
        "the server ping should pass through unchanged"
    );
    socket.flush().await.unwrap();
    let after_idle = tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("connection should remain alive during the idle window")
        .unwrap()
        .unwrap();
    assert_eq!(
        after_idle.into_text().unwrap(),
        "after-idle",
        "the connection should carry data after the idle interval"
    );

    let mut saw_pong = false;
    for _ in 0..3 {
        if let WsBackendEvent::ClientMessage(CapturedWsMessage::Pong(payload)) = next_backend_event(&mut backend).await
            && payload == bytes::Bytes::from_static(&[7, 8, 9])
        {
            saw_pong = true;
            break;
        }
    }
    assert!(saw_pong, "backend should receive the client's automatic pong");

    socket
        .send(Message::Close(Some(CloseFrame {
            code: CloseCode::Away,
            reason: "client done".into(),
        })))
        .await
        .unwrap();
    let close = next_backend_event(&mut backend).await;
    assert_eq!(
        close,
        WsBackendEvent::ClientMessage(CapturedWsMessage::Close {
            code: Some(1001),
            reason: Some("client done".to_owned()),
        }),
        "the client close frame should pass through unchanged"
    );
}

/// Propagate an abrupt upstream disconnect within the test timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_websocket_early_backend_disconnect_is_bounded() {
    let backend = start_scripted_websocket_backend(vec![WsServerAction::Disconnect]).await;
    let proxy_port = free_port();
    let (config, _db) = isolated_full_flow_config(
        "websocket_disconnect",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let _proxy = start_proxy(&config);
    let url = format!("ws://127.0.0.1:{proxy_port}/v1/responses");
    let (mut socket, _) = connect_websocket(&url).await.unwrap();

    socket.send(Message::Text("disconnect".into())).await.unwrap();
    let ended = tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("early backend disconnect should propagate promptly");
    assert!(
        ended.is_none() || ended.is_some_and(|result| result.is_err()),
        "disconnect should end the stream without a successful data message"
    );

    let (second, response) = connect_websocket(&url)
        .await
        .expect("backend listener should remain healthy");
    assert_eq!(
        response.status(),
        http::StatusCode::SWITCHING_PROTOCOLS,
        "an abrupt connection must not terminate the backend listener"
    );
    drop(second);
}

/// Preserve an upstream HTTP rejection instead of entering tunnel mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_websocket_non_101_backend_response_remains_http() {
    let backend = Backend::status(426, "upgrade rejected").start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = isolated_full_flow_config(
        "websocket_non_101",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let _proxy = start_proxy(&config);
    let url = format!("ws://127.0.0.1:{proxy_port}/v1/responses");

    let error = connect_websocket(url)
        .await
        .expect_err("backend should reject the upgrade");
    let Error::Http(response) = error else {
        panic!("expected HTTP handshake error, got {error:?}");
    };
    assert_eq!(
        response.status(),
        http::StatusCode::UPGRADE_REQUIRED,
        "the upstream non-101 status should remain an HTTP response"
    );
    assert_eq!(
        response.headers().get(http::header::CONTENT_TYPE).unwrap(),
        "application/json",
        "the normal error filter should format the HTTP rejection as JSON"
    );
    let body: serde_json::Value = serde_json::from_slice(response.body().as_deref().unwrap()).unwrap();
    assert_eq!(
        body["error"]["code"], "426",
        "the formatted error should retain the upstream status code"
    );
    assert_eq!(
        body["error"]["message"], "upstream error (HTTP 426)",
        "the formatted error should describe the upstream rejection"
    );
}

/// Keep the bodyless handshake out of the HTTP stateful branch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_flow_websocket_handshake_does_not_take_stateful_route() {
    let ws_backend = start_scripted_websocket_backend(vec![WsServerAction::Text("native".to_owned())]).await;
    let stateful_backend = Backend::fixed("stateful-backend").start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("websocket_routing");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/full-flow.yaml"))
        .expect("example config should exist")
        .replace("sqlite://responses.db?mode=rwc", db.url())
        .replace(
            "          - path: \"/v1/responses\"\n            headers:\n              x-praxis-ai-format: \"openai_responses\"",
            "          - path: \"/v1/responses\"\n            headers:\n              x-praxis-responses-mode: \"stateful\"\n            cluster: \"stateful-backend\"\n\n          - path: \"/v1/responses\"\n            headers:\n              x-praxis-ai-format: \"openai_responses\"",
        )
        .replace(
            "              - \"127.0.0.1:3001\"",
            "              - \"127.0.0.1:3001\"\n          - name: \"stateful-backend\"\n            endpoints:\n              - \"127.0.0.1:3002\"",
        );
    let patched = patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([
            ("127.0.0.1:3001", ws_backend.port()),
            ("127.0.0.1:3002", stateful_backend.port()),
        ]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("routing config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", r#"{"input":"stateful"}"#));
    assert_eq!(
        parse_body(&raw),
        "stateful-backend",
        "control request should use stateful route"
    );

    let url = format!("ws://127.0.0.1:{proxy_port}/v1/responses");
    let (mut socket, response) = connect_websocket(url).await.expect("handshake should use native route");
    assert_eq!(
        response.status(),
        http::StatusCode::SWITCHING_PROTOCOLS,
        "the handshake should reach the format-only inference route"
    );
    socket.send(Message::Text("start".into())).await.unwrap();
    assert_eq!(
        next_ws_message(&mut socket).await.into_text().unwrap(),
        "native",
        "the handshake should not enter the stateful HTTP route"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Successful WebSocket client connection and handshake response.
type WebSocketConnectResult = Result<(WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>, Response), Error>;

/// Receive one proxied `WebSocket` message with the plan's test bound.
async fn next_ws_message<S>(socket: &mut WebSocketStream<S>) -> Message
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("WebSocket receive should complete within five seconds")
        .expect("WebSocket stream should remain open")
        .expect("WebSocket message should be valid")
}

/// Connect a test client within the standard handshake timeout.
async fn connect_websocket<R>(request: R) -> WebSocketConnectResult
where
    R: tokio_tungstenite::tungstenite::client::IntoClientRequest + Unpin,
{
    connect_websocket_with_timeout(request, WEBSOCKET_HANDSHAKE_TIMEOUT).await
}

/// Connect a test client within an explicit handshake timeout.
async fn connect_websocket_with_timeout<R>(request: R, timeout: Duration) -> WebSocketConnectResult
where
    R: tokio_tungstenite::tungstenite::client::IntoClientRequest + Unpin,
{
    tokio::time::timeout(timeout, Box::pin(connect_async(request)))
        .await
        .map_err(|_elapsed| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "WebSocket handshake exceeded test timeout",
            ))
        })?
}

/// Receive one backend observation with the plan's test bound.
async fn next_backend_event(backend: &mut praxis_test_utils::WsBackendGuard) -> WsBackendEvent {
    tokio::time::timeout(Duration::from_secs(5), backend.next_event())
        .await
        .expect("backend observation should arrive within five seconds")
        .expect("backend observation channel should remain open")
}

/// Load the full-flow example with isolated persistence and patched ports.
fn isolated_full_flow_config(
    test_name: &str,
    proxy_port: u16,
    ports: &HashMap<&str, u16>,
) -> (praxis_core::config::Config, TempSqlite) {
    let db = TempSqlite::new(test_name);
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/full-flow.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        ports,
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    (config, db)
}
