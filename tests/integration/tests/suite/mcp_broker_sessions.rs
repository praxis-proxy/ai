// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for current-profile MCP broker backend sessions against
//! session-enforcing MCP backends.

use praxis_core::config::Config;
use praxis_test_utils::{
    McpMockConfig, McpMockServerGuard, McpToolFixture, free_port, http_send, parse_body, parse_header, parse_status,
    start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn current_profile_tools_call_initializes_backend_session_then_forwards() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let raw = http_send(proxy.addr(), &tools_call(&gateway, 3, "weather_get_weather"));

    assert_eq!(parse_status(&raw), 200, "tools/call should succeed: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(body["id"], 3, "client request id must be preserved: {body}");
    assert_eq!(
        body["result"]["content"][0]["text"], "mock result for get_weather",
        "backend result must reach the client: {body}"
    );
    assert_eq!(
        weather.method_count("initialize"),
        1,
        "backend must be initialized once"
    );
    assert_eq!(
        weather.method_count("notifications/initialized"),
        1,
        "backend must receive notifications/initialized"
    );
    let call_sessions = weather.session_ids_for("tools/call");
    assert_eq!(
        call_sessions,
        vec![Some("mock-mcp-session-1".to_owned())],
        "tools/call must carry the backend session, not the gateway session"
    );
    assert_eq!(
        weather.session_ids_for("notifications/initialized"),
        call_sessions,
        "notifications/initialized must use the same backend session"
    );
    assert_eq!(
        weather.last_tool_call_name().as_deref(),
        Some("get_weather"),
        "prefix stripped"
    );
    assert_eq!(
        calendar.method_count("initialize"),
        0,
        "untouched servers are not initialized"
    );
}

#[test]
fn current_profile_reuses_backend_session_across_calls() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    for id in 1..=3 {
        let raw = http_send(proxy.addr(), &tools_call(&gateway, id, "weather_get_weather"));
        assert_eq!(parse_status(&raw), 200, "call {id} should succeed: {raw}");
    }

    assert_eq!(
        weather.method_count("initialize"),
        1,
        "an existing session must be reused"
    );
    assert_eq!(
        weather.tool_call_count("get_weather"),
        3,
        "every call must reach the backend"
    );
}

#[test]
fn current_profile_scopes_backend_sessions_per_gateway_session_and_server() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let first = initialize_gateway(proxy.addr());
    let second = initialize_gateway(proxy.addr());

    assert_eq!(
        parse_status(&http_send(proxy.addr(), &tools_call(&first, 1, "weather_get_weather"))),
        200
    );
    assert_eq!(
        parse_status(&http_send(proxy.addr(), &tools_call(&second, 2, "weather_get_weather"))),
        200
    );
    assert_eq!(
        parse_status(&http_send(proxy.addr(), &tools_call(&first, 3, "cal_create_event"))),
        200
    );

    assert_eq!(
        weather.sessions_minted(),
        2,
        "each gateway session gets its own backend session"
    );
    assert_eq!(
        calendar.sessions_minted(),
        1,
        "each server gets its own backend session"
    );
}

#[test]
fn current_profile_backend_404_reinitializes_on_next_call() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());
    assert_eq!(
        parse_status(&http_send(
            proxy.addr(),
            &tools_call(&gateway, 1, "weather_get_weather")
        )),
        200
    );

    weather.expire_sessions();
    let expired = http_send(proxy.addr(), &tools_call(&gateway, 2, "weather_get_weather"));

    assert_backend_session_expired(&expired, 2);
    assert_eq!(
        weather.tool_call_count("get_weather"),
        2,
        "the expired call must be dispatched once, not retried"
    );
    assert_eq!(weather.sessions_minted(), 1, "the handshake waits for the next call");

    let recovered = http_send(proxy.addr(), &tools_call(&gateway, 3, "weather_get_weather"));

    assert_eq!(
        parse_status(&recovered),
        200,
        "the next call on the same gateway session must reinitialize: {recovered}"
    );
    assert_eq!(weather.sessions_minted(), 2, "a fresh backend session must be created");
    assert_eq!(
        weather.session_ids_for("tools/call"),
        ["mock-mcp-session-1", "mock-mcp-session-1", "mock-mcp-session-2"].map(|id| Some(id.to_owned())),
        "the recovered call must use the new backend session"
    );
}

#[test]
fn current_profile_backend_404_on_first_forwarded_call_keeps_gateway_session() {
    let weather = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        expire_sessions_once_initialized: true,
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let expired = http_send(proxy.addr(), &tools_call(&gateway, 1, "weather_get_weather"));

    assert_backend_session_expired(&expired, 1);
    assert_eq!(
        weather.tool_call_count("get_weather"),
        1,
        "the expired call must be dispatched once, not retried"
    );

    let next = http_send(proxy.addr(), &tools_call(&gateway, 2, "weather_get_weather"));

    assert_backend_session_expired(&next, 2);
    assert_eq!(weather.sessions_minted(), 2, "the next call must run a fresh handshake");
    let other = http_send(proxy.addr(), &tools_call(&gateway, 3, "cal_create_event"));
    assert_eq!(
        parse_status(&other),
        200,
        "the gateway session must stay usable: {other}"
    );
}

#[test]
fn current_profile_sessionless_backend_404_keeps_gateway_session() {
    let weather = start_mcp_mock_server_with_config(McpMockConfig {
        stateful_sessions: false,
        tools_call_not_found: true,
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let rejected = http_send(proxy.addr(), &tools_call(&gateway, 1, "weather_get_weather"));

    assert_eq!(
        parse_status(&rejected),
        200,
        "a backend 404 must not reach the client: {rejected}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&rejected)).unwrap();
    assert_eq!(body["id"], 1, "client request id must be preserved: {body}");
    assert_eq!(body["error"]["code"], -32603, "a rejected call must be -32603: {body}");
    assert_eq!(
        body["error"]["data"],
        serde_json::json!({ "server": "weather", "reason": "rejected", "status": 404 }),
        "the error must name the server, the reason, and the backend status: {body}"
    );
    assert_eq!(
        weather.session_ids_for("tools/call"),
        [None],
        "the call must be dispatched once, without a session"
    );
    let other = http_send(proxy.addr(), &tools_call(&gateway, 2, "cal_create_event"));
    assert_eq!(
        parse_status(&other),
        200,
        "the gateway session must stay usable: {other}"
    );
}

#[test]
fn current_profile_backend_404_leaves_other_backend_sessions_in_use() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());
    for (id, tool) in [(1, "weather_get_weather"), (2, "cal_create_event")] {
        let raw = http_send(proxy.addr(), &tools_call(&gateway, id, tool));
        assert_eq!(parse_status(&raw), 200, "{tool} must succeed: {raw}");
    }

    weather.expire_sessions();
    let expired = http_send(proxy.addr(), &tools_call(&gateway, 3, "weather_get_weather"));
    let other = http_send(proxy.addr(), &tools_call(&gateway, 4, "cal_create_event"));

    assert_backend_session_expired(&expired, 3);
    assert_eq!(
        parse_status(&other),
        200,
        "the calendar call must succeed on the same gateway session: {other}"
    );
    assert_eq!(
        calendar.method_count("initialize"),
        1,
        "the calendar session must not be reinitialized"
    );
    assert_eq!(
        calendar.session_ids_for("tools/call"),
        vec![Some("mock-mcp-session-1".to_owned()); 2],
        "both calendar calls must use the original backend session"
    );
}

#[test]
fn current_profile_failed_handshake_after_backend_404_keeps_gateway_session() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());
    assert_eq!(
        parse_status(&http_send(
            proxy.addr(),
            &tools_call(&gateway, 1, "weather_get_weather")
        )),
        200
    );
    weather.expire_sessions();
    assert_backend_session_expired(
        &http_send(proxy.addr(), &tools_call(&gateway, 2, "weather_get_weather")),
        2,
    );
    let weather_endpoint = weather.endpoint();
    drop(weather);
    wait_until_refused(&weather_endpoint);

    let failed = http_send(proxy.addr(), &tools_call(&gateway, 3, "weather_get_weather"));
    let other = http_send(proxy.addr(), &tools_call(&gateway, 4, "cal_create_event"));

    assert_eq!(
        parse_status(&failed),
        200,
        "a failed handshake is a JSON-RPC error: {failed}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&failed)).unwrap();
    assert_eq!(body["id"], 3, "client request id must be preserved: {body}");
    assert_eq!(
        body["error"]["message"], "backend session initialization failed",
        "the fresh handshake failure must be reported: {body}"
    );
    assert_eq!(
        parse_status(&other),
        200,
        "the gateway session must stay usable: {other}"
    );
}

#[test]
fn current_profile_pins_backend_session_to_its_endpoint() {
    // Two replicas of the weather server behind one cluster. Each replica only
    // knows the sessions it minted, so a handshake or call that lands on the
    // other replica gets a 404. Round-robin would alternate between them.
    let replica_a = start_session_backend("get_weather");
    let replica_b = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let yaml = session_broker_yaml(&replica_a, &calendar, "").replacen(
        &format!("endpoints: [\"{}\"]", replica_a.endpoint()),
        &format!(
            "endpoints: [\"{}\", \"{}\"]",
            replica_a.endpoint(),
            replica_b.endpoint()
        ),
        1,
    );
    let proxy = start_proxy(&Config::from_yaml(&yaml).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    for id in 1..=4 {
        let raw = http_send(proxy.addr(), &tools_call(&gateway, id, "weather_get_weather"));
        assert_eq!(
            parse_status(&raw),
            200,
            "call {id} must reach the session's replica: {raw}"
        );
    }

    let (owner, other) = if replica_a.sessions_minted() == 1 {
        (&replica_a, &replica_b)
    } else {
        (&replica_b, &replica_a)
    };
    assert_eq!(owner.sessions_minted(), 1, "exactly one backend session must exist");
    assert_eq!(other.sessions_minted(), 0, "the other replica must not be initialized");
    assert_eq!(
        owner.method_count("notifications/initialized"),
        1,
        "notifications/initialized must reach the replica that minted the session"
    );
    assert_eq!(
        owner.tool_call_count("get_weather"),
        4,
        "every call must reach the replica that owns the session"
    );
    assert_eq!(
        other.received_requests().len(),
        0,
        "nothing may reach the replica without the session"
    );
}

#[test]
fn current_profile_unreachable_pinned_replica_reinitializes_on_survivor() {
    // The replica that issued the session goes away. Core answers the pinned
    // call with a 502 from its connect-failure path, which never reaches the
    // response hook; the session must still be dropped so the next call
    // initializes on the surviving replica instead of retrying the dead one.
    let replica_a = start_session_backend("get_weather");
    let replica_b = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let yaml = session_broker_yaml(&replica_a, &calendar, "").replacen(
        &format!("endpoints: [\"{}\"]", replica_a.endpoint()),
        &format!(
            "endpoints: [\"{}\", \"{}\"]",
            replica_a.endpoint(),
            replica_b.endpoint()
        ),
        1,
    );
    let proxy = start_proxy(&Config::from_yaml(&yaml).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let first = http_send(proxy.addr(), &tools_call(&gateway, 1, "weather_get_weather"));
    assert_eq!(parse_status(&first), 200, "first call must succeed: {first}");

    let (owner, survivor) = if replica_a.sessions_minted() == 1 {
        (replica_a, replica_b)
    } else {
        (replica_b, replica_a)
    };
    let owner_endpoint = owner.endpoint();
    drop(owner);
    wait_until_refused(&owner_endpoint);

    let failed = http_send(proxy.addr(), &tools_call(&gateway, 2, "weather_get_weather"));
    assert_eq!(
        parse_status(&failed),
        502,
        "a call pinned to the stopped replica must fail: {failed}"
    );

    let recovered = http_send(proxy.addr(), &tools_call(&gateway, 3, "weather_get_weather"));
    assert_eq!(
        parse_status(&recovered),
        200,
        "the next call must initialize on the surviving replica: {recovered}"
    );
    assert_eq!(
        survivor.sessions_minted(),
        1,
        "the surviving replica must mint the replacement session"
    );
    assert_eq!(
        survivor.tool_call_count("get_weather"),
        1,
        "the recovered call must reach the surviving replica"
    );
}

#[test]
fn current_profile_rejects_oversized_chunked_handshake() {
    let weather = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        oversized_initialize_bytes: Some(256 * 1024),
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(
        &Config::from_yaml(&session_broker_yaml(&weather, &calendar, "max_response_bytes: 16384")).unwrap(),
    );
    let gateway = initialize_gateway(proxy.addr());

    let raw = http_send(proxy.addr(), &tools_call(&gateway, 4, "weather_get_weather"));

    assert_eq!(parse_status(&raw), 200, "handshake failure is a JSON-RPC error: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        body["error"]["code"], -32603,
        "handshake failure must be -32603: {body}"
    );
    assert_eq!(
        body["error"]["data"]["reason"], "response_too_large",
        "oversized handshake must be classified: {body}"
    );
    assert_eq!(
        weather.tool_call_count("get_weather"),
        0,
        "the call must not be forwarded"
    );
}

#[test]
fn current_profile_handshake_completes_on_sse_reply_that_stays_open() {
    let weather = start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        sse_initialize_held_open: true,
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let started = std::time::Instant::now();
    let raw = http_send(proxy.addr(), &tools_call(&gateway, 5, "weather_get_weather"));

    assert_eq!(parse_status(&raw), 200, "tools/call should succeed: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        body["result"]["content"][0]["text"], "mock result for get_weather",
        "the handshake must finish without waiting for the stream to close: {body}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(4),
        "the handshake must not wait for the 5 s handshake timeout"
    );
    assert_eq!(
        weather.session_ids_for("tools/call"),
        vec![Some("mock-mcp-session-1".to_owned())],
        "the call must carry the session from the SSE handshake"
    );
}

#[test]
fn current_profile_handshake_answers_backend_ping_before_initialize_result() {
    // Two replicas behind one cluster, each holding its `initialize` result
    // until it receives the pong. The default round-robin load balancer picks
    // by a monotonic counter, so an unpinned pong would be the handshake
    // chain's next pick and land on the other replica, which rejects the
    // unknown session and fails the handshake.
    let ping_backend = || {
        start_mcp_mock_server_with_config(McpMockConfig {
            enforce_sessions: true,
            ping_before_initialize_result: true,
            tools: vec![McpToolFixture::new("get_weather")],
            ..McpMockConfig::default()
        })
    };
    let (replica_a, replica_b) = (ping_backend(), ping_backend());
    let calendar = start_session_backend("create_event");
    let yaml = session_broker_yaml(&replica_a, &calendar, "").replacen(
        &format!("endpoints: [\"{}\"]", replica_a.endpoint()),
        &format!(
            "endpoints: [\"{}\", \"{}\"]",
            replica_a.endpoint(),
            replica_b.endpoint()
        ),
        1,
    );
    let proxy = start_proxy(&Config::from_yaml(&yaml).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let call = tools_call(&gateway, 7, "weather_get_weather").replacen(
        "Host: localhost\r\n",
        "Host: localhost\r\nAuthorization: Bearer ping-token\r\n",
        1,
    );
    let raw = http_send(proxy.addr(), &call);

    assert_eq!(parse_status(&raw), 200, "tools/call should succeed: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        body["result"]["content"][0]["text"], "mock result for get_weather",
        "the handshake must complete once the ping is answered: {body}"
    );
    let (owner, other) = if replica_a.sessions_minted() == 1 {
        (&replica_a, &replica_b)
    } else {
        (&replica_b, &replica_a)
    };
    assert_eq!(
        other.received_requests().len(),
        0,
        "nothing may reach the replica without the session"
    );
    let pongs: Vec<_> = owner
        .received_requests()
        .into_iter()
        .filter(|r| r.json_rpc_method.is_none())
        .collect();
    assert_eq!(pongs.len(), 1, "the ping is answered once, on the session's replica");
    let pong: serde_json::Value = serde_json::from_str(&pongs[0].body).unwrap();
    assert_eq!(
        pong,
        serde_json::json!({ "jsonrpc": "2.0", "id": "mock-ping-mock-mcp-session-1", "result": {} }),
        "the pong echoes the ping id with an empty result"
    );
    let header = |name: &str| {
        pongs[0]
            .headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    assert_eq!(
        header("authorization"),
        Some("Bearer ping-token"),
        "credential forwarded"
    );
    assert_eq!(header("mcp-session-id"), Some("mock-mcp-session-1"), "session id sent");
    assert_eq!(
        replica_a.sessions_minted() + replica_b.sessions_minted(),
        1,
        "one backend session"
    );
    assert_eq!(
        owner.session_ids_for("tools/call"),
        vec![Some("mock-mcp-session-1".to_owned())],
        "the call carries the session from the pinged handshake"
    );
}

#[test]
fn current_profile_handshake_works_through_a_buffering_outbound_filter() {
    // `json_body` buffers response bodies, so the handshake reply is read
    // whole instead of streamed.
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let extra = "outbound_filters:\n            - filter: json_body\n              content_types: [application/json]\n              request_extract:\n                - pointer: /method\n                  metadata: mcp.method";
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, extra)).unwrap());
    let gateway = initialize_gateway(proxy.addr());

    let raw = http_send(proxy.addr(), &tools_call(&gateway, 6, "weather_get_weather"));

    assert_eq!(parse_status(&raw), 200, "tools/call should succeed: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        body["result"]["content"][0]["text"], "mock result for get_weather",
        "a chain that buffers responses must still complete the handshake: {body}"
    );
    assert_eq!(weather.method_count("initialize"), 1, "one handshake");
}

#[test]
fn current_profile_tools_call_without_gateway_session_is_rejected() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());

    let raw = http_send(
        proxy.addr(),
        &json_post("/mcp", &call_body(1, "weather_get_weather"), None),
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "missing Mcp-Session-Id must be rejected: {raw}"
    );
    assert_eq!(
        weather.method_count("initialize"),
        0,
        "no backend handshake without a session"
    );
}

#[test]
fn current_profile_delete_terminates_backend_sessions() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());
    assert_eq!(
        parse_status(&http_send(
            proxy.addr(),
            &tools_call(&gateway, 1, "weather_get_weather")
        )),
        200
    );

    let raw = http_send(
        proxy.addr(),
        &format!("DELETE /mcp HTTP/1.1\r\nHost: localhost\r\nMcp-Session-Id: {gateway}\r\nConnection: close\r\n\r\n"),
    );

    assert_eq!(parse_status(&raw), 204, "gateway DELETE must succeed");
    let deletes: Vec<_> = weather
        .received_requests()
        .into_iter()
        .filter(|r| r.http_method == "DELETE")
        .collect();
    assert_eq!(deletes.len(), 1, "the backend session must be terminated");
}

#[test]
fn current_profile_never_issued_gateway_session_is_not_found() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());

    let raw = http_send(proxy.addr(), &tools_call("mcp-forged", 1, "weather_get_weather"));

    assert_eq!(parse_status(&raw), 404, "a never-issued id must be a 404: {raw}");
    assert_eq!(
        weather.method_count("initialize"),
        0,
        "a never-issued id must open no backend session"
    );
}

#[test]
fn current_profile_ended_gateway_session_is_not_found_until_reinitialized() {
    let weather = start_session_backend("get_weather");
    let calendar = start_session_backend("create_event");
    let proxy = start_proxy(&Config::from_yaml(&session_broker_yaml(&weather, &calendar, "")).unwrap());
    let gateway = initialize_gateway(proxy.addr());
    let delete =
        format!("DELETE /mcp HTTP/1.1\r\nHost: localhost\r\nMcp-Session-Id: {gateway}\r\nConnection: close\r\n\r\n");
    assert_eq!(
        parse_status(&http_send(proxy.addr(), &delete)),
        204,
        "DELETE must end the session"
    );

    let raw = http_send(proxy.addr(), &tools_call(&gateway, 1, "weather_get_weather"));
    assert_eq!(parse_status(&raw), 404, "an ended id must be a 404: {raw}");
    assert_eq!(
        weather.method_count("initialize"),
        0,
        "an ended id must open no backend session"
    );
    let get = format!("GET /mcp HTTP/1.1\r\nHost: localhost\r\nMcp-Session-Id: {gateway}\r\nConnection: close\r\n\r\n");
    assert_eq!(
        parse_status(&http_send(proxy.addr(), &get)),
        404,
        "a reconnecting GET with an ended id must be a 404"
    );
    let reinit =
        r#"{"jsonrpc":"2.0","id":3,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#;
    let raw = http_send(proxy.addr(), &json_post("/mcp", reinit, Some(&gateway)));
    assert_eq!(
        parse_status(&raw),
        404,
        "initialize carrying an ended id must be a 404: {raw}"
    );

    let fresh = initialize_gateway(proxy.addr());
    let raw = http_send(proxy.addr(), &tools_call(&fresh, 2, "weather_get_weather"));
    assert_eq!(
        parse_status(&raw),
        200,
        "a new initialize must issue a working id: {raw}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        body["result"]["content"][0]["text"], "mock result for get_weather",
        "the fresh id's call must succeed at the backend: {raw}"
    );
    assert_eq!(
        weather.tool_call_count("get_weather"),
        1,
        "only the fresh id's call reaches the backend"
    );
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Start a session-enforcing MCP backend exposing one tool.
fn start_session_backend(tool: &str) -> McpMockServerGuard {
    start_mcp_mock_server_with_config(McpMockConfig {
        enforce_sessions: true,
        tools: vec![McpToolFixture::new(tool)],
        ..McpMockConfig::default()
    })
}

/// Broker config with `backend_sessions`; `extra` adds session settings.
fn session_broker_yaml(weather: &McpMockServerGuard, calendar: &McpMockServerGuard, extra: &str) -> String {
    let proxy_port = free_port();
    let extra = if extra.is_empty() {
        String::new()
    } else {
        format!("\n          {extra}")
    };
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
clusters:
  - name: weather-mcp
    endpoints: ["{weather}"]
  - name: calendar-mcp
    endpoints: ["{calendar}"]
filter_chains:
  - name: main
    filters:
      - filter: mcp
        path: /mcp
        backend_sessions:
          handshake_timeout_ms: 5000{extra}
        servers:
          - name: weather
            cluster: weather-mcp
            path: /mcp
            tool_prefix: "weather_"
            tools:
              - name: get_weather
          - name: calendar
            cluster: calendar-mcp
            path: /mcp
            tool_prefix: "cal_"
            tools:
              - name: create_event
      - filter: load_balancer
        clusters:
          - name: weather-mcp
          - name: calendar-mcp
insecure_options:
  allow_private_endpoints: true
"#,
        weather = weather.endpoint(),
        calendar = calendar.endpoint(),
    )
}

/// Assert `raw` answers call `id` with the broker's backend-session-expired
/// error instead of the backend's `404`.
fn assert_backend_session_expired(raw: &str, id: u64) {
    assert_eq!(parse_status(raw), 200, "a backend 404 must not reach the client: {raw}");
    let body: serde_json::Value = serde_json::from_str(&parse_body(raw)).unwrap();
    assert_eq!(body["id"], id, "client request id must be preserved: {body}");
    assert_eq!(
        body["error"]["code"], -32603,
        "backend session expiry must be -32603: {body}"
    );
    assert_eq!(
        body["error"]["data"],
        serde_json::json!({ "server": "weather", "reason": "session_expired" }),
        "the error must name the server and the reason: {body}"
    );
}

/// Wait until a stopped mock no longer accepts connections on `endpoint`.
fn wait_until_refused(endpoint: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::net::TcpStream::connect(endpoint).is_ok() {
        assert!(
            std::time::Instant::now() < deadline,
            "stopped replica {endpoint} still accepts connections"
        );
        std::thread::yield_now();
    }
}

/// Initialize with the broker and return the gateway session id.
fn initialize_gateway(addr: &str) -> String {
    let body =
        r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}"#;
    let raw = http_send(addr, &json_post("/mcp", body, None));
    assert_eq!(parse_status(&raw), 200, "broker initialize should succeed: {raw}");
    parse_header(&raw, "mcp-session-id").expect("broker must issue a gateway session")
}

/// A `tools/call` on the gateway session.
fn tools_call(gateway: &str, id: u64, tool: &str) -> String {
    json_post("/mcp", &call_body(id, tool), Some(gateway))
}

/// Serialize a `tools/call` body.
fn call_body(id: u64, tool: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"{tool}","arguments":{{}}}}}}"#)
}

/// Build a JSON POST with optional `Mcp-Session-Id`.
fn json_post(path: &str, body: &str, session: Option<&str>) -> String {
    let session = session.map_or_else(String::new, |s| format!("Mcp-Session-Id: {s}\r\n"));
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\n\
         MCP-Protocol-Version: 2025-03-26\r\n\
         {session}\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len(),
    )
}
