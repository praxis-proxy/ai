// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the llm-d ext_proc routing example configuration.

use std::{collections::HashMap, error::Error, io, net::TcpListener, time::Duration};

use praxis_test_utils::{
    free_port, http_post, http_send, json_post, json_post_with_header,
    net::ext_proc_mock::{
        start_mock_processor_without_destination, start_mock_routing_processor_with_body_replacement,
        start_mock_routing_processor_with_reply_header,
    },
    parse_body, parse_header, parse_status, start_backend_with_shutdown, start_capturing_backend,
    start_header_echo_backend, start_mock_routing_processor, start_proxy,
};
use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

/// Route a normal request to the worker returned by the mock processor.
#[test]
fn llmd_ext_proc_routing_example_routes_to_processor_selected_endpoint() {
    let backend = start_backend_with_shutdown("llmd-selected-backend");
    let processor = start_mock_routing_processor(&format!("127.0.0.1:{}", backend.port()));
    let proxy_port = free_port();

    let config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", processor.port())]),
    );
    let proxy = start_proxy(&config);

    let request = json_post_with_header(
        "/v1/chat/completions",
        r#"{"model":"llmd-demo","messages":[{"role":"user","content":"hello"}]}"#,
        concat!(
            "X-Gateway-Base-Model-Name: client-base-fixture\r\n",
            "X-Gateway-Inference-Objective: client-objective-fixture\r\n",
            "X-LLM-D-Inference-Objective: alternate-objective-fixture\r\n",
            "X-LLM-D-Model-Name-Rewrite: client-rewrite-fixture\r\n",
            "X-Gateway-Model-Name-Rewrite: alternate-rewrite-fixture"
        ),
    );
    let response = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&response), 200, "example request should return 200");
    assert_eq!(
        parse_body(&response),
        "llmd-selected-backend",
        "request should route to the processor-selected backend"
    );
    let messages = processor.take_request_headers();
    let headers = messages.first().and_then(|message| message.headers.as_ref());
    assert!(
        headers.is_some_and(|map| map.headers.iter().all(|header| !matches!(
            header.key.as_str(),
            "x-gateway-base-model-name"
                | "x-gateway-inference-objective"
                | "x-llm-d-inference-objective"
                | "x-llm-d-model-name-rewrite"
                | "x-gateway-model-name-rewrite"
        ))),
        "the example should strip unapproved scheduling headers before EPP"
    );
}

#[test]
fn llmd_ext_proc_forwards_epp_replacement_and_streams_before_worker_eof() -> Result<(), Box<dyn Error>> {
    const PATH: &str = "/v1/chat/completions";
    const REQUEST: &str = r#"{"model":"llmd-demo","stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
    const WORKER_REQUEST: &str =
        r#"{"model":"worker-model-name","stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
    const FIRST_EVENT: &str = concat!(
        "data: {\"id\":\"chatcmpl-471\",\"object\":\"chat.completion.chunk\",",
        "\"model\":\"llmd-demo\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
    );
    const LAST_EVENT: &str = "data: [DONE]\n\n";

    let listener = TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let picker = start_mock_routing_processor_with_body_replacement(
        &listener.local_addr()?.to_string(),
        WORKER_REQUEST.as_bytes(),
    );
    let config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", picker.port())]),
    );
    let proxy = start_proxy(&config);
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let (release_worker, wait_for_client) = tokio::sync::oneshot::channel();
        let worker = async {
            let (stream, _) = listener.accept().await?;
            let mut stream = BufReader::new(stream);
            let mut head = String::new();
            loop {
                let mut line = String::new();
                if stream.read_line(&mut line).await? == 0 {
                    return Err(io::Error::other("worker request ended before its headers").into());
                }
                head.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            assert!(
                head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"),
                "the picked worker should receive the chat request"
            );
            let length = parse_header(&head, "content-length")
                .ok_or_else(|| io::Error::other("the worker should receive Content-Length"))?
                .parse::<usize>()?;
            assert_eq!(
                length,
                WORKER_REQUEST.len(),
                "the worker's declared body length should match EPP's replacement"
            );
            let mut body = vec![0; length];
            stream.read_exact(&mut body).await?;
            assert_eq!(body, WORKER_REQUEST.as_bytes(), "the worker should receive EPP's replacement");

            let stream = stream.get_mut();
            let opening = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{FIRST_EVENT}\r\n",
                FIRST_EVENT.len(),
            );
            stream.write_all(opening.as_bytes()).await?;
            stream.flush().await?;
            wait_for_client.await?;
            let closing = format!("{:x}\r\n{LAST_EVENT}\r\n0\r\n\r\n", LAST_EVENT.len());
            stream.write_all(closing.as_bytes()).await?;
            stream.shutdown().await?;
            Ok::<(), Box<dyn Error>>(())
        };

        let client = async {
            let mut stream = tokio::net::TcpStream::connect(proxy.addr()).await?;
            stream.write_all(json_post(PATH, REQUEST).as_bytes()).await?;
            let mut response = Vec::new();
            let mut buffer = [0; 1024];
            loop {
                let count = stream.read(&mut buffer).await?;
                assert_ne!(count, 0, "the proxy closed before delivering the first SSE event");
                response.extend_from_slice(&buffer[..count]);
                if parse_body(&String::from_utf8_lossy(&response)).starts_with(FIRST_EVENT) {
                    break;
                }
            }
            release_worker
                .send(())
                .map_err(|()| io::Error::other("the worker should wait for the first client event"))?;
            stream.read_to_end(&mut response).await?;
            let response = String::from_utf8(response)?;
            assert_eq!(parse_status(&response), 200, "the chat stream should succeed");
            assert_eq!(
                parse_body(&response),
                format!("{FIRST_EVENT}{LAST_EVENT}"),
                "both SSE events should reach the client in order"
            );
            Ok::<(), Box<dyn Error>>(())
        };

        tokio::time::timeout(Duration::from_secs(8), async { tokio::try_join!(worker, client) })
            .await
            .expect("the client must see the first SSE event before the worker can finish")?;
        Ok::<(), Box<dyn Error>>(())
    })?;
    assert_eq!(picker.stream_count(), 1, "the request should use one picker exchange");
    assert_eq!(picker.take_request_bodies(), [REQUEST.as_bytes()]);
    Ok(())
}

/// Verify that the first EPP message reflects BBR and request header policy.
#[test]
fn llmd_ext_proc_first_message_contains_gateway_model_and_objective() {
    let backend = start_capturing_backend("llmd-selected-backend");
    let processor = start_mock_routing_processor(&format!("127.0.0.1:{}", backend.port()));
    let proxy_port = free_port();
    let mut config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", processor.port())]),
    );
    let header_policy = serde_yaml::from_str(
        "
filter: headers
request_remove:
  - x-gateway-destination-endpoint
request_set:
  - name: x-gateway-inference-objective
    value: legacy-batch-471
  - name: x-llm-d-inference-objective
    value: batch-471
  - name: x-gateway-base-model-name
    value: llmd-demo-base
  - name: x-llm-d-model-name-rewrite
    value: worker-model-name
  - name: x-gateway-model-name-rewrite
    value: legacy-worker-model-name
",
    )
    .expect("header policy should parse");
    config.filter_chains[0].filters.insert(2, header_policy);
    let rewrite = serde_yaml::from_str(
        "
filter: json_body
request_add:
  - pointer: /user
    value: trusted-tenant
",
    )
    .expect("body rewrite should parse");
    config.filter_chains[0].filters.insert(3, rewrite);
    let proxy = start_proxy(&config);

    let body = r#"{"model":"llmd-demo","messages":[{"role":"user","content":"hello"}]}"#;
    let request = json_post_with_header(
        "/v1/chat/completions",
        body,
        concat!(
            "X-Gateway-Model-Name: client-model\r\n",
            "X-Gateway-Base-Model-Name: client-base\r\n",
            "X-Gateway-Inference-Objective: client-objective\r\n",
            "X-LLM-D-Inference-Objective: alternate-name-fixture\r\n",
            "X-LLM-D-Model-Name-Rewrite: client-rewrite\r\n",
            "X-Gateway-Model-Name-Rewrite: alternate-rewrite-fixture\r\n",
            "X-Gateway-Destination-Endpoint: reserved-header-fixture"
        ),
    );
    let response = http_send(proxy.addr(), &request);
    assert_eq!(
        parse_status(&response),
        200,
        "the earlier removal must preserve EPP's later pick"
    );
    assert_eq!(
        parse_body(&response),
        "llmd-selected-backend",
        "the response must come from EPP's pick"
    );

    let bodies = processor.take_request_bodies();
    let [rewritten] = bodies.as_slice() else {
        panic!("the picker should receive one complete request body");
    };
    assert_ne!(
        rewritten.len(),
        body.len(),
        "the rewrite should change the length advertised by the original client"
    );
    let json: serde_json::Value = serde_json::from_slice(rewritten).expect("rewritten request should be JSON");
    assert_eq!(
        json["user"], "trusted-tenant",
        "EPP should receive the body after the rewrite"
    );
    assert_eq!(
        backend.body().as_bytes(),
        rewritten.as_slice(),
        "the selected worker should receive the same rewritten bytes"
    );
    let messages = processor.take_request_headers();
    assert_eq!(messages.len(), 1);
    let headers = &messages
        .first()
        .expect("EPP should receive request headers")
        .headers
        .as_ref()
        .expect("RequestHeaders should contain a header map")
        .headers;
    let value = |key: &str| {
        headers
            .iter()
            .filter(|header| header.key.eq_ignore_ascii_case(key))
            .map(|header| {
                if header.raw_value.is_empty() {
                    header.value.as_str()
                } else {
                    std::str::from_utf8(&header.raw_value).expect("test header should be UTF-8")
                }
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        (value("x-gateway-model-name"), value("x-gateway-inference-objective")),
        (vec!["llmd-demo"], vec!["legacy-batch-471"]),
        "the first RequestHeaders message must contain the body-derived model and configured objective"
    );
    assert_eq!(value("x-gateway-base-model-name"), vec!["llmd-demo-base"]);
    assert_eq!(value("x-llm-d-inference-objective"), vec!["batch-471"]);
    assert_eq!(value("x-llm-d-model-name-rewrite"), vec!["worker-model-name"]);
    assert_eq!(value("x-gateway-model-name-rewrite"), vec!["legacy-worker-model-name"]);
    assert_eq!(
        value("content-length"),
        [rewritten.len().to_string().as_str()],
        "the first EPP message should advertise the exact rewritten body length once"
    );
    assert!(
        value("x-gateway-destination-endpoint").is_empty(),
        "the picker should supply the destination"
    );
}

#[test]
fn llmd_ext_proc_requires_a_pick_even_if_a_preceding_policy_sets_a_destination() -> Result<(), serde_yaml::Error> {
    let worker = start_backend_with_shutdown("reserved-destination-fixture");
    let processor = start_mock_processor_without_destination();
    let mut config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", processor.port())]),
    );
    config.filter_chains[0].filters.insert(
        1,
        serde_yaml::from_str(&format!(
            "
filter: headers
request_set:
  - name: x-gateway-destination-endpoint
    value: '127.0.0.1:{}'
",
            worker.port(),
        ))?,
    );
    let filters = &mut config.filter_chains[0].filters;
    filters.pop();
    filters.push(serde_yaml::from_str(
        "
filter: endpoint_selector
source_header: x-gateway-destination-endpoint
required: true
status_on_required_failure: 424
strip_header: true
",
    )?);
    let proxy = start_proxy(&config);
    let (status, _) = http_post(proxy.addr(), "/v1/chat/completions", r#"{"model":"llmd-demo"}"#);
    assert_eq!(
        status, 424,
        "the endpoint selector must reject a completed EPP response with no pick"
    );
    assert_eq!(
        processor.take_request_bodies().len(),
        1,
        "the exchange should reach body EOS"
    );
    let messages = processor.take_request_headers();
    let headers = messages.first().and_then(|message| message.headers.as_ref());
    assert!(
        headers.is_some_and(|map| map
            .headers
            .iter()
            .all(|header| header.key != "x-gateway-destination-endpoint")),
        "a preexisting route value must not be sent to EPP"
    );
    Ok(())
}

#[test]
fn llmd_ext_proc_keeps_worker_credentials_on_the_worker_request() -> Result<(), serde_yaml::Error> {
    let worker = start_header_echo_backend();
    let processor = start_mock_routing_processor_with_reply_header(
        &format!("127.0.0.1:{}", worker.port()),
        ("X-Test-Worker-Key", "picker-fixture"),
    );
    let mut config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", processor.port())]),
    );
    let filters = &mut config.filter_chains[0].filters;
    filters.insert(
        1,
        serde_yaml::from_str(
            "
filter: router
routes:
  - path_prefix: /v1/
    cluster: model-worker
",
        )?,
    );
    filters.insert(
        2,
        serde_yaml::from_str(
            "
filter: credential_injection
clusters:
  - name: model-worker
    header: x-test-worker-key
    value: worker-fixture
",
        )?,
    );
    filters.insert(
        3,
        serde_yaml::from_str(
            "
filter: headers
request_set:
  - name: x-test-policy-worker-key
    value: policy-worker-fixture
",
        )?,
    );
    let proxy = start_proxy(&config);
    let (status, worker_headers) = http_post(proxy.addr(), "/v1/chat/completions", r#"{"model":"llmd-demo"}"#);
    assert_eq!(status, 200, "EPP should route the request to the worker echo server");
    let credential_headers: Vec<_> = worker_headers
        .lines()
        .filter(|line| {
            line.split_once(':')
                .is_some_and(|(name, _)| name.eq_ignore_ascii_case("x-test-worker-key"))
        })
        .collect();
    assert!(
        matches!(credential_headers.as_slice(), [line] if line.eq_ignore_ascii_case("x-test-worker-key: worker-fixture")),
        "the selected worker should receive exactly the gateway's credential when EPP appends to the same header"
    );
    assert!(
        worker_headers
            .lines()
            .any(|line| line.eq_ignore_ascii_case("x-test-policy-worker-key: policy-worker-fixture")),
        "the worker should receive the credential queued through header policy"
    );
    assert!(
        !worker_headers
            .to_ascii_lowercase()
            .contains("x-gateway-destination-endpoint:"),
        "the destination header should be consumed before forwarding to the worker"
    );
    let messages = processor.take_request_headers();
    assert_eq!(messages.len(), 1, "the picker should receive one opening message");
    let picker_headers = messages.first().and_then(|message| message.headers.as_ref());
    assert!(
        picker_headers.is_some_and(|headers| headers
            .headers
            .iter()
            .all(|header| { !matches!(header.key.as_str(), "x-test-worker-key" | "x-test-policy-worker-key") })),
        "the picker must not receive the worker's credential header"
    );
    Ok(())
}

/// A policy sees the BBR header and can reject before EPP sees the request.
#[test]
fn llmd_ext_proc_deferred_request_respects_prior_policy_rejection() {
    let processor = start_mock_routing_processor("127.0.0.1:1");
    let mut config = super::load_example_config(
        "llmd-ext-proc-routing.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", processor.port())]),
    );
    let rejection = serde_yaml::from_str(
        "
filter: static_response
status: 403
conditions:
  - when:
      headers_present: [X-Gateway-Model-Name]
",
    )
    .expect("static response should parse");
    config.filter_chains[0].filters.insert(1, rejection);
    let proxy = start_proxy(&config);

    let (status, _) = http_post(proxy.addr(), "/v1/chat/completions", r#"{"model":"llmd-demo"}"#);
    assert_eq!(status, 403);
    assert_eq!(processor.stream_count(), 0, "the rejected request must never reach EPP");
}
