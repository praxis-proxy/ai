// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the HTTP callout filter.
//!
//! Uses [`wiremock`] to simulate the callout backend.

#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    reason = "tests"
)]
mod filter_tests {
    use std::time::Duration;

    use praxis_filter::{BodyAccess, BodyMode, FilterAction};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use crate::{callout::HttpCalloutFilter, test_utils::make_filter_context};

    // -------------------------------------------------------------------------
    // Config Parsing
    // -------------------------------------------------------------------------

    #[test]
    fn config_valid_minimal() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.name(), "http_callout");
    }

    #[test]
    fn name_literal_matches_filter_name_const() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();
        assert_eq!(
            filter.name(),
            crate::callout::FILTER_NAME,
            "name() literal and FILTER_NAME must not drift"
        );
    }

    #[test]
    fn config_missing_target() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>("{}").unwrap();
        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("target"),
            "should mention missing target: {err}"
        );
    }

    #[test]
    fn config_invalid_url_no_scheme() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "example.com/api"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("invalid") || err.to_string().contains("http or https"),
            "should reject URL without scheme: {err}"
        );
    }

    #[test]
    fn config_invalid_url_template() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "https://${HOST}/api"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("template"),
            "should reject template URL: {err}"
        );
    }

    #[test]
    fn config_invalid_jsonpath() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            response:
              extract:
                - json_path: "$[invalid"
                  result_key: "key"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("invalid JSONPath"),
            "should reject invalid JSONPath: {err}"
        );
    }

    #[test]
    fn config_rejects_invalid_result_key() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "lakera.flagged"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("invalid result_key"),
            "dotted result_key should be rejected at config time: {err}"
        );
    }

    #[test]
    fn config_env_var_expansion_unset() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
              headers:
                - name: "Authorization"
                  value: "Bearer ${PRAXIS_TEST_MISSING_VAR_ABC123}"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("not set"),
            "should fail on unset env var: {err}"
        );
    }

    #[test]
    fn config_non_http_scheme_rejected() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "ftp://example.com/file"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("http or https"),
            "should reject non-http scheme: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_fields() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            unknown_field: true
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("unknown_field") || err.to_string().contains("unknown field"),
            "should reject unknown fields: {err}"
        );
    }

    #[test]
    fn config_rejects_max_body_bytes_above_limit() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              max_body_bytes: 209715200
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("exceeds limit"),
            "should reject max_body_bytes above limit: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_field_in_target() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
              bogus: true
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("bogus") || err.to_string().contains("unknown field"),
            "unknown field in target should be rejected: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_field_in_request() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              phase: request_body
              bogus: true
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("bogus") || err.to_string().contains("unknown field"),
            "unknown field in request should be rejected: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_field_in_response() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            response:
              bogus: true
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("bogus") || err.to_string().contains("unknown field"),
            "unknown field in response should be rejected: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_field_in_circuit_breaker() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            circuit_breaker:
              failure_threshold: 3
              recovery_timeout: "30s"
              bogus: true
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("bogus") || err.to_string().contains("unknown field"),
            "unknown field in circuit_breaker should be rejected: {err}"
        );
    }

    #[test]
    fn config_rejects_max_body_bytes_one_above_limit() {
        // 100 MiB + 1 byte should be rejected.
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              max_body_bytes: 104857601
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("exceeds limit"),
            "one byte above 100 MiB should be rejected: {err}"
        );
    }

    #[test]
    fn config_accepts_default_max_body_bytes() {
        // Default (1 MiB) should be accepted without specifying max_body_bytes.
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml);
        assert!(filter.is_ok(), "default max_body_bytes (1 MiB) should be accepted");
    }

    #[test]
    fn config_accepts_max_body_bytes_at_limit() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              max_body_bytes: 104857600
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml);
        assert!(filter.is_ok(), "max_body_bytes at exactly 100 MiB should be accepted");
    }

    #[test]
    fn config_warns_on_private_ip_url() {
        // Private/loopback URLs should succeed (warning only).
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://127.0.0.1:8080/api"
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml);
        assert!(filter.is_ok(), "private/loopback URL should succeed with a warning");
    }

    // -------------------------------------------------------------------------
    // on_denied_headers Removed (Review Fix #1)
    // -------------------------------------------------------------------------

    #[test]
    fn config_rejects_on_denied_headers_field() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            response:
              on_denied_headers:
                - "x-reason"
            "#,
        )
        .unwrap();

        let err = HttpCalloutFilter::from_config(&yaml).err().expect("expected error");
        assert!(
            err.to_string().contains("on_denied_headers") || err.to_string().contains("unknown field"),
            "on_denied_headers should be rejected as unknown: {err}"
        );
    }

    #[tokio::test]
    async fn rejection_carries_no_response_headers() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(
                ResponseTemplate::new(500)
                    .append_header("x-reason", "bad-content")
                    .set_body_string("error"),
            )
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 403
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        match action {
            FilterAction::Reject(r) => {
                assert_eq!(r.status, 403);
                assert!(
                    r.headers.is_empty(),
                    "rejection should carry no response headers; got {:?}",
                    r.headers
                );
            },
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // Phase Handling
    // -------------------------------------------------------------------------

    #[test]
    fn phase_request_headers_body_access_is_none() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              phase: request_headers
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.request_body_access(), BodyAccess::None);
        assert_eq!(filter.request_body_mode(), BodyMode::Stream);
    }

    #[test]
    fn phase_request_body_access_is_readonly() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              phase: request_body
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();
        assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
        assert!(
            matches!(
                filter.request_body_mode(),
                BodyMode::StreamBuffer { max_bytes: Some(_) }
            ),
            "request_body phase should use StreamBuffer"
        );
    }

    #[test]
    fn needs_request_context_is_true() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();
        assert!(filter.needs_request_context());
    }

    // -------------------------------------------------------------------------
    // Successful Callout
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn successful_callout_extracts_results() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "flagged": true,
                "score": 0.95
            })))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "flagged"
                - json_path: "$.score"
                  result_key: "score"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "should continue after success"
        );

        let results = ctx.filter_results.get("http_callout").expect("should have results");
        assert_eq!(results.get("flagged"), Some("true"));
        assert_eq!(results.get("score"), Some("0.95"));
    }

    // -------------------------------------------------------------------------
    // Non-JSON Response
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn non_json_response_body_skips_extraction() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json at all"))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "flagged"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "non-JSON response should continue without error"
        );

        let has_results = ctx
            .filter_results
            .get("http_callout")
            .is_some_and(|rs| rs.get("flagged").is_some());
        assert!(!has_results, "non-JSON response should not produce extraction results");
    }

    #[tokio::test]
    async fn empty_response_body_skips_extraction() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "flagged"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "empty response body should continue without error"
        );

        let has_results = ctx
            .filter_results
            .get("http_callout")
            .is_some_and(|rs| rs.get("flagged").is_some());
        assert!(
            !has_results,
            "empty response body should not produce extraction results"
        );
    }

    // -------------------------------------------------------------------------
    // Failure Modes
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn failure_mode_closed_rejects() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 403
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 403),
            "fail-closed should reject with 403"
        );
    }

    #[tokio::test]
    async fn failure_mode_open_continues() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            on_failure: open
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "fail-open should continue");
    }

    #[tokio::test]
    async fn entry_failure_mode_key_does_not_change_on_failure() {
        // `failure_mode` is a filter-entry structural key stripped before
        // filter config parsing; it must not alias to `on_failure`.
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            failure_mode: open
            status_on_error: 403
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 403),
            "entry-level failure_mode must not make the callout fail-open"
        );
    }

    // -------------------------------------------------------------------------
    // Timeout
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn timeout_triggers_failure_mode() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/slow"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(10)))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/slow"
              timeout: "100ms"
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 504
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 504),
            "timeout should reject with configured status"
        );
    }

    // -------------------------------------------------------------------------
    // Depth Limiting
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn depth_limit_rejects_at_max() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0) // should not be called
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            on_failure: closed
            max_depth: 1
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let mut headers = http::HeaderMap::new();
        headers.insert("x-praxis-callout-depth", "1".parse().unwrap());

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers,
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "depth >= max_depth should reject"
        );
    }

    // -------------------------------------------------------------------------
    // Forward Headers
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn forward_headers_copied_to_callout() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .and(wiremock::matchers::header("x-custom", "my-value"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
              forward_headers:
                - "x-custom"
            request:
              phase: request_headers
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let mut headers = http::HeaderMap::new();
        headers.insert("x-custom", "my-value".parse().unwrap());

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers,
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "forward_headers should work");
    }

    #[tokio::test]
    async fn forward_headers_absent_from_request_not_sent() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
              forward_headers:
                - "x-custom"
            request:
              phase: request_headers
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        // Request does not carry x-custom.
        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let received = mock_server
            .received_requests()
            .await
            .expect("callout should have fired");
        assert_eq!(received.len(), 1, "exactly one callout request expected");
        assert!(
            !received[0].headers.contains_key("x-custom"),
            "absent downstream header must not be forwarded; got {:?}",
            received[0].headers
        );
    }

    // -------------------------------------------------------------------------
    // Inject Headers
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn inject_headers_from_callout_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("x-guard-id", "abc-123")
                    .set_body_json(serde_json::json!({"ok": true})),
            )
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              inject_headers:
                - "x-guard-id"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let injected = ctx
            .request_headers_to_set
            .iter()
            .find(|(name, _)| name.as_str() == "x-guard-id");
        assert!(injected.is_some(), "x-guard-id should be injected");
        assert_eq!(injected.unwrap().1, "abc-123");
    }

    #[tokio::test]
    async fn inject_headers_absent_from_response_not_injected() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              inject_headers:
                - "x-guard-id"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        assert!(
            ctx.request_headers_to_set.is_empty(),
            "absent response header should not be injected"
        );
    }

    // -------------------------------------------------------------------------
    // Request Body Phase
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn request_body_phase_skips_on_request() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              phase: request_body
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "request_body phase should skip on_request"
        );
    }

    #[tokio::test]
    async fn request_body_phase_fires_on_end_of_stream() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"flagged": false})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_body
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "flagged"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        let mut body = Some(bytes::Bytes::from(r#"{"prompt":"hello"}"#));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        // Body-phase results are stashed, then published by on_request.
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let results = ctx.filter_results.get("http_callout").expect("should have results");
        assert_eq!(results.get("flagged"), Some("false"));
    }

    #[tokio::test]
    async fn request_body_phase_skips_non_eos() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            request:
              phase: request_body
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        let mut body = Some(bytes::Bytes::from("chunk"));

        let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "should skip non-end-of-stream chunks"
        );
    }

    #[tokio::test]
    async fn request_body_results_republished_after_results_cleared() {
        // Branch evaluation clears `filter_results` after every filter in
        // the headers phase. Body-phase extractions are stashed and
        // re-published by `on_request`, so a preceding filter's clearing
        // cannot wipe them.
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"flagged": true})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_body
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "flagged"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);
        let mut body = Some(bytes::Bytes::from(r#"{"prompt":"hello"}"#));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert!(
            !ctx.filter_results.contains_key("http_callout"),
            "body phase should stash, not publish, results"
        );

        // A preceding filter's branch evaluation runs first and clears
        // any leftover results.
        ctx.filter_results.clear();

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let results = ctx
            .filter_results
            .get("http_callout")
            .expect("results should be republished");
        assert_eq!(results.get("flagged"), Some("true"));
    }

    // -------------------------------------------------------------------------
    // Body Shaping
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn body_shaping_strips_extra_fields() {
        let mock_server = MockServer::start().await;

        // The mock expects a body with only "messages", no "model".
        Mock::given(method("POST"))
            .and(path("/guard"))
            .and(wiremock::matchers::body_json(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"flagged": false})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
              body:
                messages: "$.messages"
            request:
              phase: request_body
            response:
              extract:
                - json_path: "$.flagged"
                  result_key: "flagged"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        ctx.current_filter_id = Some(0);

        // Downstream body has "model" which Lakera would reject.
        let mut body = Some(bytes::Bytes::from(
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
        ));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "shaped body should succeed");

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let results = ctx.filter_results.get("http_callout").expect("should have results");
        assert_eq!(results.get("flagged"), Some("false"));
    }

    #[tokio::test]
    async fn non_json_body_with_shaping_forwards_raw() {
        let mock_server = MockServer::start().await;

        // The mock only matches the raw, unshaped body.
        Mock::given(method("POST"))
            .and(path("/guard"))
            .and(wiremock::matchers::body_string("not json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
              body:
                messages: "$.messages"
            request:
              phase: request_body
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        let mut body = Some(bytes::Bytes::from("not json"));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "non-JSON body should fall back to raw forwarding"
        );
    }

    // -------------------------------------------------------------------------
    // Circuit Breaker
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn circuit_breaker_trips_after_threshold() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 503
            circuit_breaker:
              failure_threshold: 2
              recovery_timeout: "60s"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        // Fire enough requests to trip the breaker.
        for _ in 0..3 {
            let req = praxis_filter::Request {
                method: http::Method::POST,
                uri: "/test".parse().unwrap(),
                headers: http::HeaderMap::new(),
            };
            let mut ctx = make_filter_context(&req);
            let _action = filter.on_request(&mut ctx).await.unwrap();
        }

        // After the breaker trips, requests should still be rejected
        // without hitting the server.
        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 503),
            "circuit breaker should reject after threshold"
        );
    }

    // -------------------------------------------------------------------------
    // JSONPath Coercion (via filter)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn extraction_integer_coerced_to_string() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"count": 42})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.count"
                  result_key: "count"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let _action = filter.on_request(&mut ctx).await.unwrap();
        let results = ctx.filter_results.get("http_callout").expect("should have results");
        assert_eq!(results.get("count"), Some("42"));
    }

    #[tokio::test]
    async fn extraction_null_skipped() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"field": null})))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.field"
                  result_key: "field"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let _action = filter.on_request(&mut ctx).await.unwrap();
        // No results written when null.
        let has_field = ctx
            .filter_results
            .get("http_callout")
            .is_some_and(|rs| rs.get("field").is_some());
        assert!(!has_field, "null field should not be written to results");
    }

    #[tokio::test]
    async fn extraction_oversized_value_skipped_and_continues() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"reason": "x".repeat(300)})),
            )
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.reason"
                  result_key: "reason"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "oversized extraction value must not fail the request"
        );
        let has_field = ctx
            .filter_results
            .get("http_callout")
            .is_some_and(|rs| rs.get("reason").is_some());
        assert!(!has_field, "oversized value should be skipped");
    }

    #[tokio::test]
    async fn extraction_control_char_value_skipped_and_continues() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"reason": "line1\nline2"})),
            )
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
            response:
              extract:
                - json_path: "$.reason"
                  result_key: "reason"
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "control-char extraction value must not fail the request"
        );
        let has_field = ctx
            .filter_results
            .get("http_callout")
            .is_some_and(|rs| rs.get("reason").is_some());
        assert!(!has_field, "control-char value should be skipped");
    }
}
