// SPDX-License-Identifier: Apache-2.0
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

    use praxis_filter::{BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use crate::{callout::HttpCalloutFilter, test_utils::make_filter_context};

    /// Build a test filter, explicitly opting local mock endpoints into the
    /// private-address policy while leaving public-target fixtures unchanged.
    fn test_filter(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let mut config = config.clone();
        let needs_private_opt_in = config
            .get("target")
            .and_then(|target| target.get("url"))
            .and_then(serde_yaml::Value::as_str)
            .is_some_and(|url| {
                praxis_ai_apis::callout_target::validate_configured_http_target(
                    "http_callout test",
                    url,
                    praxis_ai_apis::callout_target::AddressPolicy::PublicOnly,
                )
                .is_err()
            });
        if needs_private_opt_in {
            config["target"]["allow_private_addresses"] = serde_yaml::Value::Bool(true);
        }
        HttpCalloutFilter::from_config(&config)
    }

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

        let filter = test_filter(&yaml).unwrap();
        assert_eq!(filter.name(), "http_callout");
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
            "should reject invalid result key at config time: {err}"
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
    fn config_rejects_status_on_error_out_of_range() {
        for bad in ["0", "99", "100", "200", "204", "302", "399", "600", "65535"] {
            let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
                r#"
                target:
                  url: "http://example.com/api"
                status_on_error: {bad}
                "#,
            ))
            .unwrap();

            let err = HttpCalloutFilter::from_config(&yaml)
                .err()
                .unwrap_or_else(|| panic!("status_on_error={bad} should be rejected"));
            assert!(
                err.to_string().contains("status_on_error"),
                "should mention status_on_error for {bad}: {err}"
            );
        }
    }

    #[test]
    fn config_accepts_valid_status_on_error() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            status_on_error: 503
            "#,
        )
        .unwrap();

        assert!(
            HttpCalloutFilter::from_config(&yaml).is_ok(),
            "a valid HTTP status should be accepted"
        );
    }

    #[test]
    fn config_rejects_userinfo_in_url() {
        // Embedded credentials could leak into logs or be forwarded to the
        // callout target; both `user:pass@host` and bare `user@host` must
        // be rejected at config time.
        for bad in ["http://user:pass@example.com/api", "http://user@example.com/api"] {
            let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
                r#"
                target:
                  url: "{bad}"
                "#,
            ))
            .unwrap();

            let err = HttpCalloutFilter::from_config(&yaml)
                .err()
                .unwrap_or_else(|| panic!("URL with userinfo should be rejected: {bad}"));
            assert!(
                err.to_string().contains("embedded credentials"),
                "error should mention embedded credentials for {bad}: {err}"
            );
        }
    }

    #[test]
    fn config_rejects_private_ip_url_without_opt_in() {
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://127.0.0.1:8080/api"
            "#,
        )
        .unwrap();

        let filter = HttpCalloutFilter::from_config(&yaml);
        assert!(filter.is_err(), "private/loopback URL should require explicit opt-in");
    }

    #[test]
    fn config_accepts_disallowed_forward_header_with_warning() {
        // A hop-by-hop/sensitive forward_header (e.g. connection) is a
        // config no-op, not an error: the filter builds successfully and
        // warns at config time. (Request-time skipping is covered by
        // disallowed_forward_header_not_sent.)
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
              forward_headers:
                - "connection"
                - "x-allowed"
            "#,
        )
        .unwrap();

        assert!(
            HttpCalloutFilter::from_config(&yaml).is_ok(),
            "a disallowed forward_header should warn, not fail config"
        );
    }

    // -------------------------------------------------------------------------
    // Target Parsing
    // -------------------------------------------------------------------------

    #[test]
    fn target_parse_https_preserves_target_components() {
        let target =
            praxis_ai_apis::callout_target::validate_http_target("http_callout", "https://example.com:8443/api")
                .unwrap();

        assert_eq!(target.scheme(), "https");
        assert_eq!(target.host_str(), Some("example.com"));
        assert_eq!(target.port(), Some(8443));
        assert_eq!(target.path(), "/api");
    }

    #[test]
    fn target_parse_rejects_userinfo() {
        assert!(
            praxis_ai_apis::callout_target::validate_http_target("http_callout", "https://user:pass@example.com/api",)
                .is_err()
        );
    }

    #[test]
    fn target_parse_rejects_non_http_scheme() {
        assert!(praxis_ai_apis::callout_target::validate_http_target("http_callout", "file:///tmp/secret").is_err());
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

        let filter = test_filter(&yaml).unwrap();
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

        let filter = test_filter(&yaml).unwrap();
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
    fn name_matches_filter_name() {
        // `name()` must return the "http_callout" string literal so the
        // filter-docs generator can discover it; guard against drift from
        // the internal FILTER_NAME const.
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(
            r#"
            target:
              url: "http://example.com/api"
            "#,
        )
        .unwrap();

        let filter = test_filter(&yaml).unwrap();
        assert_eq!(filter.name(), crate::callout::FILTER_NAME);
        assert_eq!(filter.name(), "http_callout");
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

        let filter = test_filter(&yaml).unwrap();
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

        let filter = test_filter(&yaml).unwrap();

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

    #[tokio::test]
    async fn non_2xx_callout_response_forwards_status_to_downstream() {
        // A *completed* callout that answers with a non-2xx status is
        // distinct from a transport failure: the filter forwards the
        // callout's own status to the downstream client, it does not apply
        // `status_on_error` (that path is only for DNS/connect/I/O errors).
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
            status_on_error: 403
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 500),
            "a completed non-2xx callout should forward its own status (500), \
             not status_on_error and not fail-open Continue"
        );
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

        let filter = test_filter(&yaml).unwrap();

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

    // -------------------------------------------------------------------------
    // Failure Modes
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn on_failure_closed_rejects() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "http://{addr}/guard"
              timeout: "500ms"
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 403
            "#,
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 403),
            "fail-closed should reject with status_on_error on transport failure"
        );
    }

    #[tokio::test]
    async fn on_failure_open_continues() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "http://{addr}/guard"
              timeout: "500ms"
            request:
              phase: request_headers
            on_failure: open
            "#,
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "fail-open should continue on transport failure"
        );
    }

    #[tokio::test]
    async fn oversized_response_body_treated_as_failure() {
        // The client's response-byte ceiling is set to the configured
        // max_body_bytes, so a response larger than that must fail the
        // callout (not silently truncate). With on_failure: closed that
        // surfaces as Reject(status_on_error).
        let mock_server = MockServer::start().await;

        // Configured cap is 1024 bytes; return well over that.
        let big_body = "x".repeat(4096);
        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_string(big_body))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
            request:
              phase: request_headers
              max_body_bytes: 1024
            on_failure: closed
            status_on_error: 502
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 502),
            "a response body over max_body_bytes should fail the callout"
        );
    }

    // -------------------------------------------------------------------------
    // SSRF / DNS-rebinding (allow_private_addresses)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn private_target_requires_explicit_opt_in() {
        // A live server on loopback that would answer 200 if reached. With
        // allow_private_addresses: false, resolve_peer must reject the
        // loopback peer *before* any request, so the callout fails and
        // on_failure: closed surfaces as Reject(status_on_error).
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
              allow_private_addresses: false
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 502
            "#,
            mock_server.uri()
        ))
        .unwrap();

        assert!(
            HttpCalloutFilter::from_config(&yaml).is_err(),
            "a literal private target must fail configuration without the opt-in"
        );
    }

    #[tokio::test]
    async fn private_target_allowed_with_explicit_opt_in() {
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
              allow_private_addresses: true
            request:
              phase: request_headers
            on_failure: closed
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "a loopback target must be reachable after explicit opt-in"
        );
    }

    // -------------------------------------------------------------------------
    // Timeout
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn timeout_triggers_on_failure() {
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

        let filter = test_filter(&yaml).unwrap();

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

        let filter = test_filter(&yaml).unwrap();

        let mut headers = http::HeaderMap::new();
        headers.insert("x-praxis-iterative-depth", "1".parse().unwrap());

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

        let filter = test_filter(&yaml).unwrap();

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
    async fn static_host_is_overwritten_with_target_authority_on_wire() {
        let mock_server = MockServer::start().await;
        let authority = mock_server.address().to_string();

        Mock::given(method("POST"))
            .and(path("/guard"))
            .and(wiremock::matchers::header("host", authority))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
              headers:
                - name: "Host"
                  value: "attacker.example"
            request:
              phase: request_headers
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();
        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
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

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        // Injected with set (overwrite) semantics via request_headers_to_set.
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

        // Response omits the configured inject header.
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

        let filter = test_filter(&yaml).unwrap();

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
        assert!(injected.is_none(), "absent response header should not be injected");
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

        let filter = test_filter(&yaml).unwrap();

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

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);
        let mut body = Some(bytes::Bytes::from(r#"{"prompt":"hello"}"#));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
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

        let filter = test_filter(&yaml).unwrap();

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

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        // Downstream body has "model" which Lakera would reject.
        let mut body = Some(bytes::Bytes::from(
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
        ));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "shaped body should succeed");

        let results = ctx.filter_results.get("http_callout").expect("should have results");
        assert_eq!(results.get("flagged"), Some("false"));
    }

    // -------------------------------------------------------------------------
    // Circuit Breaker
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn circuit_breaker_trips_after_threshold() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "http://{addr}/guard"
              timeout: "500ms"
            request:
              phase: request_headers
            on_failure: closed
            status_on_error: 503
            circuit_breaker:
              failure_threshold: 2
              recovery_timeout: "60s"
            "#,
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        // Fire enough requests to trip the breaker via connect failures.
        for _ in 0..3 {
            let req = praxis_filter::Request {
                method: http::Method::POST,
                uri: "/test".parse().unwrap(),
                headers: http::HeaderMap::new(),
            };
            let mut ctx = make_filter_context(&req);
            let _action = filter.on_request(&mut ctx).await.unwrap();
        }

        // After the breaker trips, requests should be rejected
        // without attempting a connection.
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

        let filter = test_filter(&yaml).unwrap();

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

        let filter = test_filter(&yaml).unwrap();

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

    // -------------------------------------------------------------------------
    // Forward Headers — absence
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn forward_header_absent_from_request_not_sent() {
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

        let filter = test_filter(&yaml).unwrap();

        // Downstream request omits the configured forward header.
        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        // A forward header that is absent downstream must not appear on the callout.
        let requests = mock_server.received_requests().await.expect("recorded requests");
        let callout = requests.first().expect("callout should have fired");
        assert!(
            callout.headers.get("x-custom").is_none(),
            "absent forward header should not be sent to the callout"
        );
    }

    #[tokio::test]
    async fn disallowed_forward_header_not_sent() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&mock_server)
            .await;

        // `proxy-authorization` is hop-by-hop/sensitive (disallowed); `x-ok`
        // is an ordinary header that should forward normally.
        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&format!(
            r#"
            target:
              url: "{}/guard"
              forward_headers:
                - "proxy-authorization"
                - "x-ok"
            request:
              phase: request_headers
            "#,
            mock_server.uri()
        ))
        .unwrap();

        let filter = test_filter(&yaml).unwrap();

        // The client supplies BOTH headers downstream.
        let mut headers = http::HeaderMap::new();
        headers.insert("proxy-authorization", "Bearer secret".parse().unwrap());
        headers.insert("x-ok", "fine".parse().unwrap());

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers,
        };
        let mut ctx = make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));

        let requests = mock_server.received_requests().await.expect("recorded requests");
        let callout = requests.first().expect("callout should have fired");
        assert!(
            callout.headers.get("proxy-authorization").is_none(),
            "a disallowed forward header must not reach the callout even when the client sends it"
        );
        assert_eq!(
            callout.headers.get("x-ok").map(http::HeaderValue::as_bytes),
            Some(&b"fine"[..]),
            "an allowed forward header should still be forwarded"
        );
    }

    // -------------------------------------------------------------------------
    // Body Shaping — non-JSON fallback
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn body_shaping_non_json_forwards_raw() {
        let mock_server = MockServer::start().await;

        // Shaping is configured, but the downstream body is not JSON, so the
        // raw body must be forwarded verbatim rather than dropped.
        Mock::given(method("POST"))
            .and(path("/guard"))
            .and(wiremock::matchers::body_string("this is not json"))
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

        let filter = test_filter(&yaml).unwrap();

        let req = praxis_filter::Request {
            method: http::Method::POST,
            uri: "/test".parse().unwrap(),
            headers: http::HeaderMap::new(),
        };
        let mut ctx = make_filter_context(&req);

        let mut body = Some(bytes::Bytes::from("this is not json"));

        let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "non-JSON body with shaping should forward raw and succeed"
        );

        let results = ctx.filter_results.get("http_callout").expect("should have results");
        assert_eq!(results.get("flagged"), Some("false"));
    }

    // -------------------------------------------------------------------------
    // Nested Config — deny_unknown_fields
    // -------------------------------------------------------------------------

    #[test]
    fn config_rejects_unknown_target_field() {
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
            "should reject unknown target field: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_response_field() {
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
            "should reject unknown response field: {err}"
        );
    }

    #[test]
    fn config_rejects_unknown_circuit_breaker_field() {
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
            "should reject unknown circuit_breaker field: {err}"
        );
    }
}
