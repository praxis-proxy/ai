// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Provider request and response translation helpers.

pub(crate) mod chat_completions;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::cognitive_complexity,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use serde_json::{Value, json};

    fn map(request: &Value) -> Value {
        super::chat_completions::responses_request_to_chat_request(request).unwrap()
    }

    fn map_state(request: &Value, messages: &[Value], tools: &[Value], tool_choice: &Value) -> Value {
        super::chat_completions::responses_state_to_chat_request(request, messages, tools, tool_choice).unwrap()
    }

    fn map_error(request: &Value) -> String {
        super::chat_completions::responses_request_to_chat_request(request)
            .unwrap_err()
            .to_string()
    }

    fn chat_completion_response_fixture() -> Value {
        serde_json::from_str(include_str!("fixtures/chat_completion_response.json")).unwrap()
    }

    #[test]
    fn non_object_responses_request_returns_expected_object_error() {
        let error = super::chat_completions::responses_request_to_chat_request(&json!("hello")).unwrap_err();
        assert_eq!(error.to_string(), "Responses request must be a JSON object");
    }

    #[test]
    fn state_overrides_supply_enriched_messages_tools_and_tool_choice() {
        let request = json!({
            "model": "gpt-4.1-mini",
            "input": "client input",
            "tools": [],
            "tool_choice": "none"
        });
        let messages = vec![json!({"role": "user", "content": "rehydrated input"})];
        let tools = vec![json!({
            "type": "function",
            "name": "lookup",
            "parameters": {"type": "object"}
        })];

        let mapped = map_state(&request, &messages, &tools, &json!("required"));

        assert_eq!(mapped["messages"][0]["content"], "rehydrated input");
        assert_eq!(mapped["tools"][0]["function"]["name"], "lookup");
        assert_eq!(mapped["tool_choice"], "required");
    }

    #[test]
    fn state_default_tool_choice_is_omitted_without_tools() {
        let request = json!({"model": "gpt-4.1-mini", "input": "hello"});
        let messages = vec![json!({"role": "user", "content": "hello"})];

        let mapped = map_state(&request, &messages, &[], &json!("auto"));

        assert!(mapped.get("tools").is_none());
        assert!(
            mapped.get("tool_choice").is_none(),
            "Chat Completions backends may reject tool_choice when no tools are present"
        );
    }

    #[test]
    fn responses_request_maps_to_chat_completions_wire_shape() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "instructions": "Keep replies short.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Remember the code word: ember."}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "store_memory",
                    "description": "Store a memory.",
                    "strict": true,
                    "parameters": {"type": "object", "properties": {"memory": {"type": "string"}}, "required": ["memory"]}
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 64
        }));

        assert_eq!(mapped["model"], "gpt-4o-mini");
        assert_eq!(mapped["temperature"], 0.2);
        assert_eq!(mapped["top_p"], 0.9);
        assert_eq!(mapped["max_completion_tokens"], 64);
        assert_eq!(mapped["tool_choice"], "auto");
        assert_eq!(
            mapped["messages"][0],
            json!({"role": "system", "content": "Keep replies short."})
        );
        assert_eq!(
            mapped["messages"][1],
            json!({"role": "user", "content": "Remember the code word: ember."})
        );
        assert_eq!(
            mapped["tools"][0],
            json!({
                "type": "function",
                "function": {
                    "name": "store_memory",
                    "description": "Store a memory.",
                    "strict": true,
                    "parameters": {"type": "object", "properties": {"memory": {"type": "string"}}, "required": ["memory"]}
                }
            })
        );
    }

    #[test]
    fn streaming_request_preserves_options_and_forces_usage() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "stream": true,
            "stream_options": {
                "include_usage": false,
                "include_obfuscation": false
            }
        }));

        assert_eq!(mapped["stream"], true);
        assert_eq!(mapped["stream_options"]["include_usage"], true);
        assert_eq!(mapped["stream_options"]["include_obfuscation"], false);
    }

    #[test]
    fn non_streaming_request_does_not_invent_stream_options() {
        let mapped = map(&json!({"model": "m", "input": "hello", "stream": false}));

        assert_eq!(mapped["stream"], false);
        assert!(mapped.get("stream_options").is_none());
    }

    #[test]
    fn simple_inputs_map_or_drop_cleanly() {
        let string_input = map(&json!({"model": "gpt-4o-mini", "instructions": "", "input": "Hello"}));
        let object_input = map(&json!({"model": "gpt-4o-mini", "input": {"role": "developer", "content": "terse"}}));
        let no_input = map(&json!({"model": "gpt-4o-mini"}));
        let unsupported_input = map(&json!({"model": "gpt-4o-mini", "input": 42}));

        assert_eq!(string_input["messages"], json!([{"role": "user", "content": "Hello"}]));
        assert_eq!(
            object_input["messages"],
            json!([{"role": "developer", "content": "terse"}])
        );
        assert_eq!(no_input["messages"], Value::Array(Vec::new()));
        assert_eq!(unsupported_input["messages"], Value::Array(Vec::new()));
    }

    #[test]
    fn function_tool_choice_maps_without_widening() {
        let function_choice = map(&json!({
            "model": "gpt-4o-mini", "input": "hello",
            "tool_choice": {"type": "function", "name": "lookup_weather"}
        }));

        assert_eq!(
            function_choice["tool_choice"],
            json!({"type": "function", "function": {"name": "lookup_weather"}})
        );
    }

    #[test]
    fn unsupported_hosted_responses_tools_are_rejected() {
        let only_unsupported = map_error(&json!({
            "model": "gpt-4o-mini",
            "input": "hello",
            "tools": [{"type": "code_interpreter"}]
        }));
        let mixed = map_error(&json!({
            "model": "gpt-4o-mini",
            "input": "hello",
            "tools": [
                {"type": "image_generation"},
                {"type": "function", "name": "lookup_weather", "parameters": {"type": "object"}}
            ]
        }));

        assert!(only_unsupported.contains("code_interpreter"));
        assert!(mixed.contains("image_generation"));
    }

    #[test]
    fn file_search_tool_maps_to_bounded_private_function() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "find the quarterly results",
            "tools": [{
                "type": "file_search",
                "vector_store_ids": ["vs_q4"],
                "max_num_results": 8,
                "filters": {"type": "eq", "key": "year", "value": 2026},
                "ranking_options": {"ranker": "auto", "score_threshold": 0.2}
            }]
        }));

        assert_eq!(mapped["tools"].as_array().map(Vec::len), Some(1));
        let function = &mapped["tools"][0]["function"];
        assert_eq!(function["name"], "file_search");
        assert_eq!(function["strict"], true);
        assert_eq!(function["parameters"]["type"], "object");
        assert_eq!(function["parameters"]["required"], json!(["query"]));
        assert_eq!(function["parameters"]["properties"]["query"]["type"], "string");
        assert_eq!(function["parameters"]["properties"]["query"]["minLength"], 1);
        assert_eq!(function["parameters"]["properties"]["query"]["maxLength"], 65_536);
        assert_eq!(function["parameters"]["additionalProperties"], false);
        assert!(
            mapped["tools"][0].get("vector_store_ids").is_none(),
            "hosted-tool configuration must not leak into the Chat function"
        );
    }

    #[test]
    fn file_search_tool_choice_maps_to_forced_function() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "find the report",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_reports"]}],
            "tool_choice": {"type": "file_search"}
        }));

        assert_eq!(
            mapped["tool_choice"],
            json!({"type": "function", "function": {"name": "file_search"}})
        );
    }

    #[test]
    fn client_function_choice_cannot_target_synthesized_file_search() {
        let error = map_error(&json!({
            "model": "gpt-4o-mini",
            "input": "find the report",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_reports"]}],
            "tool_choice": {"type": "function", "name": "file_search"}
        }));

        assert_eq!(
            error,
            "invalid Responses file_search tool for Chat Completions translation: tool_choice for hosted file_search must use type file_search"
        );
    }

    #[test]
    fn file_search_function_name_collisions_are_rejected() {
        for function in [
            json!({"type": "function", "name": "file_search", "parameters": {"type": "object"}}),
            json!({
                "type": "function",
                "function": {"name": "file_search", "parameters": {"type": "object"}}
            }),
        ] {
            let error = map_error(&json!({
                "model": "gpt-4o-mini",
                "input": "find the report",
                "tools": [
                    {"type": "file_search", "vector_store_ids": ["vs_reports"]},
                    function
                ]
            }));

            assert_eq!(
                error,
                "Responses function tool name `file_search` conflicts with the synthesized file_search function"
            );
        }
    }

    #[test]
    fn client_file_search_function_without_hosted_tool_is_preserved() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "call my function",
            "tools": [{
                "type": "function",
                "name": "file_search",
                "parameters": {"type": "object"}
            }]
        }));

        assert_eq!(mapped["tools"][0]["function"]["name"], "file_search");
    }

    #[test]
    fn malformed_file_search_definitions_are_rejected() {
        let malformed = [
            json!({"type": "file_search"}),
            json!({"type": "file_search", "vector_store_ids": []}),
            json!({"type": "file_search", "vector_store_ids": [""]}),
            json!({"type": "file_search", "vector_store_ids": [42]}),
            json!({"type": "file_search", "vector_store_ids": ["vs"], "max_num_results": 0}),
            json!({"type": "file_search", "vector_store_ids": ["vs"], "max_num_results": 51}),
            json!({"type": "file_search", "vector_store_ids": ["vs"], "filters": []}),
            json!({"type": "file_search", "vector_store_ids": ["vs"], "ranking_options": "auto"}),
        ];

        for tool in malformed {
            let error = map_error(&json!({"model": "m", "input": "hello", "tools": [tool]}));
            assert!(
                error.starts_with("invalid Responses file_search tool for Chat Completions translation:"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn excessive_vector_store_ids_are_rejected() {
        // One past the maximum (10) amplifies fan-out and is rejected...
        let too_many: Vec<String> = (0..11).map(|i| format!("vs_{i}")).collect();
        let error = map_error(&json!({
            "model": "m",
            "input": "hello",
            "tools": [{"type": "file_search", "vector_store_ids": too_many}]
        }));
        assert_eq!(
            error,
            "invalid Responses file_search tool for Chat Completions translation: \
             vector_store_ids must contain at most 10 entries"
        );

        // ...but the maximum count itself is accepted.
        let at_limit: Vec<String> = (0..10).map(|i| format!("vs_{i}")).collect();
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "tools": [{"type": "file_search", "vector_store_ids": at_limit}]
        }));
        assert_eq!(mapped["tools"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn duplicate_file_search_definitions_are_rejected() {
        let error = map_error(&json!({
            "model": "m",
            "input": "hello",
            "tools": [
                {"type": "file_search", "vector_store_ids": ["vs_a"]},
                {"type": "file_search", "vector_store_ids": ["vs_b"]}
            ]
        }));

        assert_eq!(
            error,
            "invalid Responses file_search tool for Chat Completions translation: only one file_search tool may be declared"
        );
    }

    #[test]
    fn file_search_choice_without_definition_is_rejected() {
        let error = map_error(&json!({
            "model": "m",
            "input": "hello",
            "tool_choice": {"type": "file_search"}
        }));

        assert_eq!(
            error,
            "invalid Responses file_search tool for Chat Completions translation: tool_choice requires a declared file_search tool"
        );
    }

    #[test]
    fn web_search_tools_map_to_one_bounded_private_function() {
        for tool_type in [
            "web_search",
            "web_search_preview",
            "web_search_preview_2025_03_11",
            "web_search_2025_08_26",
        ] {
            let mapped = map(&json!({
                "model": "gpt-4o-mini",
                "input": "latest news",
                "tools": [{
                    "type": tool_type,
                    "search_context_size": "high",
                    "user_location": {
                        "type": "approximate",
                        "country": "FR",
                        "city": "Paris",
                        "timezone": "Europe/Paris"
                    }
                }]
            }));

            assert_eq!(mapped["tools"][0]["type"], "function");
            assert_eq!(mapped["tools"][0]["function"]["name"], "web_search");
            assert_eq!(
                mapped["tools"][0]["function"]["parameters"]["required"],
                json!(["query"])
            );
            assert_eq!(
                mapped["tools"][0]["function"]["parameters"]["properties"]["query"]["minLength"],
                1
            );
            assert_eq!(
                mapped["tools"][0]["function"]["parameters"]["properties"]["query"]["maxLength"],
                4096
            );
            assert_eq!(
                mapped["tools"][0]["function"]["parameters"]["additionalProperties"],
                false
            );
            assert!(mapped["tools"][0].get("search_context_size").is_none());
            assert!(mapped["tools"][0].get("user_location").is_none());
        }
    }

    #[test]
    fn web_search_null_user_location_and_null_members_are_treated_as_unset() {
        // `WebSearchApproximateLocation` is object-or-null: a null location is a
        // valid "unset" and must translate cleanly, not fail with a 400.
        let null_location = map(&json!({
            "model": "gpt-4o-mini",
            "input": "latest news",
            "tools": [{"type": "web_search", "user_location": null}]
        }));

        assert_eq!(null_location["tools"][0]["type"], "function");
        assert_eq!(null_location["tools"][0]["function"]["name"], "web_search");
        assert!(null_location["tools"][0].get("user_location").is_none());

        // Each optional member is string-or-null: null members are valid "unset".
        let null_members = map(&json!({
            "model": "gpt-4o-mini",
            "input": "latest news",
            "tools": [{
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": null,
                    "region": null,
                    "city": null,
                    "timezone": null
                }
            }]
        }));

        assert_eq!(null_members["tools"][0]["function"]["name"], "web_search");
    }

    #[test]
    fn web_search_non_null_user_location_members_are_still_validated() {
        let empty_country = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search", "user_location": {"type": "approximate", "country": ""}}]
        }));
        let non_string_city = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search", "user_location": {"type": "approximate", "city": 42}}]
        }));

        assert!(empty_country.contains("user_location.country must be a non-empty string"));
        assert!(non_string_city.contains("user_location.city must be a non-empty string"));
    }

    #[test]
    fn forced_web_search_choice_maps_without_widening() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "latest news",
            "tools": [{"type": "web_search_preview"}],
            "tool_choice": {"type": "web_search_preview"}
        }));

        assert_eq!(
            mapped["tool_choice"],
            json!({"type": "function", "function": {"name": "web_search"}})
        );
    }

    #[test]
    fn web_search_function_name_collision_is_rejected() {
        let error = map_error(&json!({
            "model": "gpt-4o-mini",
            "input": "latest news",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
            ]
        }));

        assert_eq!(
            error,
            "Responses function tool name `web_search` conflicts with the synthesized web_search function"
        );
    }

    #[test]
    fn malformed_or_unsupported_web_search_config_is_rejected() {
        let duplicate = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search"}, {"type": "web_search_preview"}]
        }));
        let unsupported_field = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}]
        }));
        let unsupported_variant = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search_future"}]
        }));
        let invalid_context = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search", "search_context_size": "huge"}]
        }));
        let invalid_location = map_error(&json!({
            "model": "m",
            "tools": [{"type": "web_search", "user_location": {"type": "precise"}}]
        }));

        assert!(duplicate.contains("only one web-search tool"));
        assert!(unsupported_field.contains("field `filters` is not supported"));
        assert!(unsupported_variant.contains("unsupported Responses tool type"));
        assert!(invalid_context.contains("search_context_size must be one of"));
        assert!(invalid_location.contains("user_location.type must be approximate"));
    }

    #[test]
    fn forced_web_search_choice_requires_a_hosted_declaration() {
        let error = map_error(&json!({
            "model": "m",
            "input": "latest news",
            "tool_choice": {"type": "web_search"}
        }));

        assert!(error.contains("tool_choice requires a declared web-search tool"));
    }

    #[test]
    fn allowed_tools_choice_is_rejected_until_policy_filter_lands() {
        let error = map_error(&json!({
            "model": "m",
            "input": "hello",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "lookup"}]
            }
        }));

        assert_eq!(
            error,
            "unsupported Responses tool_choice type for Chat Completions translation: allowed_tools"
        );
    }

    #[test]
    fn multimodal_content_parts_use_chat_shapes() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {"type": "input_image", "image_url": "https://example.com/cat.png", "detail": "high"},
                        {"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64,bm90ZXM="},
                        {"type": "input_file", "filename": "remote.pdf", "file_url": "https://example.com/report.pdf"}
                    ]
                }
            ]
        }));

        assert_eq!(
            mapped["messages"][0],
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
                    {"type": "file", "file": {"filename": "notes.txt", "file_data": "data:text/plain;base64,bm90ZXM="}},
                    {"type": "file", "file": {"filename": "remote.pdf", "file_url": "https://example.com/report.pdf"}}
                ]
            })
        );
    }

    #[test]
    fn unsupported_content_parts_are_rejected() {
        let error = super::chat_completions::responses_request_to_chat_request(&json!({
            "model": "gpt-4o-mini",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe the attached image."},
                    {"type": "reasoning", "summary": []}
                ]
            }]
        }))
        .unwrap_err();
        assert!(error.to_string().contains("reasoning"));
    }

    #[test]
    fn file_id_input_images_report_specific_unsupported_reason() {
        let error = map_error(&json!({
            "model": "gpt-4o-mini",
            "input": [{
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file-abc"}]
            }]
        }));

        assert!(error.contains("input_image requires image_url; file_id references are not supported"));
    }

    #[test]
    fn empty_input_files_report_specific_unsupported_reason() {
        let error = map_error(&json!({
            "model": "gpt-4o-mini",
            "input": [{
                "role": "user",
                "content": [{"type": "input_file"}]
            }]
        }));

        assert!(error.contains("input_file requires file_id, filename, file_data, or file_url"));
    }

    #[test]
    fn unsupported_typed_input_items_are_rejected() {
        let error = super::chat_completions::responses_request_to_chat_request(&json!({
            "model": "gpt-4o-mini",
            "input": [
                {"type": "item_reference", "id": "msg_123"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .unwrap_err();
        assert!(error.to_string().contains("item_reference"));
    }

    #[test]
    fn compaction_input_item_translates_to_assistant_message() {
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode("Summary of prior conversation.");
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": [
                {"type": "compaction", "id": "compact_1", "encrypted_content": encoded},
                {"role": "user", "content": "What next?"}
            ]
        }));
        let messages = mapped["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("Summary of prior conversation.")
        );
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn empty_compaction_summary_is_dropped() {
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode("");
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": [
                {"type": "compaction", "id": "compact_2", "encrypted_content": encoded},
                {"role": "user", "content": "Hello"}
            ]
        }));
        let messages = mapped["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1, "empty compaction should be dropped");
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn tool_history_items_map_to_chat_messages() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_weather",
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"NYC\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": "{\"temperature\":72}"
                },
                {"role": "user", "content": "continue"}
            ]
        }));

        assert_eq!(
            mapped["messages"],
            json!([
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": "{\"city\":\"NYC\"}"
                            }
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "call_weather", "content": "{\"temperature\":72}"},
                {"role": "user", "content": "continue"}
            ])
        );
    }

    #[test]
    fn single_function_call_input_maps_through_batched_path() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "lookup_weather",
                "arguments": {"city": "NYC"}
            }
        }));

        assert_eq!(
            mapped["messages"],
            json!([{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": "{\"city\":\"NYC\"}"
                    }
                }]
            }])
        );
    }

    #[test]
    fn adjacent_function_call_items_share_one_assistant_message() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_weather",
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"NYC\"}"
                },
                {
                    "type": "function_call",
                    "call_id": "call_timezone",
                    "name": "lookup_timezone",
                    "arguments": "{\"city\":\"NYC\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": "{\"temperature\":72}"
                }
            ]
        }));

        assert_eq!(
            mapped["messages"],
            json!([
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": "{\"city\":\"NYC\"}"
                            }
                        },
                        {
                            "id": "call_timezone",
                            "type": "function",
                            "function": {
                                "name": "lookup_timezone",
                                "arguments": "{\"city\":\"NYC\"}"
                            }
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "call_weather", "content": "{\"temperature\":72}"}
            ])
        );
    }

    #[test]
    fn responses_request_forwards_chat_generation_controls() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "hello",
            "temperature": 0.4,
            "top_p": 0.8,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.2,
            "parallel_tool_calls": false,
            "service_tier": "flex",
            "top_logprobs": 5,
            "reasoning": {"effort": "medium"},
            "extra_body": {"chat_template_kwargs": {"thinking": true}}
        }));

        assert_eq!(mapped["presence_penalty"], 0.3);
        assert_eq!(mapped["frequency_penalty"], 0.2);
        assert_eq!(mapped["parallel_tool_calls"], false);
        assert_eq!(mapped["service_tier"], "flex");
        assert_eq!(mapped["top_logprobs"], 5);
        assert_eq!(mapped["logprobs"], true);
        assert_eq!(mapped["reasoning_effort"], "medium");
        assert_eq!(mapped["extra_body"]["chat_template_kwargs"]["thinking"], true);
        assert!(mapped.get("reasoning").is_none());
    }

    #[test]
    fn responses_text_format_maps_to_chat_response_format() {
        let json_object = map(&json!({
            "model": "gpt-4o-mini",
            "input": "return json",
            "text": {"format": {"type": "json_object"}}
        }));
        let json_schema = map(&json!({
            "model": "gpt-4o-mini",
            "input": "return json",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather",
                    "description": "Weather payload",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"temperature": {"type": "number"}}}
                }
            }
        }));

        assert_eq!(json_object["response_format"], json!({"type": "json_object"}));
        assert_eq!(
            json_schema["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "description": "Weather payload",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"temperature": {"type": "number"}}}
                }
            })
        );
    }

    #[test]
    fn responses_text_format_is_omitted_when_tools_are_translated() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "look up the weather",
            "text": {"format": {"type": "json_object"}},
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"}
            }]
        }));

        assert!(mapped.get("response_format").is_none());
        assert_eq!(mapped["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn responses_request_maps_semantic_chat_parameters() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "hello",
            "max_output_tokens": 128,
            "prompt_cache_key": "cache-123",
            "text": {"format": {"type": "json_object"}}
        }));

        assert_eq!(mapped["max_completion_tokens"], 128);
        assert!(mapped.get("max_tokens").is_none());
        assert_eq!(mapped["prompt_cache_key"], "cache-123");
        assert_eq!(mapped["response_format"], json!({"type": "json_object"}));
    }

    #[test]
    fn responses_text_format_maps_json_schema() {
        let mapped = map(&json!({
            "model": "gpt-4o-mini",
            "input": "return json",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "weather",
                    "description": "Weather payload",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"temperature": {"type": "number"}}}
                }
            }
        }));

        assert_eq!(
            mapped["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "description": "Weather payload",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"temperature": {"type": "number"}}}
                }
            })
        );
    }

    #[test]
    fn recorded_chat_response_maps_to_schema_complete_response_resource() {
        let fixture = chat_completion_response_fixture();
        let response = &fixture["response"];
        let request = json!({
            "model": "gpt-4o-mini",
            "instructions": "Reply tersely.",
            "input": "Remember the code word: ember.",
            "metadata": {"provider": "recording"},
            "text": {"format": {"type": "text"}},
            "temperature": 1.0,
            "top_p": 1.0
        });
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_123".to_owned(), 0);

        let mapped = super::chat_completions::chat_response_to_response_resource(response, &context).unwrap();

        assert_eq!(mapped["id"], "resp_123");
        assert_eq!(mapped["status"], "completed");
        assert_eq!(mapped["completed_at"], 0);
        assert_eq!(mapped["max_tool_calls"], Value::Null);
        assert_eq!(mapped["safety_identifier"], Value::Null);
        assert_eq!(mapped["prompt_cache_key"], Value::Null);
        assert_eq!(mapped["output"][0]["type"], "message");
        assert_eq!(mapped["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(
            mapped["output"][0]["content"][0]["text"],
            response["choices"][0]["message"]["content"]
        );
        assert_eq!(mapped["usage"]["input_tokens"], 126);
        assert_eq!(mapped["usage"]["output_tokens"], 194);
        assert_eq!(mapped["usage"]["total_tokens"], 320);
    }

    #[test]
    fn response_resource_preserves_required_request_fields() {
        let response = json!({
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}]
        });
        let request = json!({
            "model": "gpt-4o-mini",
            "input": "hello",
            "max_tool_calls": 3,
            "safety_identifier": "user-123",
            "prompt_cache_key": "cache-123",
            "text": {"format": {"type": "json_schema", "name": "weather", "schema": {"type": "object"}}}
        });
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_123".to_owned(), 7)
                .with_completed_at(11);

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["completed_at"], 11);
        assert_eq!(mapped["max_tool_calls"], 3);
        assert_eq!(mapped["safety_identifier"], "user-123");
        assert_eq!(mapped["prompt_cache_key"], "cache-123");
        assert_eq!(mapped["text"], request["text"]);
    }

    #[test]
    fn chat_tool_calls_and_content_filter_map_to_responses_items() {
        let tool_response = json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                    }]
                }
            }]
        });
        let filter_response = json!({
            "id": "chatcmpl-filtered",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{"finish_reason": "content_filter", "index": 0, "message": {"role": "assistant", "content": null}}]
        });
        let request = json!({"model": "gpt-4o-mini", "input": "hello"});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_123".to_owned(), 0);

        let tool_mapped =
            super::chat_completions::chat_response_to_response_resource(&tool_response, &context).unwrap();
        let filter_mapped =
            super::chat_completions::chat_response_to_response_resource(&filter_response, &context).unwrap();

        assert_eq!(tool_mapped["output"][0]["type"], "function_call");
        assert_eq!(tool_mapped["output"][0]["call_id"], "call_weather");
        assert_eq!(tool_mapped["output"][0]["arguments"], "{\"city\":\"NYC\"}");
        assert_eq!(filter_mapped["status"], "incomplete");
        assert_eq!(filter_mapped["incomplete_details"], json!({"reason": "content_filter"}));
    }

    #[test]
    fn private_web_search_call_maps_to_canonical_hosted_output() {
        let response = json!({
            "id": "chatcmpl-web-search",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_search_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{\"query\":\"Praxis Proxy\"}"}
                    }]
                }
            }]
        });
        let request = json!({
            "model": "gpt-4o-mini",
            "input": "search",
            "tools": [{"type": "web_search", "search_context_size": "high"}]
        });
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_web_search".to_owned(), 0);

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(
            mapped["output"][0],
            json!({
                "id": "call_search_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "Praxis Proxy"}
            })
        );
        assert_eq!(mapped["tools"], request["tools"]);
    }

    #[test]
    fn malformed_private_web_search_arguments_are_rejected() {
        let request = json!({"model": "m", "tools": [{"type": "web_search"}]});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_ws".to_owned(), 0);

        let oversized = serde_json::to_string(&json!({"query": "x".repeat(4097)})).unwrap();
        let cases = [
            "{}",
            "{\"query\":\"\"}",
            "{\"query\":42}",
            "{\"query\":\"ok\",\"extra\":true}",
            "not json",
            oversized.as_str(),
        ];
        for arguments in cases {
            let response = json!({
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "tool_calls": [{
                        "id": "call_ws",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": arguments}
                    }]}
                }]
            });
            let error = super::chat_completions::chat_response_to_response_resource(&response, &context)
                .unwrap_err()
                .to_string();
            assert!(
                error.starts_with("invalid synthesized web_search function call:"),
                "{error}"
            );
        }
    }

    #[test]
    fn chat_refusals_map_to_response_refusal_content() {
        let response = json!({
            "id": "chatcmpl-refusal",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": null, "refusal": "I can't help with that."}
            }]
        });
        let request = json!({"model": "gpt-4o-mini", "input": "hello"});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_refusal".to_owned(), 0);

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["output"][0]["type"], "message");
        assert_eq!(
            mapped["output"][0]["content"],
            json!([{"type": "refusal", "refusal": "I can't help with that."}])
        );
    }

    #[test]
    fn chat_response_logprobs_map_to_output_text_logprobs() {
        let response = json!({
            "id": "chatcmpl-logprobs",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "logprobs": {
                    "content": [{
                        "token": "Hi",
                        "logprob": -0.1,
                        "bytes": [72, 105],
                        "top_logprobs": [{"token": "Hi", "logprob": -0.1, "bytes": [72, 105]}]
                    }]
                }
            }]
        });
        let request = json!({"model": "gpt-4o-mini", "input": "hello", "top_logprobs": 1});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_logprobs".to_owned(), 0);

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(
            mapped["output"][0]["content"][0]["logprobs"],
            response["choices"][0]["logprobs"]["content"]
        );
    }

    // -------------------------------------------------------------------------
    // ResponseContext construction
    // -------------------------------------------------------------------------

    #[test]
    fn response_context_from_request_extracts_all_fields() {
        let request = json!({
            "model": "gpt-4o",
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "hi"}],
            "metadata": {"key": "value"},
            "text": {"format": {"type": "json_object"}},
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 256,
            "max_tool_calls": 5,
            "parallel_tool_calls": false,
            "previous_response_id": "resp_prev",
            "store": false,
            "tools": [{"type": "function", "name": "f"}],
            "tool_choice": "required",
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "top_logprobs": 3,
            "service_tier": "flex",
            "safety_identifier": "safe-1",
            "prompt_cache_key": "cache-1"
        });

        let ctx =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_test".to_owned(), 42);

        assert_eq!(ctx.response_id, "resp_test");
        assert_eq!(ctx.created_at, 42);
        assert!(ctx.completed_at.is_none());
        assert_eq!(ctx.model, "gpt-4o");
        assert_eq!(ctx.instructions, Some("Be helpful."));
        assert_eq!(ctx.input.cloned(), Some(json!([{"role": "user", "content": "hi"}])));
        assert_eq!(ctx.metadata.cloned(), Some(json!({"key": "value"})));
        assert_eq!(ctx.text.cloned(), Some(json!({"format": {"type": "json_object"}})));
        assert_eq!(ctx.temperature.cloned(), Some(json!(0.7)));
        assert_eq!(ctx.top_p.cloned(), Some(json!(0.95)));
        assert_eq!(ctx.max_output_tokens, Some(256));
        assert_eq!(ctx.max_tool_calls, Some(5));
        assert!(!ctx.parallel_tool_calls);
        assert_eq!(ctx.previous_response_id, Some("resp_prev"));
        assert!(!ctx.store);
        assert_eq!(ctx.tools.len(), 1);
        assert_eq!(ctx.tool_choice.cloned(), Some(json!("required")));
        assert_eq!(ctx.presence_penalty.cloned(), Some(json!(0.1)));
        assert_eq!(ctx.frequency_penalty.cloned(), Some(json!(0.2)));
        assert_eq!(ctx.top_logprobs, Some(3));
        assert_eq!(ctx.service_tier.cloned(), Some(json!("flex")));
        assert_eq!(ctx.safety_identifier.cloned(), Some(json!("safe-1")));
        assert_eq!(ctx.prompt_cache_key.cloned(), Some(json!("cache-1")));
    }

    #[test]
    fn response_context_defaults_for_minimal_request() {
        let request = json!({});
        let ctx = super::chat_completions::ResponseContext::from_responses_request(&request, "resp_min".to_owned(), 0);

        assert_eq!(ctx.model, "");
        assert!(ctx.instructions.is_none());
        assert!(ctx.input.is_none());
        assert!(ctx.metadata.is_none());
        assert!(ctx.text.is_none());
        assert!(ctx.temperature.is_none());
        assert!(ctx.top_p.is_none());
        assert!(ctx.max_output_tokens.is_none());
        assert!(ctx.max_tool_calls.is_none());
        assert!(ctx.parallel_tool_calls);
        assert!(ctx.previous_response_id.is_none());
        assert!(ctx.store);
        assert!(ctx.tools.is_empty());
        assert!(ctx.tool_choice.is_none());
        assert!(ctx.presence_penalty.is_none());
        assert!(ctx.frequency_penalty.is_none());
        assert!(ctx.top_logprobs.is_none());
        assert!(ctx.service_tier.is_none());
        assert!(ctx.safety_identifier.is_none());
        assert!(ctx.prompt_cache_key.is_none());
    }

    #[test]
    fn response_context_from_non_object_uses_defaults() {
        let request = json!("not an object");
        let ctx = super::chat_completions::ResponseContext::from_responses_request(&request, "resp_str".to_owned(), 0);

        assert_eq!(ctx.model, "");
        assert!(ctx.instructions.is_none());
    }

    #[test]
    fn response_context_with_completed_at_sets_timestamp() {
        let request = json!({"model": "m"});
        let ctx = super::chat_completions::ResponseContext::from_responses_request(&request, "resp_1".to_owned(), 10)
            .with_completed_at(20);

        assert_eq!(ctx.completed_at, Some(20));
    }

    // -------------------------------------------------------------------------
    // Request translation: message edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn message_item_without_content_gets_empty_content() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{"role": "user"}]
        }));

        assert_eq!(mapped["messages"][0]["content"], "");
    }

    #[test]
    fn untyped_item_without_role_or_content_returns_error() {
        let error = map_error(&json!({
            "model": "m",
            "input": [{"foo": "bar"}]
        }));

        assert_eq!(
            error,
            "unsupported Responses input item type for Chat Completions translation: unknown"
        );
    }

    #[test]
    fn untyped_item_with_content_key_maps_as_message() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{"content": "implicit user"}]
        }));

        assert_eq!(mapped["messages"][0]["role"], "user");
        assert_eq!(mapped["messages"][0]["content"], "implicit user");
    }

    #[test]
    fn non_object_input_items_are_skipped() {
        let mapped = map(&json!({
            "model": "m",
            "input": [42, "bare string", {"role": "user", "content": "real"}]
        }));

        assert_eq!(mapped["messages"].as_array().unwrap().len(), 1);
        assert_eq!(mapped["messages"][0]["content"], "real");
    }

    // -------------------------------------------------------------------------
    // Request translation: content part edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn text_only_content_parts_collapse_to_string() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello "},
                    {"type": "output_text", "text": "world"},
                    {"type": "text", "text": "!"}
                ]
            }]
        }));

        assert_eq!(mapped["messages"][0]["content"], "Hello world!");
    }

    #[test]
    fn mixed_content_parts_stay_as_array() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Look at this:"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"}
                ]
            }]
        }));

        let content = mapped["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn text_parts_without_text_field_are_skipped() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text"},
                    {"type": "input_text", "text": "valid"}
                ]
            }]
        }));

        assert_eq!(mapped["messages"][0]["content"], "valid");
    }

    #[test]
    fn content_part_without_type_returns_error() {
        let error = map_error(&json!({
            "model": "m",
            "input": [{"role": "user", "content": [{"text": "no type"}]}]
        }));

        assert_eq!(
            error,
            "unsupported Responses content part type for Chat Completions translation: unknown"
        );
    }

    #[test]
    fn input_image_without_url_returns_error() {
        let error = map_error(&json!({
            "model": "m",
            "input": [{"role": "user", "content": [{"type": "input_image"}]}]
        }));

        assert!(error.contains("input_image requires image_url"));
    }

    #[test]
    fn string_content_passes_through_directly() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{"role": "user", "content": "plain text"}]
        }));

        assert_eq!(mapped["messages"][0]["content"], "plain text");
    }

    // -------------------------------------------------------------------------
    // Request translation: function call edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn function_call_with_non_string_arguments_serializes_to_json() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": {"city": "NYC"}
            }]
        }));

        assert_eq!(
            mapped["messages"][0]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"NYC\"}"
        );
    }

    #[test]
    fn function_call_without_arguments_uses_empty_string() {
        let mapped = map(&json!({
            "model": "m",
            "input": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "ping"
            }]
        }));

        assert_eq!(mapped["messages"][0]["tool_calls"][0]["function"]["arguments"], "");
    }

    #[test]
    fn malformed_function_calls_without_call_id_or_name_are_dropped() {
        let mapped = map(&json!({
            "model": "m",
            "input": [
                {"type": "function_call", "name": "missing_id", "arguments": "{}"},
                {"type": "function_call", "call_id": "missing_name", "arguments": "{}"},
                {"type": "function_call", "call_id": "valid_call", "name": "valid_function", "arguments": "{}"}
            ]
        }));

        let tool_calls = mapped["messages"][0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "valid_call");
        assert_eq!(tool_calls[0]["function"]["name"], "valid_function");
    }

    #[test]
    fn function_call_output_with_non_string_output_serializes() {
        let mapped = map(&json!({
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": 42}
            ]
        }));

        assert_eq!(mapped["messages"][1]["content"], "42");
    }

    #[test]
    fn function_call_output_without_output_uses_empty_string() {
        let mapped = map(&json!({
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1"}
            ]
        }));

        assert_eq!(mapped["messages"][1]["content"], "");
    }

    #[test]
    fn function_calls_separated_by_other_items_produce_separate_messages() {
        let mapped = map(&json!({
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "f1", "arguments": "{}"},
                {"role": "user", "content": "interlude"},
                {"type": "function_call", "call_id": "c2", "name": "f2", "arguments": "{}"}
            ]
        }));

        let messages = mapped["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(messages[0]["tool_calls"][0]["id"], "c1");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(messages[2]["tool_calls"][0]["id"], "c2");
    }

    // -------------------------------------------------------------------------
    // Request translation: tool definitions
    // -------------------------------------------------------------------------

    #[test]
    fn pre_wrapped_function_tool_passes_through() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "tools": [{
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}}
            }]
        }));

        assert_eq!(mapped["tools"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn function_tool_missing_required_name_is_forwarded_to_provider() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "tools": [{"type": "function", "parameters": {"type": "object"}}]
        }));

        assert_eq!(mapped["tools"][0]["type"], "function");
        assert!(mapped["tools"][0]["function"].get("name").is_none());
        assert_eq!(mapped["tools"][0]["function"]["parameters"], json!({"type": "object"}));
    }

    #[test]
    fn tool_object_without_type_returns_unknown_tool_error() {
        let error = map_error(&json!({
            "model": "m",
            "input": "hello",
            "tools": [{"name": "missing_type"}]
        }));

        assert_eq!(
            error,
            "unsupported Responses tool type for Chat Completions translation: unknown"
        );
    }

    #[test]
    fn empty_tools_array_omits_tools_field() {
        let mapped = map(&json!({"model": "m", "input": "hello", "tools": []}));

        assert!(mapped.get("tools").is_none());
    }

    #[test]
    fn non_object_tool_entries_are_skipped() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "tools": ["not_an_object"]
        }));

        assert!(mapped.get("tools").is_none());
    }

    // -------------------------------------------------------------------------
    // Request translation: tool_choice edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn string_tool_choice_passes_through() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "tool_choice": "required"
        }));

        assert_eq!(mapped["tool_choice"], "required");
    }

    #[test]
    fn tool_choice_none_means_no_field() {
        let mapped = map(&json!({"model": "m", "input": "hello"}));

        assert!(mapped.get("tool_choice").is_none());
    }

    #[test]
    fn non_string_non_object_tool_choice_is_rejected() {
        let error = map_error(&json!({"model": "m", "input": "hello", "tool_choice": 42}));

        assert_eq!(
            error,
            "unsupported Responses tool_choice type for Chat Completions translation: number"
        );
    }

    #[test]
    fn tool_choice_object_without_type_is_rejected() {
        let error = map_error(&json!({"model": "m", "input": "hello", "tool_choice": {"name": "f"}}));

        assert_eq!(
            error,
            "unsupported Responses tool_choice type for Chat Completions translation: unknown"
        );
    }

    #[test]
    fn mcp_tool_choice_is_rejected_in_base_converter() {
        let error = map_error(&json!({
            "model": "m",
            "input": "hello",
            "tool_choice": {
                "type": "mcp",
                "server_label": "docs"
            }
        }));

        assert!(error.ends_with(": mcp"));
    }

    // -------------------------------------------------------------------------
    // Request translation: text format edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn json_schema_format_with_nested_json_schema_key_passes_through() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "weather",
                        "schema": {"type": "object"}
                    }
                }
            }
        }));

        assert_eq!(mapped["response_format"]["json_schema"]["name"], "weather");
    }

    #[test]
    fn text_format_type_plain_text_does_not_set_response_format() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "text": {"format": {"type": "text"}}
        }));

        assert!(mapped.get("response_format").is_none());
    }

    #[test]
    fn text_format_without_type_does_not_set_response_format() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "text": {"format": {"unknown": true}}
        }));

        assert!(mapped.get("response_format").is_none());
    }

    #[test]
    fn text_without_format_key_does_not_set_response_format() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "text": {"other": true}
        }));

        assert!(mapped.get("response_format").is_none());
    }

    #[test]
    fn no_text_field_does_not_set_response_format() {
        let mapped = map(&json!({"model": "m", "input": "hello"}));

        assert!(mapped.get("response_format").is_none());
    }

    // -------------------------------------------------------------------------
    // Request translation: reasoning
    // -------------------------------------------------------------------------

    #[test]
    fn reasoning_without_effort_does_not_set_reasoning_effort() {
        let mapped = map(&json!({
            "model": "m",
            "input": "hello",
            "reasoning": {"other": true}
        }));

        assert!(mapped.get("reasoning_effort").is_none());
    }

    #[test]
    fn no_reasoning_field_does_not_set_reasoning_effort() {
        let mapped = map(&json!({"model": "m", "input": "hello"}));

        assert!(mapped.get("reasoning_effort").is_none());
    }

    // -------------------------------------------------------------------------
    // Response translation: non-object error
    // -------------------------------------------------------------------------

    #[test]
    fn non_object_chat_response_returns_expected_object_error() {
        let request = json!({"model": "m", "input": "hello"});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_1".to_owned(), 0);

        let error =
            super::chat_completions::chat_response_to_response_resource(&json!("not an object"), &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Chat Completions response must be a JSON object")
        );
    }

    // -------------------------------------------------------------------------
    // Response translation: finish reasons
    // -------------------------------------------------------------------------

    #[test]
    fn length_finish_reason_maps_to_incomplete_status() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = simple_chat_response("length", "truncated");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["status"], "incomplete");
        assert_eq!(mapped["incomplete_details"], json!({"reason": "max_output_tokens"}));
    }

    #[test]
    fn stop_finish_reason_maps_to_completed_status() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "done");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["status"], "completed");
        assert_eq!(mapped["incomplete_details"], Value::Null);
    }

    #[test]
    fn tool_calls_finish_reason_maps_to_completed_status() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "tool_calls", "index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["status"], "completed");
    }

    // -------------------------------------------------------------------------
    // Response translation: completed_at behavior
    // -------------------------------------------------------------------------

    #[test]
    fn completed_at_uses_completed_at_context_when_available() {
        let request = json!({"model": "m", "input": "hello"});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_1".to_owned(), 100)
                .with_completed_at(200);
        let response = simple_chat_response("stop", "done");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["completed_at"], 200);
    }

    #[test]
    fn completed_at_falls_back_to_created_at_when_not_set() {
        let request = json!({"model": "m", "input": "hello"});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_1".to_owned(), 100);
        let response = simple_chat_response("stop", "done");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["completed_at"], 100);
    }

    // -------------------------------------------------------------------------
    // Response translation: no choices
    // -------------------------------------------------------------------------

    #[test]
    fn response_with_no_choices_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": []
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(error.to_string().contains("choices must contain an object"));
    }

    #[test]
    fn response_without_choices_key_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m"
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(error.to_string().contains("choices must be an array"));
    }

    #[test]
    fn malformed_first_choices_are_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let malformed = [
            (
                "non-object choice",
                json!({"choices": [null]}),
                "choices must contain an object",
            ),
            (
                "missing finish reason",
                json!({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}),
                "first choice must contain a string finish_reason",
            ),
            (
                "unsupported finish reason",
                json!({
                    "choices": [{
                        "finish_reason": "unknown",
                        "message": {"role": "assistant", "content": "ok"}
                    }]
                }),
                "first choice contains an unsupported finish_reason",
            ),
            (
                "missing message",
                json!({"choices": [{"finish_reason": "stop"}]}),
                "first choice must contain a message object",
            ),
            (
                "invalid role",
                json!({
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {"role": "user", "content": "ok"}
                    }]
                }),
                "first choice message must have the assistant role",
            ),
            (
                "missing output",
                json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}]}),
                "first choice message has no supported output",
            ),
        ];

        for (case, response, expected) in malformed {
            let error =
                super::chat_completions::chat_response_to_response_resource(&response, &context).expect_err(case);

            assert!(error.to_string().contains(expected), "{case}: {error}");
        }
    }

    // -------------------------------------------------------------------------
    // Response translation: empty content
    // -------------------------------------------------------------------------

    #[test]
    fn response_with_empty_string_content_on_completed_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "");

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("first choice message has no supported output"),
            "{error}"
        );
    }

    #[test]
    fn response_with_empty_array_content_on_completed_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": []}}]
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("first choice message has no supported output"),
            "{error}"
        );
    }

    #[test]
    fn response_with_only_empty_text_parts_on_completed_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": ""}
                    ]
                }
            }]
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("first choice message has no supported output"),
            "{error}"
        );
    }

    #[test]
    fn response_with_only_empty_refusal_on_completed_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": null, "refusal": ""}
            }]
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("first choice message has no supported output"),
            "{error}"
        );
    }

    #[test]
    fn response_with_empty_content_on_incomplete_terminal_is_preserved() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        for empty in [json!(""), json!([]), json!([{"type": "text", "text": ""}])] {
            let response = json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{
                    "finish_reason": "length",
                    "index": 0,
                    "message": {"role": "assistant", "content": empty}
                }]
            });

            let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context)
                .expect("incomplete terminal with empty content should be preserved");

            assert_eq!(mapped["status"], "incomplete", "{empty}");
            assert_eq!(
                mapped["incomplete_details"],
                json!({"reason": "max_output_tokens"}),
                "{empty}"
            );
            assert_eq!(mapped["output"], json!([]), "{empty}");
        }
    }

    #[test]
    fn response_with_mixed_empty_and_nonempty_parts_keeps_nonempty() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "kept"}
                    ]
                }
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        let content = mapped["output"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "kept");
    }

    #[test]
    fn response_with_null_content_on_completed_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": null}}]
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("first choice message has no supported output"),
            "{error}"
        );
    }

    #[test]
    fn response_with_null_content_on_incomplete_terminal_is_preserved() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        for (finish_reason, reason) in [("length", "max_output_tokens"), ("content_filter", "content_filter")] {
            let response = json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {"role": "assistant", "content": null}
                }]
            });

            let mapped =
                super::chat_completions::chat_response_to_response_resource(&response, &context).expect(finish_reason);

            assert_eq!(mapped["status"], "incomplete", "{finish_reason}");
            assert_eq!(
                mapped["incomplete_details"],
                json!({"reason": reason}),
                "{finish_reason}"
            );
            assert_eq!(mapped["output"], json!([]), "{finish_reason}");
        }
    }

    // -------------------------------------------------------------------------
    // Response translation: content with text + refusal combined
    // -------------------------------------------------------------------------

    #[test]
    fn response_with_both_content_and_refusal_includes_both() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Partial response",
                    "refusal": "Cannot continue"
                }
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        let content = mapped["output"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "output_text");
        assert_eq!(content[0]["text"], "Partial response");
        assert_eq!(content[1]["type"], "refusal");
        assert_eq!(content[1]["refusal"], "Cannot continue");
    }

    #[test]
    fn empty_refusal_string_is_not_included() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "ok", "refusal": ""}
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        let content = mapped["output"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "output_text");
    }

    // -------------------------------------------------------------------------
    // Response translation: content parts array
    // -------------------------------------------------------------------------

    #[test]
    fn response_with_array_content_extracts_text_parts() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " World"}
                    ]
                }
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        let content = mapped["output"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Hello");
        assert_eq!(content[1]["text"], " World");
    }

    #[test]
    fn logprobs_only_attached_to_first_text_part() {
        let request = json!({"model": "m", "input": "hello", "top_logprobs": 1});
        let context = make_response_context(&request);
        let logprob_entry = json!({"token": "Hi", "logprob": -0.1, "bytes": [72, 105], "top_logprobs": []});
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " World"}
                    ]
                },
                "logprobs": {"content": [logprob_entry]}
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        let content = mapped["output"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["logprobs"].as_array().unwrap().len(), 1);
        assert_eq!(content[1]["logprobs"], json!([]));
    }

    #[test]
    fn empty_text_parts_in_array_are_skipped() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "valid"}
                    ]
                }
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        let content = mapped["output"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "valid");
    }

    // -------------------------------------------------------------------------
    // Response translation: usage
    // -------------------------------------------------------------------------

    #[test]
    fn usage_with_cached_and_reasoning_tokens() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 20}
            }
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["usage"]["input_tokens"], 100);
        assert_eq!(mapped["usage"]["output_tokens"], 50);
        assert_eq!(mapped["usage"]["total_tokens"], 150);
        assert_eq!(mapped["usage"]["input_tokens_details"]["cached_tokens"], 80);
        assert_eq!(mapped["usage"]["output_tokens_details"]["reasoning_tokens"], 20);
    }

    #[test]
    fn missing_usage_defaults_to_zeros() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["usage"]["input_tokens"], 0);
        assert_eq!(mapped["usage"]["output_tokens"], 0);
        assert_eq!(mapped["usage"]["total_tokens"], 0);
        assert_eq!(mapped["usage"]["input_tokens_details"]["cached_tokens"], 0);
        assert_eq!(mapped["usage"]["output_tokens_details"]["reasoning_tokens"], 0);
    }

    #[test]
    fn usage_without_details_defaults_cached_and_reasoning_to_zero() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["usage"]["input_tokens_details"]["cached_tokens"], 0);
        assert_eq!(mapped["usage"]["output_tokens_details"]["reasoning_tokens"], 0);
    }

    // -------------------------------------------------------------------------
    // Response translation: service tier
    // -------------------------------------------------------------------------

    #[test]
    fn service_tier_from_response_takes_precedence() {
        let request = json!({"model": "m", "input": "hello", "service_tier": "flex"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "service_tier": "scale"
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["service_tier"], "scale");
    }

    #[test]
    fn service_tier_falls_back_to_context() {
        let request = json!({"model": "m", "input": "hello", "service_tier": "flex"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["service_tier"], "flex");
    }

    #[test]
    fn service_tier_defaults_when_absent_from_both() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "ok"}}]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["service_tier"], "default");
    }

    // -------------------------------------------------------------------------
    // Response translation: tool call output items
    // -------------------------------------------------------------------------

    #[test]
    fn multiple_tool_calls_produce_multiple_function_call_items() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{\"a\":1}"}},
                        {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": "{\"b\":2}"}}
                    ]
                }
            }]
        });

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["output"][0]["type"], "function_call");
        assert_eq!(mapped["output"][0]["call_id"], "c1");
        assert_eq!(mapped["output"][0]["name"], "f1");
        assert_eq!(mapped["output"][0]["id"], "fc_c1");
        assert_eq!(mapped["output"][1]["type"], "function_call");
        assert_eq!(mapped["output"][1]["call_id"], "c2");
        assert_eq!(mapped["output"][1]["name"], "f2");
        assert_eq!(mapped["output"][1]["id"], "fc_c2");
    }

    #[test]
    fn tool_call_missing_function_fields_is_rejected() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {}}]
                }
            }]
        });

        let error = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("message contains an invalid function tool call")
        );
    }

    // -------------------------------------------------------------------------
    // Response translation: response resource shape
    // -------------------------------------------------------------------------

    #[test]
    fn response_resource_includes_all_required_fields() {
        let request = json!({
            "model": "m",
            "input": "hello",
            "instructions": "Be brief.",
            "metadata": {"env": "test"},
            "max_output_tokens": 100,
            "max_tool_calls": 3,
            "parallel_tool_calls": false,
            "previous_response_id": "resp_prev",
            "store": false,
            "tools": [{"type": "function", "name": "f"}],
            "tool_choice": "required",
            "temperature": 0.5,
            "top_p": 0.8,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "top_logprobs": 2,
            "safety_identifier": "safe-1",
            "prompt_cache_key": "cache-1"
        });
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "done");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["id"], "resp_test");
        assert_eq!(mapped["object"], "response");
        assert_eq!(mapped["created_at"], 100);
        assert_eq!(mapped["status"], "completed");
        assert_eq!(mapped["error"], Value::Null);
        assert_eq!(mapped["instructions"], "Be brief.");
        assert_eq!(mapped["max_output_tokens"], 100);
        assert_eq!(mapped["max_tool_calls"], 3);
        assert_eq!(mapped["model"], "m");
        assert_eq!(mapped["input"], "hello");
        assert!(!mapped["parallel_tool_calls"].as_bool().unwrap());
        assert_eq!(mapped["previous_response_id"], "resp_prev");
        assert_eq!(mapped["reasoning"], Value::Null);
        assert!(!mapped["store"].as_bool().unwrap());
        assert_eq!(mapped["temperature"], 0.5);
        assert_eq!(mapped["top_p"], 0.8);
        assert_eq!(mapped["tool_choice"], "required");
        assert_eq!(mapped["tools"].as_array().unwrap().len(), 1);
        assert_eq!(mapped["truncation"], "disabled");
        assert_eq!(mapped["metadata"], json!({"env": "test"}));
        assert!(!mapped["background"].as_bool().unwrap());
        assert_eq!(mapped["presence_penalty"], 0.1);
        assert_eq!(mapped["frequency_penalty"], 0.2);
        assert_eq!(mapped["top_logprobs"], 2);
        assert_eq!(mapped["safety_identifier"], "safe-1");
        assert_eq!(mapped["prompt_cache_key"], "cache-1");
        assert_eq!(mapped["completed_at"], 200);
    }

    #[test]
    fn response_resource_defaults_for_minimal_request() {
        let request = json!({"model": "m"});
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "ok");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["instructions"], Value::Null);
        assert_eq!(mapped["max_output_tokens"], Value::Null);
        assert_eq!(mapped["max_tool_calls"], Value::Null);
        assert!(mapped["parallel_tool_calls"].as_bool().unwrap());
        assert_eq!(mapped["previous_response_id"], Value::Null);
        assert!(mapped["store"].as_bool().unwrap());
        assert_eq!(mapped["temperature"], 1.0);
        assert_eq!(mapped["top_p"], 1.0);
        assert_eq!(mapped["tool_choice"], "auto");
        assert_eq!(mapped["tools"], json!([]));
        assert_eq!(mapped["metadata"], json!({}));
        assert_eq!(mapped["presence_penalty"], 0.0);
        assert_eq!(mapped["frequency_penalty"], 0.0);
        assert_eq!(mapped["top_logprobs"], 0);
        assert_eq!(mapped["safety_identifier"], Value::Null);
        assert_eq!(mapped["prompt_cache_key"], Value::Null);
    }

    #[test]
    fn non_object_metadata_defaults_to_empty_object() {
        let request = json!({"model": "m", "metadata": "not an object"});
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "ok");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["metadata"], json!({}));
    }

    #[test]
    fn non_number_temperature_defaults_to_one() {
        let request = json!({"model": "m", "temperature": "warm"});
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "ok");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["temperature"], 1.0);
    }

    // -------------------------------------------------------------------------
    // Response translation: message item id
    // -------------------------------------------------------------------------

    #[test]
    fn message_output_item_id_prefixed_with_msg() {
        let request = json!({"model": "m", "input": "hello"});
        let context =
            super::chat_completions::ResponseContext::from_responses_request(&request, "resp_abc".to_owned(), 0);
        let response = simple_chat_response("stop", "hi");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["output"][0]["id"], "msg_resp_abc");
        assert_eq!(mapped["output"][0]["role"], "assistant");
        assert_eq!(mapped["output"][0]["status"], "completed");
    }

    // -------------------------------------------------------------------------
    // Response translation: output_text annotations and logprobs shape
    // -------------------------------------------------------------------------

    #[test]
    fn output_text_items_include_empty_annotations() {
        let request = json!({"model": "m", "input": "hello"});
        let context = make_response_context(&request);
        let response = simple_chat_response("stop", "text");

        let mapped = super::chat_completions::chat_response_to_response_resource(&response, &context).unwrap();

        assert_eq!(mapped["output"][0]["content"][0]["annotations"], json!([]));
        assert_eq!(mapped["output"][0]["content"][0]["logprobs"], json!([]));
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn make_response_context(request: &Value) -> super::chat_completions::ResponseContext<'_> {
        super::chat_completions::ResponseContext::from_responses_request(request, "resp_test".to_owned(), 100)
            .with_completed_at(200)
    }

    fn simple_chat_response(finish_reason: &str, content: &str) -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "finish_reason": finish_reason,
                "index": 0,
                "message": {"role": "assistant", "content": content}
            }]
        })
    }
}
