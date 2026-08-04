# Anthropic Messages Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-streaming native Anthropic Messages model → You.com search → model loop using Praxis core's `iterative_request_router`, plus a deterministic local model mock and runnable example.

**Architecture:** Extract the existing search-provider client from the OpenAI Responses namespace into an internal protocol-neutral module while preserving `openai_web_search`. Add an `anthropic_web_search` adapter that uses IRR's accounted request/response state, recognizes exactly one `WebSearch` call, executes You.com on re-entry, appends assistant `tool_use` and user `tool_result` turns, and rejects streaming before upstream dispatch.

**Tech Stack:** Rust 1.96+, `praxis-filter` HTTP filters, Praxis core `SubRequestClient` and `iterative_request_router`, `serde_json`, `bytes`, You.com Search API, YAML examples, Rust integration-test TCP backends.

## Global Constraints

- Rust stable 1.96+ is the production compiler; Rust nightly is used for `rustfmt`.
- CMake 3.31+ is required by the Praxis/Pingora dependency build.
- Keep versioned Praxis dependencies; do not run `make patch-praxis` for this feature.
- Avoid cloning request, response, header, body, or SSE data unless ownership across an IRR iteration requires it.
- This version supports non-streaming native `/v1/messages` JSON only and rejects `"stream": true` before model or search calls.
- The proxy manages only exactly one `tool_use` named `WebSearch` with a non-empty string `input.query` no larger than 8192 UTF-8 bytes.
- Non-2xx model responses always pass through unchanged, even if their bodies resemble a managed tool-use message.
- Non-web-search, multiple, and mixed tool calls remain client-owned and pass through unchanged.
- Preserve the existing `openai_web_search` YAML contract, diagnostics, provider behavior, and tests.
- Include unit tests, an example config, a functional integration test, and synchronized generated docs.
- Never amend commits; add one commit per independently reviewable task.

---

## File Structure

- `apis/src/web_search/{mod.rs,config.rs,provider.rs}`: protocol-neutral configuration, client, normalized results, and formatter.
- `apis/src/anthropic/web_search/{mod.rs,tests.rs}`: Messages lifecycle, IRR-accounted state access, errors, request reconstruction, and unit tests.
- `filters/src/register.rs`: callout-aware filter registration.
- `examples/configs/anthropic/messages-web-search.yaml`: native Messages IRR pipeline.
- `tests/integration/tests/suite/examples/anthropic_messages_web_search.rs`: functional round trip.
- `tests/integration/fixtures/anthropic/messages/web_search_nonstreaming.json`: reformatted recording.
- `tests/utils/examples/anthropic_messages_web_search_mock.rs`: runnable model mock on port 8000.
- `docs/filters/anthropic_web_search.md`, `docs/filters/reference.md`, `examples/README.md`: generated docs.

---

### Task 1: Extract the protocol-neutral search provider

**Files:**
- Create: `apis/src/web_search/mod.rs`
- Create: `apis/src/web_search/config.rs`
- Create: `apis/src/web_search/provider.rs`
- Modify: `apis/src/lib.rs`
- Modify: `apis/src/openai/responses/web_search/mod.rs`
- Delete: `apis/src/openai/responses/web_search/config.rs`
- Delete: `apis/src/openai/responses/web_search/provider.rs`
- Test: `apis/src/web_search/config.rs`
- Test: `apis/src/web_search/provider.rs`
- Test: `apis/src/openai/responses/web_search/tests.rs`

**Interfaces:**
- Produces: `build_config(filter_name: &'static str, raw: &WebSearchFilterConfig) -> Result<ValidatedConfig, FilterError>`.
- Produces: `SearchClient::from_config(filter_name: &'static str, config: &ValidatedConfig, client: SubRequestClient) -> Result<SearchClient, FilterError>`.
- Produces: `SearchClient::search(&self, query: &str, context_size: Option<SearchContextSize>) -> SearchOutcome`.
- Produces: `format_search_results(results: &[SearchResult]) -> String`.
- Preserves: `WebSearchFilter::{from_config, from_config_with_client}` and registry name `openai_web_search`.

- [ ] **Step 1: Add the failing owner-diagnostic test**

Add to the current config tests:

```rust
#[test]
fn build_config_uses_owner_name_in_diagnostics() {
    let mut raw = base_config();
    raw.api_key = None;
    let error = build_config("anthropic_web_search", &raw).unwrap_err();
    assert!(
        error.to_string().contains("anthropic_web_search: api_key is required"),
        "diagnostic should name the owning filter: {error}"
    );
}
```

- [ ] **Step 2: Verify RED**

```console
cargo test -p praxis-ai-apis build_config_uses_owner_name_in_diagnostics
```

Expected: compilation fails because `build_config` currently accepts only one argument.

- [ ] **Step 3: Move provider code and parameterize validation**

Create `apis/src/web_search/mod.rs`:

```rust
//! Protocol-neutral web-search provider support.

pub(crate) mod config;
pub(crate) mod provider;

use std::fmt::Write as _;

pub(crate) use config::{
    SearchContextSize, ValidatedConfig, WebSearchFilterConfig, build_config,
};
pub(crate) use provider::{SearchClient, SearchOutcome, SearchResult};

pub(crate) fn format_search_results(results: &[SearchResult]) -> String {
    let mut output = String::with_capacity(results.len() * 200);
    for (index, result) in results.iter().enumerate() {
        if index > 0 { output.push_str("\n\n"); }
        let _infallible = write!(
            output,
            "[{}] {}\n{}\n{}",
            index + 1,
            result.title,
            result.url,
            result.snippet
        );
    }
    output
}
```

Move the existing config/provider bodies without changing HTTP or parsing behavior. Make the raw config and fields `pub(crate)`. Parameterize every error and `validate_max_body_bytes` call:

```rust
pub(crate) fn build_config(
    filter_name: &'static str,
    raw: &WebSearchFilterConfig,
) -> Result<ValidatedConfig, FilterError> {
    let raw_key = raw.api_key.as_ref().ok_or_else(|| {
        FilterError::from(format!("{filter_name}: api_key is required"))
    })?;
    let api_key = resolve_api_key(filter_name, raw_key.expose_secret())?;
    if api_key.is_empty() {
        return Err(FilterError::from(format!(
            "{filter_name}: api_key must not be empty"
        )));
    }
    build_validated_config(filter_name, raw, api_key)
}

fn build_validated_config(
    filter_name: &'static str,
    raw: &WebSearchFilterConfig,
    api_key: String,
) -> Result<ValidatedConfig, FilterError> {
    Ok(ValidatedConfig {
        provider: raw.provider,
        api_key: SecretString::from(api_key),
        default_context_size: validate_context_size(
            filter_name,
            raw.default_context_size.as_deref(),
        )?,
        timeout_ms: validate_timeout_ms(filter_name, raw.timeout_ms)?,
        max_body_bytes: validate_max_body_bytes_field(
            filter_name,
            raw.max_body_bytes,
        )?,
        failure_mode: raw.provider_failure_mode.unwrap_or(FailureMode::Closed),
        status_on_error: validate_status_on_error(
            filter_name,
            raw.status_on_error,
        )?,
        base_url: raw.base_url.clone(),
    })
}
```

Give the existing validators these exact owner-aware signatures and retain
their current defaults/range checks:

```rust
fn validate_timeout_ms(filter_name: &'static str, raw: Option<u64>) -> Result<u64, FilterError>;
fn validate_status_on_error(filter_name: &'static str, raw: Option<u16>) -> Result<u16, FilterError>;
fn validate_context_size(filter_name: &'static str, raw: Option<&str>) -> Result<SearchContextSize, FilterError>;
fn validate_max_body_bytes_field(filter_name: &'static str, raw: Option<usize>) -> Result<usize, FilterError>;
fn resolve_api_key(filter_name: &'static str, raw: &str) -> Result<String, FilterError>;
```

Add `pub(crate) mod web_search;` in `apis/src/lib.rs`. Update OpenAI imports and constructor:

```rust
use crate::web_search::{
    SearchClient, SearchContextSize, SearchOutcome, SearchResult,
    WebSearchFilterConfig, build_config, format_search_results,
};

let validated = build_config("openai_web_search", &cfg)?;
```

Delete the old nested config/provider files once imports compile.

- [ ] **Step 4: Verify GREEN and no OpenAI regression**

```console
cargo test -p praxis-ai-apis web_search
cargo clippy -p praxis-ai-apis --all-targets -- -D warnings
git diff --check
```

Expected: all existing OpenAI/provider tests and the new owner diagnostic pass; clippy and diff checks exit 0.

- [ ] **Step 5: Commit**

```console
git add apis/src/lib.rs apis/src/web_search apis/src/openai/responses/web_search
git commit -m "refactor(web_search): share provider across protocols"
```

---

### Task 2: Add Messages response classification

**Files:**
- Create: `apis/src/anthropic/web_search/mod.rs`
- Create: `apis/src/anthropic/web_search/tests.rs`
- Modify: `apis/src/anthropic/mod.rs`

**Interfaces:**
- Consumes: shared config/provider types from Task 1.
- Produces: `pub struct AnthropicWebSearchFilter` with both standard and shared-client factories.
- Produces: `anthropic_web_search.action` values `loop` and `done`.
- Preserves: all cross-iteration payloads in IRR's byte-accounted `IterationState`.

- [ ] **Step 1: Write failing lifecycle tests**

Create tests using `crate::test_utils::{make_filter_context, make_request}`:

```rust
#[tokio::test]
async fn streaming_request_is_rejected_before_reentry() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"test","max_tokens":32,"stream":true,"messages":[]}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    let FilterAction::Reject(rejection) = action else { panic!("expected rejection"); };
    assert_eq!(rejection.status, 400);
    assert!(String::from_utf8_lossy(rejection.body.as_ref().unwrap())
        .contains("streaming is not supported"));
}

#[tokio::test]
async fn sole_web_search_tool_use_signals_loop() {
    let filter = test_filter();
    let request = make_request(Method::POST, "/v1/messages");
    let mut ctx = initialized_context(&request).await;
    let mut body = Some(message_response(json!([{
        "type":"tool_use","id":"toolu_search_1","name":"WebSearch",
        "input":{"query":"potato"}
    }]), "tool_use"));
    filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert_eq!(result_action(&ctx).as_deref(), Some("loop"));
    let response: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let ResponseDecision::Managed(pending) = classify_response(&response) else {
        panic!("expected managed search");
    };
    assert_eq!(pending.query, "potato");
}

#[tokio::test]
async fn client_owned_and_mixed_tools_signal_done() {
    for content in [
        json!([{"type":"tool_use","id":"toolu_bash","name":"Bash","input":{}}]),
        json!([
            {"type":"tool_use","id":"toolu_search","name":"WebSearch","input":{"query":"potato"}},
            {"type":"tool_use","id":"toolu_bash","name":"Bash","input":{}}
        ]),
    ] {
        let filter = test_filter();
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = initialized_context(&request).await;
        let original = message_response(content, "tool_use");
        let mut body = Some(original.clone());
        filter.on_response_body(&mut ctx, &mut body, true).unwrap();
        assert_eq!(result_action(&ctx).as_deref(), Some("done"));
        assert_eq!(body, Some(original));
    }
}
```

Also add explicit tests named `managed_call_without_query_is_rejected`, `final_text_signals_done_without_mutating_body`, `non_message_error_body_signals_done`, `non_end_of_stream_is_noop`, and `initial_request_body_is_not_mutated`.

- [ ] **Step 2: Verify RED**

```console
cargo test -p praxis-ai-apis anthropic::web_search
```

Expected: compilation fails because the module and filter do not exist.

- [ ] **Step 3: Implement constructors and classification**

Use these types:

```rust
const FILTER_NAME: &str = "anthropic_web_search";
const ACTION_LOOP: &str = "loop";
const ACTION_DONE: &str = "done";

struct PendingSearch {
    id: String,
    query: String,
}

pub struct AnthropicWebSearchFilter {
    default_context_size: SearchContextSize,
    max_body_bytes: usize,
    search_client: SearchClient,
}
```

Factories parse `WebSearchFilterConfig`, call `build_config(FILTER_NAME, &cfg)`, and build `SearchClient`. Use `ReadWrite`/bounded `StreamBuffer` for requests and `ReadOnly`/bounded `StreamBuffer` for responses. Reject `stream == true`; leave the original request body in IRR's already-accounted `original_request`.

Implement a pure classifier:

```rust
enum ResponseDecision {
    Done,
    Managed(PendingSearch),
    InvalidManagedCall,
}

fn classify_response(response: &Value) -> ResponseDecision {
    let stop_reason = response.get("stop_reason").and_then(Value::as_str);
    if response.get("type").and_then(Value::as_str) != Some("message")
        || response.get("role").and_then(Value::as_str) != Some("assistant")
        || !matches!(stop_reason, Some("tool_use" | "end_turn"))
    { return ResponseDecision::Done; }
    let Some(content) = response.get("content").and_then(Value::as_array) else {
        return ResponseDecision::Done;
    };
    let tools: Vec<usize> = content.iter().enumerate()
        .filter_map(|(i, block)|
            (block.get("type").and_then(Value::as_str) == Some("tool_use")).then_some(i))
        .collect();
    if tools.len() != 1 { return ResponseDecision::Done; }
    let tool = &content[tools[0]];
    if tool.get("name").and_then(Value::as_str) != Some("WebSearch") {
        return ResponseDecision::Done;
    }
    let Some(id) = tool.get("id").and_then(Value::as_str).filter(|v| !v.is_empty()) else {
        return ResponseDecision::InvalidManagedCall;
    };
    let Some(query) = tool.get("input").and_then(|v| v.get("query"))
        .and_then(Value::as_str).map(str::trim).filter(|v| !v.is_empty()) else {
        return ResponseDecision::InvalidManagedCall;
    };
    let id = id.to_owned();
    let query = query.to_owned();
    ResponseDecision::Managed(PendingSearch { id, query })
}
```

Parse from existing `Bytes`, publish only the action, and leave response bytes unchanged. IRR installs the response as its accounted `previous_response`; do not retain parsed payloads in private extensions. Add `set_action` through `ctx.filter_results`. Construct rejections with:

```rust
fn anthropic_rejection(status: u16, error_type: &str, message: &str) -> Rejection {
    let body = serde_json::json!({"error":{"type":error_type,"message":message}});
    Rejection::status(status)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(body.to_string()))
}
```

Export the filter from `apis/src/anthropic/mod.rs`.

- [ ] **Step 4: Verify GREEN and commit**

```console
cargo test -p praxis-ai-apis anthropic::web_search
cargo +nightly fmt --all -- --check
git add apis/src/anthropic/mod.rs apis/src/anthropic/web_search
git commit -m "feat(anthropic): detect Messages web search calls"
```

---

### Task 3: Execute search and rebuild the next Messages request

**Files:**
- Modify: `apis/src/anthropic/web_search/mod.rs`
- Modify: `apis/src/anthropic/web_search/tests.rs`

**Interfaces:**
- Consumes: `PendingSearch`, `IterationState`, `NextIterationBody`, `SearchClient`, and `format_search_results`.
- Produces: a re-entry body containing the complete assistant turn and matching user `tool_result`.
- Preserves: model, system, original history, tools, and unrelated top-level request fields.
- Accounts: the latest serialized request under `anthropic_web_search.request` in IRR's accumulator.

- [ ] **Step 1: Write failing re-entry tests**

Add a local You.com-compatible stub returning `{"results":{"web":[{"title":"Potato - Wikipedia","url":"https://en.wikipedia.org/wiki/Potato","description":"Potato is a starchy tuber native to the Americas."}],"news":[]}}`. Add:

```rust
#[tokio::test]
async fn pending_search_executes_and_appends_tool_result() {
    let search = start_you_search_stub(200, valid_you_body());
    let filter = test_filter_impl_with_base_url(search.base_url(), "closed");
    let pending = pending_search("potato");
    let results = filter.execute_pending_search(&pending).await.unwrap();
    let mut rebuilt = base_request();
    append_search_turns(
        &mut rebuilt,
        assistant_content("potato"),
        pending,
        &results,
    ).unwrap();
    assert_eq!(rebuilt["model"], "openai/gpt-oss-20b");
    assert_eq!(rebuilt["tools"][0]["name"], "WebSearch");
    assert_eq!(rebuilt["tool_choice"], json!({"type":"auto"}));
    let messages = rebuilt["messages"].as_array().unwrap();
    assert_eq!(messages[messages.len() - 2]["role"], "assistant");
    assert_eq!(messages[messages.len() - 1]["content"][0]["type"], "tool_result");
    assert_eq!(messages[messages.len() - 1]["content"][0]["tool_use_id"], "toolu_search_1");
    assert!(messages[messages.len() - 1]["content"][0]["content"]
        .as_str().unwrap().contains("Potato - Wikipedia"));
    assert_eq!(search.last_json()["query"], "potato");
    assert!(search.last_request()
        .to_ascii_lowercase()
        .contains("x-api-key: test-key"));
}
```

Also add `closed_provider_failure_returns_anthropic_error`, `open_provider_failure_appends_no_results_tool_result`, and `accounted_previous_response_recovers_complete_assistant_content`. Functional tests cover the real IRR re-entry boundary because external crates cannot construct an `IterationState` whose framework fields are private.

- [ ] **Step 2: Verify RED**

```console
cargo test -p praxis-ai-apis pending_search_executes_and_appends_tool_result
```

Expected: test fails because request re-entry does not execute or rebuild.

- [ ] **Step 3: Implement provider mapping and request reconstruction**

```rust
async fn execute_pending_search(
    &self,
    pending: &PendingSearch,
) -> Result<Vec<SearchResult>, Rejection> {
    match self.search_client
        .search(&pending.query, Some(self.default_context_size)).await
    {
        SearchOutcome::Results(results) => Ok(results),
        SearchOutcome::Skipped => Ok(Vec::new()),
        SearchOutcome::Rejected { status } => Err(anthropic_rejection(
            status, "api_error", "web search provider unavailable",
        )),
    }
}

fn append_search_turns(
    request: &mut Value,
    assistant_content: Vec<Value>,
    pending: PendingSearch,
    results: &[SearchResult],
) -> Result<(), Rejection> {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return Err(anthropic_rejection(
            400, "invalid_request_error", "messages must be an array for web search re-entry",
        ));
    };
    let content = if results.is_empty() {
        "No search results found.".to_owned()
    } else {
        format_search_results(results)
    };
    let mut assistant_turn = serde_json::Map::new();
    assistant_turn.insert("role".to_owned(), Value::String("assistant".to_owned()));
    assistant_turn.insert("content".to_owned(), Value::Array(assistant_content));
    messages.push(Value::Object(assistant_turn));
    messages.push(json!({"role":"user","content":[{
        "type":"tool_result","tool_use_id":pending.id,"content":content
    }]}));
    if request.get("tool_choice").is_some() {
        request.as_object_mut().unwrap().insert(
            "tool_choice".to_owned(), json!({"type":"auto"}),
        );
    }
    Ok(())
}
```

On re-entry, reclassify `IterationState.previous_response.body` and move out its complete content. Load the latest request from `IterationState.accumulator`, falling back to `original_request.body` before the first mutation. Execute the provider call, append both turns, and serialize once. Reject with HTTP 413 before assignment when `rebuilt.len() > max_body_bytes`; otherwise insert the same `Bytes` allocation into the accumulator, `NextIterationBody`, and the active body, then set `Content-Type`. IRR's post-hook `retained_bytes()` check enforces `max_state_bytes` before transport.

- [ ] **Step 4: Verify GREEN and commit**

```console
cargo test -p praxis-ai-apis anthropic::web_search
cargo test -p praxis-ai-apis web_search
cargo clippy -p praxis-ai-apis --all-targets -- -D warnings
git add apis/src/anthropic/web_search
git commit -m "feat(anthropic): execute web search in Messages loop"
```

---

### Task 4: Register the filter and add the IRR example

**Files:**
- Modify: `filters/src/register.rs`
- Create: `examples/configs/anthropic/messages-web-search.yaml`
- Test: `filters/src/register.rs`

**Interfaces:**
- Consumes: `AnthropicWebSearchFilter::{from_config, from_config_with_client}`.
- Produces: registry key `anthropic_web_search`, sharing the runtime `SubRequestClient` when present.
- Produces: example listener `127.0.0.1:8080`, You.com configuration, and model endpoint `127.0.0.1:8000`.

- [ ] **Step 1: Add the failing registry assertion**

```rust
assert!(
    names.contains(&"anthropic_web_search"),
    "expected anthropic_web_search in registry"
);
```

- [ ] **Step 2: Verify RED**

```console
cargo test -p praxis-ai-filters build_ai_registry_includes_ai_and_builtin_filters
```

Expected: the assertion fails because the registry does not contain the filter.

- [ ] **Step 3: Add callout-aware registration**

Pass `subrequest_client` into `register_anthropic_filters` and add:

```rust
#[expect(clippy::panic, reason = "matches register_filters! macro convention")]
fn register_anthropic_web_search(
    registry: &mut FilterRegistry,
    subrequest_client: Option<&SubRequestClient>,
) {
    if let Some(client) = subrequest_client {
        let client = client.clone();
        registry.register(
            "anthropic_web_search",
            praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |config| {
                praxis_ai_apis::anthropic::AnthropicWebSearchFilter::from_config_with_client(
                    config,
                    client.clone(),
                )
            })),
        ).unwrap_or_else(|_| panic!("duplicate filter name: 'anthropic_web_search'"));
    } else {
        praxis_filter::register_filters!(
            @register registry,
            http "anthropic_web_search" => praxis_ai_apis::anthropic::AnthropicWebSearchFilter::from_config
        );
    }
}
```

Call it from `register_anthropic_filters` after the protocol/validation registrations.

- [ ] **Step 4: Add the complete example YAML**

Create `examples/configs/anthropic/messages-web-search.yaml`:

```yaml
# Anthropic Messages Web Search
#
# Runs a non-streaming native Messages model -> You.com search -> model
# loop through Praxis core's iterative_request_router.
# Requires WEB_SEARCH_API_KEY. The Messages backend listens on 127.0.0.1:8000.

listeners:
  - name: anthropic-web-search
    address: "127.0.0.1:8080"
    filter_chains: [messages-web-search]

filter_chains:
  - name: messages-web-search
    filters:
      - filter: anthropic_messages_format
        on_invalid: reject
      - filter: anthropic_validate
      - filter: iterative_request_router
        initial_step: inference
        max_iterations: 6
        timeout_ms: 90000
        steps:
          - name: inference
            filters:
              - filter: anthropic_web_search
                provider: you
                api_key: ${WEB_SEARCH_API_KEY}
                default_context_size: medium
                timeout_ms: 10000
                provider_failure_mode: closed
              - filter: anthropic_messages_protocol
                default_version: "2023-06-01"
              - filter: router
                routes:
                  - path_prefix: "/v1/messages"
                    cluster: messages-backend
              - filter: load_balancer
                clusters:
                  - name: messages-backend
                    endpoints: ["127.0.0.1:8000"]
            on_result:
              - filter: anthropic_web_search
                key: action
                value: loop
                next: inference
              - default: true
                done: true
```

- [ ] **Step 5: Verify GREEN and commit**

```console
cargo test -p praxis-ai-filters build_ai_registry_includes_ai_and_builtin_filters
WEB_SEARCH_API_KEY=test-key cargo test -p praxis-tests-schema all_example_configs_parse
git add filters/src/register.rs examples/configs/anthropic/messages-web-search.yaml
git commit -m "feat(anthropic): configure Messages web search loop"
```

Expected: registry and schema/example config suites pass.

---

### Task 5: Add the reformatted fixture and functional round trip

**Files:**
- Create: `tests/integration/fixtures/anthropic/messages/web_search_nonstreaming.json`
- Create: `tests/integration/tests/suite/examples/anthropic_messages_web_search.rs`
- Modify: `tests/integration/tests/suite/examples/mod.rs`

**Interfaces:**
- Consumes: example config, `StatefulCapturingBackend`, `patch_yaml`, `start_proxy`, and HTTP helpers.
- Produces fixture keys: `initial_request`, `first_model_response`, `search_response`, `final_model_response`.

- [ ] **Step 1: Add the exact Messages fixture**

```json
{
  "source": "Reformatted from agentic_api/web_search_nonstreaming.json; sanitized vLLM plus You.com recording.",
  "initial_request": {
    "model": "openai/gpt-oss-20b",
    "max_tokens": 1024,
    "stream": false,
    "messages": [{"role": "user", "content": "Use web search to look up potato, then summarize in one sentence."}],
    "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
  },
  "first_model_response": {
    "id": "msg_web_search_01",
    "type": "message",
    "role": "assistant",
    "model": "openai/gpt-oss-20b",
    "content": [{"type": "tool_use", "id": "toolu_web_search_01", "name": "WebSearch", "input": {"query": "potato"}}],
    "stop_reason": "tool_use",
    "stop_sequence": null,
    "usage": {"input_tokens": 20, "output_tokens": 8}
  },
  "search_response": {
    "results": {"web": [{"title": "Potato - Wikipedia", "url": "https://en.wikipedia.org/wiki/Potato", "description": "Potato is a starchy underground tuber native to the Americas."}], "news": []}
  },
  "final_model_response": {
    "id": "msg_web_search_02",
    "type": "message",
    "role": "assistant",
    "model": "openai/gpt-oss-20b",
    "content": [{"type": "text", "text": "Potato is a starchy underground tuber native to the Americas and now eaten worldwide."}],
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {"input_tokens": 74, "output_tokens": 18}
  }
}
```

- [ ] **Step 2: Write the failing functional test**

Load the fixture as `Value`, prepend a text block to a local copy of the first
model response before its `WebSearch` block, start `StatefulCapturingBackend`
with the two model responses, start a one-request You.com stub, patch
port/key/base URL, and assert:

```rust
assert_eq!(parse_status(&raw), 200);
let client_response: Value = serde_json::from_str(&parse_body(&raw)).unwrap();
assert_eq!(client_response, fixture["final_model_response"]);

let requests = model.requests();
assert_eq!(requests.len(), 2, "model should receive two Messages requests");
assert_eq!(requests[0].uri, "/v1/messages");
assert_eq!(requests[1].uri, "/v1/messages");
let second: Value = serde_json::from_str(&requests[1].body).unwrap();
assert_eq!(second["model"], fixture["initial_request"]["model"]);
assert_eq!(second["tools"], fixture["initial_request"]["tools"]);
let messages = second["messages"].as_array().unwrap();
assert_eq!(
    messages[messages.len() - 2]["content"],
    first_model_response["content"],
);
assert_eq!(messages[messages.len() - 1]["content"][0]["type"], "tool_result");
assert_eq!(messages[messages.len() - 1]["content"][0]["tool_use_id"], "toolu_web_search_01");
assert!(messages[messages.len() - 1]["content"][0]["content"]
    .as_str().unwrap().contains("Potato - Wikipedia"));
assert_eq!(search.request_count(), 1);
assert_eq!(search.last_json()["query"], "potato");
assert!(search.last_request()
    .to_ascii_lowercase()
    .contains("x-api-key: test-key"));
```

Add a second test sending `stream: true` and asserting HTTP 400, zero model requests, and zero search requests. Register the module in `examples/mod.rs`.

- [ ] **Step 3: Verify RED**

```console
cargo test -p praxis-tests-integration --test suite examples::anthropic_messages_web_search
```

Expected: the new round trip fails at the first unconnected lifecycle/config boundary; it must not pass without two model requests and one search request.

- [ ] **Step 4: Correct integration boundaries without weakening assertions**

Use this deterministic loader patch:

```rust
let yaml = patch_yaml(
    &yaml,
    proxy_port,
    &HashMap::from([("127.0.0.1:8000", model.port())]),
);
let yaml = yaml.replace(
    "api_key: ${WEB_SEARCH_API_KEY}",
    &format!(
        "api_key: test-key\n                base_url: http://127.0.0.1:{}",
        search.port()
    ),
);
```

Limit fixes to YAML patching, stub framing, fixture loading, response serialization, and filter lifecycle wiring.

- [ ] **Step 5: Verify GREEN and commit**

```console
cargo test -p praxis-tests-integration --test suite examples::anthropic_messages_web_search
cargo test -p praxis-tests-integration --test suite examples::anthropic_messages
cargo test -p praxis-tests-integration --test suite examples::agentic_loop
git add tests/integration/fixtures/anthropic/messages/web_search_nonstreaming.json \
  tests/integration/tests/suite/examples/anthropic_messages_web_search.rs \
  tests/integration/tests/suite/examples/mod.rs
git commit -m "test(anthropic): cover Messages web search round trip"
```

Expected: new tests pass; the existing 11 Anthropic and 5 Responses agentic-loop examples remain green.

---

### Task 6: Add the runnable mock, generated docs, and final verification

**Files:**
- Create: `tests/utils/examples/anthropic_messages_web_search_mock.rs`
- Modify: `examples/configs/anthropic/messages-web-search.yaml`
- Generate: `docs/filters/anthropic_web_search.md`
- Modify: `docs/filters/reference.md`
- Modify: `examples/README.md`

**Interfaces:**
- Produces: standalone model mock listening on `127.0.0.1:8000`.
- Produces: exact mock/Praxis/curl commands in the example comments and generated docs.

- [ ] **Step 1: Write failing mock selector tests**

```rust
fn response_for_request(request: &Value) -> Value;

#[test]
fn request_without_tool_result_gets_tool_use() {
    let response = response_for_request(&json!({
        "messages":[{"role":"user","content":"search potato"}]
    }));
    assert_eq!(response["stop_reason"], "tool_use");
    assert_eq!(response["content"][0]["name"], "WebSearch");
}

#[test]
fn matching_tool_result_gets_final_text() {
    let response = response_for_request(&json!({"messages":[{
        "role":"user","content":[{
            "type":"tool_result","tool_use_id":"toolu_web_search_01",
            "content":"[1] Potato - Wikipedia"
        }]
    }]}));
    assert_eq!(response["stop_reason"], "end_turn");
    assert_eq!(response["content"][0]["type"], "text");
}
```

- [ ] **Step 2: Verify RED**

```console
cargo test -p praxis-test-utils --example anthropic_messages_web_search_mock
```

Expected: compilation fails because the example does not exist.

- [ ] **Step 3: Implement the standalone mock**

Implement a `TcpListener` server that parses `Content-Length`, accepts `POST /v1/messages`, returns 400 for invalid JSON, and selects by presence of the matching `tool_result`, not connection count:

```rust
fn main() -> std::io::Result<()> {
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
```

Use the exact IDs/text from Task 5, serialize once, and send `Content-Type: application/json`, exact `Content-Length`, and `Connection: close`.

- [ ] **Step 4: Test the mock and document the live demo**

```console
cargo test -p praxis-test-utils --example anthropic_messages_web_search_mock
```

Add these commands to the YAML header:

```console
cargo run -p praxis-test-utils --example anthropic_messages_web_search_mock
WEB_SEARCH_API_KEY="$WEB_SEARCH_API_KEY" cargo run -p praxis-ai-proxy -- \
  -c examples/configs/anthropic/messages-web-search.yaml
curl http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":"Use web search to look up potato, then summarize in one sentence."}],"tools":[{"name":"WebSearch","description":"Search the web","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}'
```

If a live You.com key is unavailable, report this smoke as skipped and rely on the deterministic provider integration test.

- [ ] **Step 5: Generate documentation**

Ensure the filter Rust docs contain short/full YAML examples, then run:

```console
cargo xtask generate-filter-docs
cargo xtask sync-example-readme --fix
```

Expected: the new filter reference, reference index, and Anthropic example table entry are generated.

- [ ] **Step 6: Run final verification**

```console
cargo +nightly fmt --all -- --check
cargo test -p praxis-ai-apis web_search
cargo test -p praxis-ai-filters
cargo test -p praxis-tests-integration --test suite examples::anthropic_messages_web_search
cargo test -p praxis-tests-integration --test suite examples::anthropic_messages
cargo test -p praxis-tests-integration --test suite examples::agentic_loop
cargo test -p praxis-test-utils --example anthropic_messages_web_search_mock
cargo xtask lint-filter-docs
cargo xtask sync-example-readme
cargo xtask lint-markdown-links
cargo clippy -p praxis-ai-apis -p praxis-ai-filters -p praxis-test-utils \
  -p praxis-tests-integration --all-targets -- -D warnings
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 7: Commit demo and generated docs**

```console
git add tests/utils/examples/anthropic_messages_web_search_mock.rs \
  examples/configs/anthropic/messages-web-search.yaml \
  docs/filters/anthropic_web_search.md docs/filters/reference.md \
  examples/README.md
git commit -m "docs(anthropic): add runnable Messages web search demo"
```

- [ ] **Step 8: Verify the committed branch**

```console
git status --short
git log --oneline --decorate -7
```

Expected: status is empty and history contains the design commit plus the six implementation commits above.
