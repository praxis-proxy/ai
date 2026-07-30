---
issue: https://github.com/praxis-proxy/ai/issues/76
discussion: https://github.com/praxis-proxy/ai/issues/76
status: accepted
authors:
  - liavweiss
  - christinaexyou
graduation_criteria:
  - How? section with requirements and design reviewed
  - ai_guardrails filter scaffold merged
  - NeMo request-side integration passing e2e tests
stakeholders:
  - christinaexyou
  - shaneutt
  - twghu
---

> **Current status (2026-07):** Task 0 (scaffold) and Task 1
> (NeMo provider + request-side integration) are merged and covered
> by unit, integration, and example e2e tests — the graduation
> criteria above are met. Task 2 (redact / mask action,
> [#579](https://github.com/praxis-proxy/praxis/issues/579)) and
> Task 3 (response-side evaluation,
> [#580](https://github.com/praxis-proxy/praxis/issues/580)) are not
> yet implemented, and the shared-types plan for `security/guardrails`
> was dropped. See the **Design** and **Known Limitations** sections
> for what shipped vs. what deviated or is still deferred.

# External Guardrail Provider Integration

## What?

Create a new **AI guardrails filter** (`ai_guardrails`) under
`filters/src/guardrails/` that calls external content safety providers
via HTTP, inspects request and response bodies, and acts on the
provider's verdict: pass, block, or redact (mask).

This is a standalone filter, separate from the existing
`security/guardrails` filter (which handles local string/regex
rules). Both filters can coexist in the same pipeline.

A `GuardProvider` trait makes the filter generic - adding a new
provider means implementing one trait, not duplicating filter
logic. The first provider is NeMo Guardrails, using the
`/v1/guardrail/checks` endpoint.

### Goals

- New standalone filter under `filters/src/guardrails/`
- Generic provider trait so new providers are a single-file
  addition
- Request-side guardrails: evaluate client requests before
  forwarding to the LLM
- Response-side guardrails: evaluate LLM responses before
  returning to the client
- Three outcomes: pass (forward unchanged), block (reject
  with 403), redact (mask sensitive content)
- Common message extraction from OpenAI Chat and MCP body
  formats, shared by all providers
- Shared guardrails types (`GuardResult`, `GuardPhase`) in
  `builtins/http/guardrails.rs` for reuse by both AI and
  security guardrails filters
- Pipeline coexistence with existing `security/guardrails`
  filter (local rules run first, AI guardrails run after)
- Error handling via the existing pipeline-level
  `failure_mode` (fail-open/fail-closed)

### Non-Goals

- Modifying the existing `security/guardrails` filter
- Streaming (SSE) response inspection in v1 (buffered-only)
- A generic webhook provider (each provider has its own
  trait implementation)

## Why?

### Motivation

The current guardrails filter supports local string and regex matching
on headers and request bodies. This catches simple patterns (e.g.
"DROP TABLE") but cannot detect nuanced policy violations, prompt
injection, or PII without calling a specialized external service.

External providers like NeMo Guardrails and AWS Bedrock Guardrails
offer content safety capabilities (topic blocking, PII detection,
prompt injection detection) that are impractical to replicate with
regex rules. Without this integration, operators must either deploy
a separate proxy layer for content safety or accept the limitations
of local pattern matching.

A dedicated AI guardrails filter gives operators a clean
separation: local pattern matching stays in
`security/guardrails`, while external provider calls live in
`filters/src/guardrails`. Both can run in the same pipeline.

### User Stories

- As a proxy operator, I want to route AI requests through
  NeMo Guardrails so that prompt injection and policy
  violations are detected before reaching the LLM.

- As a security engineer, I want LLM responses inspected by
  an external provider so that sensitive data (PII, secrets)
  is masked before reaching the client.

- As a platform engineer, I want to configure fail-open or
  fail-closed behavior when the external provider is
  unreachable so that I can balance availability and safety
  per deployment.

- As a proxy operator, I want to use local rules and an
  external provider together in the same pipeline so that
  cheap local checks run first and expensive provider calls
  only happen when needed.

- As a developer, I want to add a new guardrail provider
  (e.g. Bedrock) by implementing a single trait so that I
  do not need to understand or modify the filter pipeline.

## How?

### Requirements

- New standalone filter (`ai_guardrails`) under
  `filters/src/guardrails/`
- Guardrails results need to indicate pass, block, and
  whether content was modified (redacted)
- Guardrails need to be configurable on both requests and
  responses
- Allow multiple backend providers to evaluate payloads
- NeMo (specifically) needs to be supported in the first
  pass

### Design

#### Proposed Structure

> **As implemented:** the shared-types split below was dropped.
> `GuardResult`, `GuardPhase`, and `GuardProvider` all live inside
> `praxis-ai-filters`, private to the `ai_guardrails` filter — there
> is no `builtins/http/guardrails.rs` in praxis core, and
> `security/guardrails` (in `../praxis/filter/src/builtins/http/security/guardrails/`)
> was left untouched with its own internal types, not updated to
> import these. The two filters coexist as originally intended, but
> do not share result types.
>
> Actual layout (in the `praxis-ai` repo):
>
> ```
> filters/src/guardrails/
> ├── mod.rs             # Module root
> ├── config.rs          # YAML config types
> ├── filter.rs          # HttpFilter impl
> ├── tests.rs           # Unit tests
> └── providers/
>     ├── mod.rs         # GuardProvider trait + GuardResult/GuardPhase
>     └── nemo.rs        # NemoProvider: HTTP call + mapping
> ```

```
filter/src/
├── guardrails.rs              # Shared types: GuardResult, GuardPhase
│                              # (used by ai/ and security/ guardrails)
├── ai/
│   └── guardrails/
│       ├── mod.rs             # Module root
│       ├── config.rs          # YAML config types
│       ├── filter.rs          # HttpFilter impl
│       ├── tests.rs           # Unit tests
│       └── provider/
│           ├── mod.rs         # GuardProvider trait
│           └── nemo.rs        # NemoProvider: HTTP call + mapping
│
└── security/
    └── guardrails/            # Existing (untouched in this proposal)
```

#### Shared Guardrails Types

> **As implemented:** this section describes the original plan.
> `GuardResult`/`GuardPhase` were not extracted to a shared core
> module; they are defined directly in
> `filters/src/guardrails/providers/mod.rs`. Revisit this if a
> second AI provider or a `security/guardrails` convergence effort
> makes sharing worthwhile.

`guardrails.rs` lives at `builtins/http/`.
It holds types that both `ai/guardrails` and
`security/guardrails` can use.

```rust
// builtins/http/guardrails.rs

pub enum GuardPhase {
    Request,
    Response,
}

pub enum GuardResult {
    Pass,
    Block { reason: String },
    Redact { modified_text: String, reason: String },
}
```

As part of Task 0, the existing `security/guardrails`
filter will be updated to import `GuardResult` and
`GuardPhase` from this shared module (replacing its
internal equivalents where applicable).

#### GuardProvider Trait

The `GuardProvider` trait lives in `ai/guardrails/`
since it is specific to external provider calls.

```rust
// ai/guardrails/provider/mod.rs

use crate::builtins::http::guardrails::{
    GuardPhase, GuardResult,
};

#[async_trait]
pub trait GuardProvider: Send + Sync {
    async fn evaluate(
        &self,
        messages: Vec<serde_json::Value>,
        phase: GuardPhase,
    ) -> Result<GuardResult, FilterError>;
}
```

The provider receives:
- `messages`: pre-extracted by `filter.rs` from the
  OpenAI/MCP body
- `phase`: Request or Response (some providers may
  need this to set phase-specific fields)

#### Common Helper (filter.rs)

`extract_messages(body: &Value, phase: GuardPhase)`
`-> Result<Vec<serde_json::Value>, FilterError>` -
extracts messages from OpenAI Chat
(`messages[]` / `choices[].message.content`) or MCP
(`params.arguments` / `result.content[].text`) format.
Returns an error for unrecognized body formats (prevents
silently skipping guardrail evaluation).
Shared by all providers since the body format is determined
by the model server, not the guard provider.

> **As implemented:** `extract_messages` only recognizes the
> OpenAI Chat `{"messages": [...]}` shape today. MCP body formats
> and the `phase`-aware `choices[].message.content` / response
> extraction are not implemented — request bodies without a
> top-level `messages` array are rejected (fail-closed) rather than
> parsed as MCP.

#### Request Path Flow

1. Body is buffered via `StreamBuffer`
2. In `on_request_body` at EOS:
   - Parse body as JSON
   - Extract messages based on phase
   - Call `provider.evaluate(messages, Request)`
   - Act on result: Pass -> continue, Block -> reject 403,
     Redact -> replace body
   - On error -> propagate as `FilterError` (pipeline
     handles `failure_mode`)

> **Note:** If the existing `security/guardrails` filter is
> also configured in the pipeline, it runs first (cheap local
> checks before the expensive provider call).

#### Response Path Flow (deferred - requires async support)

1. If provider is configured and `phase.response` is true,
   buffer response body
2. In `on_response_body` at EOS:
   - Parse body, extract messages
   - Call `provider.evaluate(messages, Response)`
   - Act on result: Pass -> forward, Block -> replace with
     error, Redact -> replace body

> **Note:** `on_response_body` is currently sync in the
> `HttpFilter` trait (Pingora constraint). Request-side
> guardrails work today via `on_request_body` (async).
> Response-side guardrails require making `on_response_body`
> async or using an alternative mechanism. This proposal
> ships request-side first; response-side is a follow-up
> once async response-body support is available.

#### Configuration

> **As implemented:** `phase.response: true` is rejected as a hard
> config error at filter construction time (not a silent no-op),
> since response-side evaluation (#580) does not exist yet. The
> example below is updated to reflect that; see `Known Limitations`
> item 1.

```yaml
# Pipeline example: both filters together

# 1. Existing local guardrails (cheap, runs first)
- filter: guardrails
  action: reject
  rules:
    - target: body
      contains: "DROP TABLE"

# 2. New AI guardrails (external provider call)
# failure_mode is a pipeline-level field (set on the
# PipelineFilter wrapper, not inside the filter config).
- filter: ai_guardrails
  provider:
    type: nemo
    endpoint: "http://nemo:8000/v1/guardrail/checks"
    timeout_ms: 5000
  phase:
    request: true   # default: true
    response: false # default: false; `true` is a config error until #580
```

#### Config Types (Rust)

> **As implemented** (`filters/src/guardrails/config.rs` /
> `providers/nemo.rs`): `endpoint` and `timeout_ms` live in a
> provider-specific `NemoConfig`, not the shared `ProviderConfig` —
> `ProviderConfig` only holds `type` plus a `#[serde(flatten)]`
> opaque `serde_yaml::Value` that each provider parses itself. NeMo
> also gained an optional `model: String` field (default `""`), and
> the default `timeout_ms` is 10s, not 5s. `PhaseConfig::response`
> is still parsed (default `false`), but `AiGuardrailsFilter::from_config`
> now returns `Err` if it's `true`.

```rust
// ai/guardrails/config.rs

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Nemo,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    pub endpoint: String,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,  // validated in from_config: > 0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseConfig {
    #[serde(default = "default_true")]
    pub request: bool,
    #[serde(default)]
    pub response: bool,  // as implemented: `true` rejected in AiGuardrailsFilter::from_config until #580
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AiGuardrailsConfig {
    pub provider: ProviderConfig,   // required
    #[serde(default)]
    pub phase: PhaseConfig,
}
```

Since `provider` is required (not `Option`), a config without
a provider is a deserialization error - no runtime validation
needed.

#### Filter

> **As implemented:** `from_config` rejects `phase.response: true`
> outright instead of storing a `phase_response: bool` for later use.
> `Redact` currently maps to `Continue` (body forwarded unchanged) —
> see Task 2 / `Known Limitations`. There is no `forbidden()` helper;
> `Block` builds `Rejection::status(403).with_body(reason)` inline.

```rust
// ai/guardrails/filter.rs

pub struct AiGuardrailsFilter {
    provider: Box<dyn GuardProvider>,
    phase_response: bool,
}

impl AiGuardrailsFilter {
    /// Parse AiGuardrailsConfig, validate timeout_ms > 0,
    /// match on ProviderType to instantiate the provider.
    /// Returns `Err` if `phase.response` is `true` (#580 not shipped).
    pub fn from_config(config: &serde_yaml::Value)
        -> Result<Box<dyn HttpFilter>, FilterError>;
}

#[async_trait]
impl HttpFilter for AiGuardrailsFilter {
    fn name(&self) -> &'static str { "ai_guardrails" }

    /// Always ReadWrite (body may be replaced on redact).
    fn request_body_access(&self) -> BodyAccess;

    /// Always StreamBuffer (provider needs full body).
    fn request_body_mode(&self) -> BodyMode;

    /// on_request() returns Continue (no header inspection).

    /// Core logic at EOS: parse JSON -> extract_messages
    /// -> provider.evaluate -> act on Pass/Block/Redact.
    /// Errors propagate as FilterError.
    async fn on_request_body(
        &self, ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>, end_of_stream: bool,
    ) -> Result<FilterAction, FilterError>;

    // response_body_access(), response_body_mode(),
    // on_response_body() deferred to Task 3.
}

// Helpers:
//   extract_messages(body, phase) -> Result<Vec<Value>>
//   write_result(ctx, status) -> writes to FilterResultSet
//   forbidden() -> Reject(403)
```

#### NeMo Provider

> **As implemented:** `NemoProvider` additionally carries a `model:
> String` field (sent on every request, default `""`) and enforces a
> 1 MiB response cap via `Content-Length` and an incremental
> read-abort. `reason` for `Block`/`Redact` is derived by joining the
> names of any rails in `rails_status` whose `status` is `"blocked"`,
> sorted alphabetically — not a single free-text field from NeMo. An
> unrecognized `status` value is a hard `Err`, not silently treated
> as `Pass`.

```rust
// provider/nemo.rs

pub struct NemoProvider {
    client: reqwest::Client,
    endpoint: String,
    timeout: Duration,
}

#[async_trait]
impl GuardProvider for NemoProvider {
    async fn evaluate(
        &self,
        messages: Vec<serde_json::Value>,
        _phase: GuardPhase,
    ) -> Result<GuardResult, FilterError>
    {
        // NeMo infers phase from the message role field,
        // so `phase` is unused here. Other providers may
        // need it (e.g. Bedrock uses INPUT/OUTPUT).
        //
        // 1. Build NeMo payload from messages
        // 2. POST to self.endpoint (/v1/guardrail/checks)
        // 3. Map NeMo response status to GuardResult:
        //    "passed"  -> Pass
        //    "blocked" -> Block { reason }
        //    "modified" -> Redact { modified_text, reason }
        // 4. On HTTP error or timeout -> Err(FilterError)
    }
}
```

#### Provider Status Mapping

| Provider | Pass | Block | Redact |
|----------|------|-------|--------|
| NeMo | `"passed"` | `"blocked"` | `"modified"` + modified text |

> Future providers (e.g. AWS Bedrock Guardrails) would follow
> the same pattern - implement `GuardProvider`, map their
> native response statuses to `GuardResult`.

#### Adding a Future Provider

1. Create `provider/<name>.rs` implementing `GuardProvider`
2. Implement `evaluate` - build the provider-specific HTTP
   payload internally, call the service, map response to
   `GuardResult`
3. Add a variant to the `ProviderType` enum and a match
   arm in `from_config`
4. Consider a Cargo feature flag if the provider pulls in
   a large dependency tree

### Task Plan

**Task 0: Scaffold the `ai_guardrails` filter** — ✅ Done
(with deviation: `security/guardrails` was not updated to share
types; see `Shared Guardrails Types` above)
- Blocked by: nothing
- Create `builtins/http/guardrails.rs` with shared types:
  `GuardResult`, `GuardPhase`
- Add `pub(crate) mod guardrails;` to
  `builtins/http/mod.rs`
- Update `security/guardrails` to import shared types
  from `builtins/http/guardrails.rs`
- Create `ai/guardrails/` module with skeleton files:
  `mod.rs`, `config.rs`, `filter.rs`, `tests.rs`,
  `provider/mod.rs`
- Define `AiGuardrailsConfig`, `ProviderConfig`,
  `PhaseConfig`, `ProviderType` enum in `config.rs`
- Define `GuardProvider` trait in `provider/mod.rs`
  (imports `GuardResult`, `GuardPhase` from shared module)
- Implement `AiGuardrailsFilter` with `from_config()` and
  a pass-through `HttpFilter` impl (no provider call yet)
- Register `"ai_guardrails"` in the `FilterRegistry`
- Add `pub mod guardrails;` to `ai/mod.rs`
- Acceptance: `filter: ai_guardrails` is accepted by the
  pipeline. Existing filters and tests unaffected.

**Task 1: NeMo provider + request-side integration** — ✅ Done
- Blocked by: Task 0
- Implement `NemoProvider` in `provider/nemo.rs`
  - POST to `/v1/guardrail/checks` endpoint
  - Map NeMo response statuses: `"passed"` -> Pass,
    `"blocked"` -> Block, `"modified"` -> Redact
  - NeMo can be deployed via
    [TrustyAI Helm chart](https://github.com/trustyai-explainability/trustyai-llm-demo/tree/add-mcp-guardrails/mcp-guardrails/deploy)
- Add `extract_messages()` helper to `filter.rs`
  (extracts messages from OpenAI/MCP body formats;
  shared by all providers, validated against NeMo)
- Wire NeMo provider into `on_request_body()`: parse,
  extract, evaluate, act on result
- Integration tests with mock NeMo HTTP server: pass,
  block, redact, provider-down scenarios
- Acceptance: end-to-end request with `provider: nemo`
  calls NeMo endpoint. Provider errors propagate as
  FilterError.

**Task 2: NeMo mask / redact action** — Not started
(tracked as [#579](https://github.com/praxis-proxy/praxis/issues/579);
`GuardResult::Redact` currently maps to `Continue` with the body
forwarded unchanged)
- Blocked by: Task 1
- When NeMo returns `"modified"` status, the redacted
  content is available in the NeMo response at:
  `guardrails_data.log.activated_rails[-1]`
  `.executed_actions[-1].return_value`
- The `NemoProvider` extracts this redacted text and
  returns it as `GuardResult::Redact { modified_text }`
- In `filter.rs`: replace the original body bytes with
  the redacted content, update `Content-Length` header
- For request-side: replace the last user message
  content in the request body with the redacted version

**Task 3: NeMo response-side guardrails** — Not started
(tracked as [#580](https://github.com/praxis-proxy/praxis/issues/580);
`phase.response: true` is currently a hard config error rather than
a silent no-op, so this is a config-schema change too, not purely
additive)
- Blocked by: Task 1 + async `on_response_body` support
- `on_response_body` is currently sync (Pingora
  constraint). This task requires making
  `on_response_body` async. Until that happens,
  response-side is deferred and v1 covers request-side
  only.
- Reuse the same NeMo provider for response inspection -
  `extract_messages()` already handles response body
  formats (`choices[].message.content`)
- Call `provider.evaluate(messages, Response)` at EOS
- Act on result: Pass -> forward, Block -> replace with
  error, Redact -> replace body with masked text

Task 2 + Task 3 can be developed in parallel after
Task 1.

## Known Limitations & Follow-ups

1. **Response-side async support** - `on_response_body` is
   sync in the `HttpFilter` trait (Pingora constraint).
   Response-side guardrails require making
   `on_response_body` async (a sync-to-async bridge at
   the Pingora boundary). Tracked as
   [#580](https://github.com/praxis-proxy/praxis/issues/580).
   As implemented, `phase.response: true` is rejected as a
   hard config error at `AiGuardrailsFilter::from_config` time
   (not silently ignored), so operators get an immediate,
   actionable error instead of a filter that appears configured
   but never runs.

2. **Streaming responses** - Buffered-only for v1.
   MCP responses are returned as JSON-RPC wrapped inside
   SSE formatting, so the implementation should include
   logic to parse SSE payloads. Full SSE streaming
   support is deferred.

3. **Duplicate checks** - When both `security/guardrails`
   (built-in PII/regex) and `ai_guardrails` (e.g. NeMo
   with PII rails) are in the same pipeline, the same
   content may be checked twice. Acceptable for v1;
   deduplication strategies can be explored in a
   follow-up.

4. **Redact is a no-op forward** - `GuardResult::Redact`
   (NeMo `"modified"` status) currently maps to `Continue`;
   the provider's masked/redacted text (`modified_text`) is
   discarded and the original, unmodified body is forwarded
   to the upstream. Tracked as
   [#579](https://github.com/praxis-proxy/praxis/issues/579).
   Operators relying on NeMo's PII-masking rails should be
   aware that masking does not yet take effect at the proxy.

5. **No shared types with `security/guardrails`** - the
   `builtins/http/guardrails.rs` shared-types module described
   above was never created; `GuardResult`/`GuardPhase` are
   private to `praxis-ai-filters`. `security/guardrails` (in
   praxis core) keeps its own independent types. Revisit if a
   second AI provider or a convergence effort makes sharing
   worthwhile.

6. **No MCP message extraction** - `extract_messages()` only
   recognizes the OpenAI Chat `{"messages": [...]}` shape.
   Request bodies in MCP JSON-RPC format are rejected
   (fail-closed) rather than evaluated, contrary to the original
   "Common Helper" design.
