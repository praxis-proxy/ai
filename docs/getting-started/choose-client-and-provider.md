# Choose a client and provider

Pick the client protocol first, then the upstream and its authentication.
Praxis AI is alpha software. This page describes the current checkout; for a
released binary, use the matching documentation and build features. The linked
references own configuration details and translation limits. A listed adapter
does not mean every provider operation or model is supported.

## Decide by client and backend

**Standard** means `cargo build -p praxis-ai-proxy` on current main. Pass
other features with `--features`; the passive `experimental` marker adds no
capability by itself. Native forwarding relies on the upstream's own API.
Translation covers a bounded subset, not every operation under a provider URL.
See [build groups](../features.md#cargo-features).

| Client API → upstream | Mode and guide/configuration | Build | Authentication and limits |
| --- | --- | --- | --- |
| Chat Completions → OpenAI or compatible Chat backend | Native; [complete hosted recipe](build-from-source.md#chat-completions) | Standard | Caller bearer in hosted recipe; backend must serve the requested model and SSE for streaming |
| Messages → native Anthropic or native vLLM | Native; [recipe](../guides/anthropic.md#native-messages), [hosted Anthropic YAML](../anthropic-messages.md#passthrough-to-anthropic-api) | Standard; recipe adds `basic-auth-filter` | Hosted `x-api-key` or caller bearer; vLLM uses isolated backend bearer. Native backend determines operation support |
| Messages → Chat backend | Translation, plus the SSE adapter for streaming; [recipe](../guides/anthropic.md#translated-messages) | Standard; isolated recipe adds `basic-auth-filter` | Strip caller Anthropic credentials, inject backend key; thinking/cache opt-ins are lossy; no count-tokens translation |
| Chat Completions → Anthropic Messages | Selected-provider translation; [operator recipe](../guides/anthropic.md#chat-client-to-selected-messages-provider), [compatibility guide](../developing/openai-anthropic-compatibility.md) | Standard | Projected selected-provider secret via `credential_inject`; explicit translation permission required; supported versions and ignored/rejected fields in [reference](../filters/apis/openai_chat_completions_to_anthropic.md) |
| Chat Completions → Azure OpenAI | Adaptation (API version, response normalization); [provider recipe](../guides/providers/azure-openai.md), [reference](../filters/providers/openai_chat_completions_to_azureai_chat_completions.md) | Standard; Entra injection adds `azure-ad-filter` | Resource/deployment/API version plus API key or supported Entra client secret flow; safety annotation loss is a [known gap](../conformance/translation-matrix.md#known-gaps) |
| Chat Completions → Vertex Gemini | Translation to `generateContent`; [provider recipe](../guides/providers/vertex-gemini.md), [reference](../filters/providers/openai_chat_completions_to_vertexai_gemini.md) | Standard; token acquisition adds `gcp-adc-filter` | Authorized GCP identity, project/location/model; schema strictness, reasoning/cache accounting gaps in [matrix](../conformance/translation-matrix.md) |
| Messages → Vertex Anthropic | Endpoint adaptation to `rawPredict`; [provider recipe](../guides/providers/vertex-anthropic.md), [reference](../filters/providers/anthropic_messages_to_vertexai_anthropic.md) | `vertex-anthropic-filter`; add `gcp-adc-filter` for token acquisition | Authorized GCP identity and model access; selected cluster metadata and model/provider resolution required |
| Chat Completions → Bedrock Converse | Translation; [provider recipe](../guides/providers/bedrock.md), [compatibility matrix](../conformance/translation-matrix.md) | Standard includes signer | AWS key/secret/region and model access; signing alone is not translation. History, reasoning and cache gaps remain; see [audit](../developing/documentation-audit.md) for reference gap |
| Responses → native Responses backend | Native; [quickstart](quickstart.md), [passthrough example](../../examples/configs/openai/responses/http-passthrough.yaml) | Standard | Caller key in quickstart; no local state, validation, or tool execution supplied by forwarding |
| Responses → gateway-managed native Responses or translated Chat | Managed, experimental; [recipe](../guides/managed-responses.md) | `openai-responses`; recipe selects `openai-all,store-sqlite` | Backend and tool credentials are separate; managed restrictions and translation gaps apply |
| Conversations → local gateway store | Managed, experimental; [state guide](../guides/managed-responses.md#conversations-and-state), [reference](../filters/agents/openai_conversations.md) | `openai-conversations,store-sqlite` or selected PostgreSQL backend | Explicit owner/auth boundary; local CRUD is distinct from forwarding upstream Conversations |

See the [Responses support and qualification table](../conformance/openai-responses-state-conformance.md#responses-support-and-qualification) for state parameters, SDK versions, and the native and translated vLLM test lanes.

The [cloud provider runbook](../guides/providers/README.md) adds credentials,
endpoint/Host/TLS selection, validation, startup, requests, and cleanup to the
example configurations. It assumes account permissions, regional model
availability, and an authorized credential source are already in place. The
[recipe index](../guides/README.md) lists the native and managed paths.

### Read compatibility status correctly

- **Native:** bytes are forwarded without protocol conversion; the backend owns
  semantics. A route matching `/v1` does not implement all `/v1` APIs locally.
- **Translated:** only the documented mapping is implemented; finite and SSE
  paths need separate evidence and filters.
- **Lossy:** accepted input loses documented semantics. Enable only deliberate
  opt-ins, and monitor degradation signals where provided.
- **Rejected:** the adapter refuses an unrepresentable feature. For example,
  Messages-to-Chat rejects document blocks and server-tool definitions.
- **Known gap:** accepted behavior is incomplete or incorrect. Consult the
  [generated matrix](../conformance/translation-matrix.md) and its source evidence;
  do not interpret a fixture as provider-wide qualification.

For verified CLI edits, permission modes and compaction outcomes by model and API route, see [Coding harness compatibility](../conformance/coding-harness-compatibility.md).

### Codex CLI qualification

Open the [release you intend to deploy](https://github.com/praxis-proxy/ai/releases)
and read **Codex CLI capabilities** in its attached `support-report.md`.

| Capability | Native Responses | Responses → Chat Completions | What the test verifies |
| --- | --- | --- | --- |
| Apply a file patch | Release report | Release report | Codex applies the patch, returns tool output through Praxis, and writes the exact expected file. |
| Cite a live web search | Release report | Release report | Codex completes a turn citing an exact source URL from a successful Tavily search through Praxis. |
| Compact a session | Release report | Outside test scope | Codex sends a summarization request and resumes with compacted history. |
| Finish a task after compaction | Release report | Outside test scope | Codex finishes the task and verifies the resulting file. |

All scenarios use HTTP/SSE. Edit and compaction disable web search and use full-access sandbox mode. The dedicated search cases enable live web search in read-only mode. The release report gives the outcome for the model tested on that release.
The matching `support-report.json` records each case's configuration (model
names, vLLM image, parsers, example configuration) under
`live_validation.suites.codex_acceptance.cases[].capabilities`, and its live
validation section names the tested commit. The
[GPU scenario guide](../developing/gpu-nightly-suite.md) describes the test setup.

## Provider authentication

For Azure Entra injection, configure the tenant ID, client ID, scope, and the
client-secret environment variable named in the
[authentication reference](../filters/providers/azure_ad.md); that filter
covers only client-secret authentication. For GCP injection, use the metadata
identity or service-account key source described in
[GCP ADC](../filters/providers/gcp_adc.md); a `gcloud` user credential file is
not supported. Authentication does not select the upstream endpoint or grant
the identity provider-side permissions.

When the gateway owns provider secrets, follow the routing prerequisites in
[credential injection](../filters/providers/credential_inject.md). The
quickstart's caller-key forwarding does not implement that flow.

## Managed APIs and agent protocols

| Goal | Reference | Build and prerequisites |
| --- | --- | --- |
| Manage Responses at the gateway instead of forwarding natively | [Responses proxy](../filters/agents/openai_responses_proxy.md) | Experimental `openai-responses` and an inference backend; storage-related groups also need a store backend and its database |
| Run an ephemeral server-side agent loop | [Agentic protocols](../architecture/agentic-protocols.md) | Experimental `openai-agentic-loop`, an inference backend, and credentials or services for the selected tools |
| Classify or broker MCP traffic | [MCP](../filters/agents/mcp.md) | Reachable MCP servers and their authentication; separate from managed Responses tool execution |
| Execute MCP tools within managed Responses | [MCP connector](../filters/agents/openai_mcp_connector.md) | Experimental `openai-mcp-tools`, a store backend, and configured MCP servers; select the agent loop separately if needed |

[Cargo features](../features.md#cargo-features) lists exact build combinations
and the [example configurations](../../examples/README.md) have complete filter
chains; a filter reference alone is not a deployment configuration. Validate
provider credentials, permissions, model availability, and operation support in
your deployment before relying on a route. Praxis, Praxis AI, and Policy are
versioned independently; appearing together in the documentation site does not
make them a tested compatibility set.
