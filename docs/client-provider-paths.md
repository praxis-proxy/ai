# Client and provider paths

Choose the client protocol first, then the upstream and its authentication.
Praxis AI is alpha software. This index describes the current checkout; use
matching documentation and build features for a released binary. The linked
references own configuration details and translation limitations. A listed
adapter is not a guarantee that every provider operation or model is supported.

## Decide by client and backend

**Standard** below means `cargo build -p praxis-ai-proxy` for current main.
Additional features are passed with `--features`; the passive `experimental`
marker alone adds no capability. Native forwarding relies on the upstream's
API implementation. Translation supports a bounded subset, not every operation
under a provider URL. See [build groups](features.md#cargo-features).

| Client API → upstream | Mode and guide/configuration | Build | Authentication and limits |
| --- | --- | --- | --- |
| Chat Completions → OpenAI or compatible Chat backend | Native; [complete hosted recipe](quickstart.md#chat-completions) | Standard | Caller bearer in hosted recipe; backend must serve the requested model and SSE for streaming |
| Messages → native Anthropic or native vLLM | Native; [recipe](recipes/anthropic.md#native-messages), [hosted Anthropic YAML](anthropic-messages.md#passthrough-to-anthropic-api) | Standard; recipe adds `basic-auth-filter` | Hosted `x-api-key` or caller bearer; vLLM uses isolated backend bearer. Native backend determines operation support |
| Messages → Chat backend | Translation; [recipe](recipes/anthropic.md#translated-messages) | Standard; isolated recipe adds `basic-auth-filter` | Strip caller Anthropic credentials, inject backend key; thinking/cache opt-ins are lossy; no count-tokens translation |
| Chat Completions → Anthropic Messages | Selected-provider translation; [operator recipe](recipes/anthropic.md#chat-client-to-selected-messages-provider), [compatibility guide](developing/openai-anthropic-compatibility.md) | Standard | Projected selected-provider secret via `credential_inject`; explicit translation permission required; supported versions and ignored/rejected fields in [reference](filters/openai_chat_completions_to_anthropic.md) |
| Chat Completions → Azure OpenAI | Adaptation; [provider recipe](recipes/providers.md#azure-openai-chat-completions), [reference](filters/openai_chat_completions_to_azureai_chat_completions.md) | Standard; Entra injection adds `azure-ad-filter` | Resource/deployment/API version plus API key or supported Entra client secret flow; safety annotation loss is a [known gap](conformance/translation-matrix.md#known-gaps) |
| Chat Completions → Vertex Gemini | Translation; [provider recipe](recipes/providers.md#vertex-gemini-from-chat-completions), [reference](filters/openai_chat_completions_to_vertexai_gemini.md) | Standard; token acquisition adds `gcp-adc-filter` | Authorized GCP identity, project/location/model; schema strictness, reasoning/cache accounting gaps in [matrix](conformance/translation-matrix.md) |
| Messages → Vertex Anthropic | Endpoint adaptation to `rawPredict`; [provider recipe](recipes/providers.md#vertex-anthropic-from-messages), [reference](filters/anthropic_messages_to_vertexai_anthropic.md) | `vertex-anthropic-filter`; add `gcp-adc-filter` for token acquisition | Authorized GCP identity and model access; selected cluster metadata and model/provider resolution required |
| Chat Completions → Bedrock Converse | Translation; [provider recipe](recipes/providers.md#bedrock-converse-from-chat-completions), [compatibility matrix](conformance/translation-matrix.md) | Standard includes signer | AWS key/secret/region and model access; signing alone is not translation. History, reasoning and cache gaps remain; see [audit](documentation-audit.md) for reference gap |
| Responses → native Responses backend | Native; [quickstart](quickstart.md), [passthrough example](../examples/configs/openai/responses/http-passthrough.yaml) | Standard | Caller key in quickstart; no local state, validation, or tool execution supplied by forwarding |
| Responses → gateway-managed native Responses or translated Chat | Managed, experimental; [recipe](recipes/managed-responses.md) | `openai-responses`; recipe selects `openai-all,store-sqlite` | Backend and tool credentials are separate; managed restrictions and translation gaps apply |
| Conversations → local gateway store | Managed, experimental; [state guide](recipes/managed-responses.md#conversations-and-state), [reference](filters/openai_conversations.md) | `openai-conversations,store-sqlite` or selected PostgreSQL backend | Explicit owner/auth boundary; local CRUD is distinct from forwarding upstream Conversations |

The [cloud provider runbook](recipes/providers.md) completes the existing
example configurations with credentials, endpoint/Host/TLS selection, validation,
startup, requests, and cleanup. It assumes account permissions, regional model
availability, and an authorized credential source are already established.
The [recipe index](recipes/README.md) includes the prioritized native and managed paths.

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
  [generated matrix](conformance/translation-matrix.md) and its source evidence;
  do not interpret a fixture as provider-wide qualification.

## Inference clients

**Native OpenAI API, including `/v1/responses` or `/v1/chat/completions`, to OpenAI**

- Configuration: [Quickstart](quickstart.md): route, Host header, TLS/SNI; no
  protocol translation
- Prerequisites: Caller supplies an OpenAI API key; account/model access and
  outbound HTTPS required
- Build: Standard forwarding; no managed Responses filters required

**Anthropic Messages to Anthropic**

- Configuration: [Anthropic Messages](anthropic-messages.md): native protocol
  handling
- Prerequisites: Anthropic credentials and model access; configure the upstream
  endpoint and TLS
- Build: Standard

**Anthropic Messages to a Chat Completions-compatible backend**

- Configuration: [Anthropic Messages](anthropic-messages.md): request/response
  adapter plus the SSE adapter for streaming
- Prerequisites: Backend credentials, a running compatible backend, and model
  access; review unsupported fields and lossy opt-ins
- Build: Standard

**OpenAI Chat Completions to Azure OpenAI**

- Configuration: [Azure
  adapter](filters/openai_chat_completions_to_azureai_chat_completions.md):
  API-version handling and response normalization
- Prerequisites: Azure resource endpoint, deployment, API version, and
  authorized API key or Entra identity; routing and credential injection are
  separate configuration
- Build: Adapter is standard; [Entra token injection](filters/azure_ad.md)
  requires experimental `azure-ad-filter`

**OpenAI Chat Completions to Vertex Gemini**

- Configuration: [Gemini
  adapter](filters/openai_chat_completions_to_vertexai_gemini.md): translates to
  `generateContent`
- Prerequisites: GCP project, region, enabled Vertex service, model access, and
  an identity authorized to invoke it; configure the Vertex endpoint and TLS
- Build: Adapter is standard; [GCP token injection](filters/gcp_adc.md) requires
  experimental `gcp-adc-filter`

**Anthropic Messages to Vertex Anthropic**

- Configuration: [Vertex Anthropic
  adapter](filters/anthropic_messages_to_vertexai_anthropic.md): translates to
  `rawPredict`; requires preceding model/provider selection and Vertex cluster
  metadata
- Prerequisites: GCP project/location, access to the selected Anthropic model,
  and an authorized GCP identity
- Build: Experimental `vertex-anthropic-filter`; add `gcp-adc-filter` if using
  GCP token injection

For Azure Entra injection, configure tenant ID, client ID, scope, and the
client-secret environment variable named in the [authentication
reference](filters/azure_ad.md). Only client-secret authentication is covered by
that filter. For GCP injection, use the supported metadata identity or
service-account key source described in [GCP ADC](filters/gcp_adc.md); a
`gcloud` user credential file is not supported. Authentication does not select
the upstream endpoint or grant the identity provider-side permissions.

When the gateway owns provider secrets, follow the routing prerequisites in
[credential injection](filters/credential_inject.md). Do not assume the simple
caller-key forwarding configuration in the quickstart implements that flow.

## Managed APIs and agent protocols

**Manage Responses at the gateway rather than forward them natively**

- Guide: [Responses proxy](filters/openai_responses_proxy.md), [Cargo
  features](features.md#cargo-features)
- Prerequisites and build: Experimental `openai-responses`; configure an
  inference backend. Storage-related groups additionally need a selected store
  backend and its database setup

**Run an ephemeral server-side agent loop**

- Guide: [Agentic protocols](architecture/agentic-protocols.md), [Cargo
  features](features.md#cargo-features)
- Prerequisites and build: Experimental `openai-agentic-loop`, a configured
  inference backend, and credentials/services for the selected tools

**Classify or broker MCP protocol traffic**

- Guide: [MCP](filters/mcp.md)
- Prerequisites and build: Configure reachable MCP servers and their
  authentication. MCP classification/brokering is separate from managed
  Responses tool execution

**Execute MCP tools within managed Responses**

- Guide: [MCP tool resolution](filters/openai_mcp_tool_resolve.md), [MCP
  dispatch](filters/openai_mcp_dispatch.md)
- Prerequisites and build: Experimental `openai-mcp-tools`, a store backend, and
  configured MCP servers; select the agent loop separately if needed

See [Cargo features](features.md#cargo-features) for exact build combinations
and [example configurations](../examples/README.md) for complete filter chains.
A filter reference alone is not a complete deployment configuration. Validate
provider credentials, permissions, model availability, and operation support in
your deployment before relying on a route. Independently versioned Praxis,
Praxis AI, and Policy releases are not a tested compatibility set merely because
they appear together in the documentation site.
