# Client and provider paths

Choose the client protocol first, then the upstream and its authentication.
Praxis AI is alpha software. This index describes the current checkout; use
matching documentation and build features for a released binary. The linked
references own configuration details and translation limitations. A listed
adapter is not a guarantee that every provider operation or model is supported.

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
