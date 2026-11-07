# Red Hat Praxis AI downstream

This repository is the ET downstream of
[`praxis-proxy/ai`](https://github.com/praxis-proxy/ai) and is the source for
EnMaaS Praxis images. Runtime configuration and credentials remain in
[`redhat-et/pricetag`](https://github.com/redhat-et/pricetag).

## Purpose of this document

This is the focused EnMaaS capability handoff for the Praxis team. It lists
only the six ET-owned capabilities that still provide functionality not yet
available in the upstream Praxis stack. The goal is to decide, with the Praxis
team, whether each capability should be implemented upstream, replaced by an
existing upstream primitive, or remain an explicitly downstream integration.

The inventory is intentionally not a list of every filter currently present in
the ET image.

## EnMaaS architecture reference

See the [EnMaaS production gateway architecture and request flow](https://noyitz.github.io/ai-gateway-docs/enmaas-flow.html).
Purple nodes in the diagram identify custom ET filters or ET-specific filter
extensions; the remaining nodes represent upstream Praxis/core components,
provider services, or deployment infrastructure.

## Six remaining EnMaaS capabilities

| Capability | EnMaaS behavior | ET implementation | Upstream tracking | Long-term question |
|---|---|---|---|---|
| `api_key_auth` | Validates MaaS API keys and publishes trusted tenant identity for gateway authorization | `filters/src/api_key_auth/` | [Issue #707](https://github.com/praxis-proxy/ai/issues/707) | Can upstream provide a generic external API-key validator with configurable validation endpoint, identity mapping, caching, and fail-closed behavior? |
| `model_catalog` | Serves credential/dialect-specific `/v1/models` responses from the authoritative routing catalog | `filters/src/model_catalog/` | [Issue #1067](https://github.com/praxis-proxy/ai/issues/1067) | Can upstream own a provider-neutral catalog projection with principal/dialect filtering and reloadable configuration? |
| `reject_upgrade` | Rejects connection upgrades that would bypass body-based authorization, metering, and token accounting | `filters/src/reject_upgrade/` | [Issue #1359](https://github.com/praxis-proxy/ai/issues/1359) | Should Praxis provide a standard policy for rejecting or safely handling upgrades before body-dependent filters? |
| `token_count` provider `auto` | Selects the correct usage parser for mixed OpenAI/Anthropic traffic on shared listeners | `filters/src/token_usage/count.rs` | [Issue #1720](https://github.com/praxis-proxy/ai/issues/1720) | Can upstream expose a documented automatic dialect/parser mode with bounded, fail-safe behavior? |
| File-backed internal metering auth | Reads a Secret-mounted bearer token for private Metering calls and supports rotation without rebuilding the image | `filters/src/metering/` | [Issue #1721](https://github.com/praxis-proxy/ai/issues/1721) | Can the external metering filter support a safe file credential source and rotation semantics? |
| Metering model-policy preflight | Sends the resolved model and tenant to the Metering policy API before provider forwarding; denial becomes HTTP 403 | `filters/src/metering/` | [Issue #1719](https://github.com/praxis-proxy/ai/issues/1719) | Can upstream support an optional authorization preflight distinct from ordinary usage-accounting fail-open behavior? |

## Questions for the Praxis team

For each of the six issues above, please confirm:

1. Is the capability appropriate for upstream Praxis, or should it remain an
   ET/platform integration?
2. If upstream-appropriate, what API/configuration contract should ET target?
3. What security, reload, failure-mode, and integration-test requirements must
   be satisfied before ET can remove its implementation?
4. Is there an existing upstream filter or core capability that should replace
   the ET behavior instead?

## Current downstream sync

The ET `main` branch is synchronized with upstream through the normal downstream
workflow. Resolve sync conflicts in ET, run the required checks, and record the
resulting sync commit here before building an EnMaaS image.

## Temporary downstream backports

- `apis/src/vertex/mod.rs` (Anthropic→Vertex `rawPredict`): the deployed
  image still carries the pre-v0.8.0 module. It now strips `Accept-Encoding`
  and WARNs when a response with an armed transform is encoded or when an SSE
  chunk cannot be parsed (backport of the v0.8.0 `apis/src/vertex/anthropic`
  behavior). Delete this module at the v0.8.0 sync. The interim
  `headers.request_remove: accept-encoding` entries in the base and QoS
  Praxis configs can be reverted once the rebuilt image is deployed (avoid a
  permanent double-strip).
