# Outbound callout security

Praxis AI treats an outbound target, its resolved socket addresses, and the
credentials attached to the request as one trust decision. Operator-configured
HTTP targets are public-only by default. Each request resolves its target once,
checks every returned address immediately before connecting, and gives those
same addresses to the transport. A single private, loopback, link-local, or
otherwise non-public result rejects the whole callout.

Private targets require a filter-specific, explicit opt-in. HTTP proxies and
redirect following are disabled for direct callouts. URL userinfo is rejected,
and forwarded or configured credentials remain bound to the validated target
origin.

Private-target opt-ins allow loopback, private, shared-address-space,
link-local, and other special-use destinations. They never allow recognized
cloud metadata endpoints, unspecified addresses, or multicast addresses.
IPv4-mapped IPv6 and well-known-prefix NAT64 forms receive the same verdict as
their embedded IPv4 address. GCP ADC is the only exception: its dedicated,
non-configurable metadata policy accepts recognized metadata addresses (and
loopback test stubs) only after its authority has been independently restricted.

## Callout inventory

`No-follow` means a redirect response is returned to the caller and its
`Location` is never requested.

| Callout | Target source | Private opt-in | Redirect policy | Authentication mode |
| --- | --- | --- | --- | --- |
| `openai_web_search_dispatch`, `anthropic_web_search` | Provider default or configured `base_url`, executed through the `outbound_chain` filtered-subrequest executor | `insecure_options.allow_private_upstreams` (executor-gated on the bound outbound chain) | No-follow | Configured provider API key; validated origin only |
| `openai_file_resolve` Files API | Configured `files_api_url` via `outbound_chain` | `allow_private_upstreams` | No-follow | Headers named by `forward_headers` plus `outbound_chain` mutations |
| `openai_file_resolve` `file_url` fetch | Request-derived URL via `SubRequestClient` | Exact `allowed_file_url_origins` | No-follow | Anonymous; no downstream headers |
| `openai_file_search_dispatch` | Configured `vector_store_url` | `allow_private_url` | No-follow | Only headers named by `forward_headers` |
| `openai_responses_compact` | Configured `inference_url` | `allow_private_inference_url` | No-follow | Anonymous; no downstream or cluster headers |
| `ai_guardrails` with NeMo | Configured `endpoint` | Global `allow_private_upstreams` | No-follow | Configured `outbound_chain`; no downstream headers by default |
| `http_callout` | Configured `target.url` | `allow_private_addresses` | No-follow | Configured static headers plus allowed `forward_headers` |
| `external_metering` balance check and usage report | Configured `metering_url` via `SubRequestClient` | `allow_private_endpoint` | No-follow | Anonymous; no downstream headers |
| MCP client | Request-derived server URL or configured connector | `allow_loopback` for loopback only | No-follow | Sanitized request-provided MCP authorization/headers |
| `azure_ad` token fetch | Configured authority plus tenant via `SubRequestClient` | `allow_private_authority` | No-follow | Client secret in the token POST body |
| `gcp_adc` metadata fetch | Protocol-owned metadata endpoint via `SubRequestClient` | Intrinsic to metadata mode | No-follow | `Metadata-Flavor` protocol header; returned token is not forwarded back to metadata |

The request-derived `file_url` and MCP transports retain stricter policies:
they resolve once per attempt, validate every address against `AddressPolicy`,
do not follow redirects, and allow private access only through their narrow
origin/loopback controls. GCP metadata intentionally allows only Google's two
protocol-owned metadata addresses via `AddressPolicy::AllowGoogleMetadata`.

`external_metering` calls out twice per request, so it resolves its configured
host through the upstream DNS cache rather than on every call. Every cached
address is still validated against `AddressPolicy` before connecting.

Upstream cluster connections are not direct callouts. They use the core Praxis
endpoint and TLS policy instead of these filter-level controls.

## Adding a callout

A new filter that opens an outbound HTTP connection must:

1. Classify the target as operator-configured, request-derived, or
   protocol-owned.
2. Reuse the shared target/address policy, or document why a stricter dedicated
   policy is required.
3. Default to public addresses and expose a narrowly named private-address
   opt-in only when the use case requires it.
4. Resolve once per connection attempt, reject the complete result set if any
   address violates policy, and connect only to that validated set.
5. Disable ambient proxies and redirects unless their security behavior is
   explicitly designed and tested.
6. Reject URL userinfo and bind every credential or forwarded header to the
   validated origin.
7. Document the authentication mode and cover loopback/private/link-local,
   mixed DNS answers, redirects, userinfo, and credential non-disclosure in
   tests.

## TLS backend

Production outbound callouts use `SubRequestClient`, which inherits TLS from
the process-wide Pingora/praxis-tls provider installed at startup
(`praxis_tls::provider::install`). No per-callout or per-filter TLS backend
selection is needed — all callouts share the same validated TLS stack as
upstream cluster connections.

On RHEL/OpenShift FIPS builds, the system OpenSSL provider enforces approved
cipher suites and algorithms for every outbound connection automatically. The
FIPS feature set includes `openai-file-resolve-filter` and `aws-sigv4-filter`;
`openai-mcp-tools` remains excluded because its `store` dependency pulls sha2
through sqlx, and the response stores and policy engine are excluded for the
same reason.

`reqwest` is a dev/test-only dependency. Test TLS infrastructure
(`tests/utils/src/net/tls.rs`) uses rustls for mock servers and certificate
generation, independent of the production TLS backend.
