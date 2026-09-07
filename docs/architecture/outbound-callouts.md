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

## Callout inventory

`No-follow` means a redirect response is returned to the caller and its
`Location` is never requested.

| Callout | Target source | Private opt-in | Redirect policy | Authentication mode |
| --- | --- | --- | --- | --- |
| `openai_web_search`, `anthropic_web_search` | Provider default or configured `base_url` | `allow_private_base_url` | No-follow | Configured provider API key; validated origin only |
| `openai_file_resolve` Files API | Configured `files_api_url` | `allow_private_files_api_url` | No-follow | Only headers named by `forward_headers` |
| `openai_file_resolve` `file_url` fetch | Request-derived URL | Exact `allowed_file_url_origins` | No-follow | Anonymous; no downstream headers |
| `openai_file_search_callout` | Configured `vector_store_url` | `allow_private_url` | No-follow | Only headers named by `forward_headers` |
| `openai_responses_compact` | Configured `inference_url` | `allow_private_inference_url` | No-follow | Anonymous; no downstream or cluster headers |
| `ai_guardrails` with NeMo | Configured `endpoint` | `allow_private_endpoint` | No-follow | Anonymous; no downstream headers |
| `http_callout` | Configured `target.url` | `allow_private_addresses` | No-follow | Configured static headers plus allowed `forward_headers` |
| MCP client | Request-derived server URL or configured connector | `allow_loopback` for loopback only | No-follow | Sanitized request-provided MCP authorization/headers |
| `azure_ad` token fetch | Configured authority plus tenant | `allow_private_authority` | No-follow | Client secret in the token POST body |
| `gcp_adc` metadata fetch | Protocol-owned metadata endpoint | Intrinsic to metadata mode | No-follow | `Metadata-Flavor` protocol header; returned token is not forwarded back to metadata |

The request-derived `file_url` and MCP transports retain stricter policies:
they pin one validated resolution set, do not follow redirects, and allow
private access only through their narrow origin/loopback controls. GCP metadata
also uses pinned resolution, no redirects, and no ambient proxy, but intentionally
allows its protocol-owned private destination.

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
