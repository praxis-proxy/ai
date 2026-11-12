# OpenAI Conformance

This directory tracks Praxis AI conformance against selected OpenAI API
surfaces. The current scope is Conversations and Responses.

This directory also vendors the upstream Anthropic Messages API OpenAPI
document at `specs/anthropic-spec.json`. It is read by the automated PR review
prompt (`.github/prompts/automated-review.md`) when a change touches Anthropic
surfaces, and a registry drift check
(`cargo xtask check-anthropic-messages-registry`, wired into `make lint`)
verifies that the Anthropic Messages operation registry's methods, paths, and
operation IDs match the pinned document. There is no `oasdiff` structural
comparison or capability projection for Anthropic.
See [Anthropic Messages Reference Spec](#anthropic-messages-reference-spec).

## Contract Sources

Conformance compares two independent truths:

- OpenAI's complete pinned contract is checked in at
  `specs/openai-openapi.yaml`, with its immutable commit and SHA-256 recorded
  in `specs/openai-openapi-source.json`. Every area is projected semantically
  from this one source during the conformance run.
- Praxis behavior is declared by the Conversations and Responses operation
  registries. Each operation has one route, handling mode, and, when Praxis
  owns the payload, request and response contract bindings.

The registry generates runtime matching, capability claims, and the local
implementation OpenAPI document. The report is derived output; editing it
does not add support.

## How It Fits Together

```text
Pinned complete OpenAI spec -> semantic area projections ----------------+
                                                                         |
Area registry + Rust operation registry + shared contract types          |
  +-> runtime matcher -> area handlers -> focused runtime tests ----------+-> generated report
  +-> per-area implementation OpenAPI ------------> pinned oasdiff -------+
  +-> capability claims --------------------------------------------------+
```

The OpenAI pin is the authority for the target contract. The Rust registry
and its shared contract types are the authority for what Praxis claims and
implements. Conformance is the reproducible comparison between those two
sources, backed by tests of the real runtime path.

### Verified Conversations request behavior

The Conversations create and update request-body edge cases were checked
against `api.openai.com` on 2026-07-27.

| Request | Result |
| --- | --- |
| Create with no body or `{}` | `200` |
| Create with null `metadata` and `items` | `200` |
| Update with no body or `{}` | `400 missing_required_parameter` for `metadata` |
| Update with null `metadata` | `400 invalid_type` for `metadata` |
| Update with object `metadata` | `200` |

The pinned and current upstream OpenAPI documents omit `requestBody.required`
for update and mark `metadata` nullable even though the live endpoint requires
the body and rejects null metadata. The implementation document follows the
verified runtime contract, and both discrepancies are recorded as explicit
`upstream_spec_exceptions` on `POST /conversations/{conversation_id}` rather
than declaring behavior Praxis does not implement.

## Code Map

- `apis/src/openai/conversations/routes.rs` declares each operation once and
  derives runtime matching, handling modes, capability claims, and contract
  bindings.
- `apis/src/openai/conversations/contracts.rs` contains the request and
  response types shared by runtime behavior and generated schemas.
- `apis/src/openai/conversations/openapi.rs` builds the implementation OpenAPI
  document from the registry.
- `apis/src/openai/operation.rs` provides handling modes and reusable
  parameter, media-type, request, response, and OpenAPI generation metadata.
- `apis/src/openai/conversations/filter.rs` and `handlers.rs` execute the
  matched operation.
- `xtask/src/openai_conformance/area.rs` registers each area selector,
  implementation adapter, support claims, and focused runtime suite.
- `xtask/src/openai_conformance/` verifies the full pinned reference, derives
  semantic area projections, runs `oasdiff` and runtime checks, and writes the
  report.
- `docs/conformance/specs/` contains the exact complete upstream document and
  its immutable source manifest.
- `docs/conformance/openai-conformance-report.json` is the generated snapshot
  of coverage, owned-contract drift, and runtime verification.
- `.github/workflows/openai-conformance.yaml` verifies the reference and
  rejects stale generated artifacts in CI.

## Handling Modes

Every operation has one proxy-boundary mode:

- `passthrough`: forward the operation without payload mutation.
- `inspect`: read selected fields while preserving the forwarded payload.
- `transform`: map between input and output contracts owned by Praxis.
- `local`: terminate the request and produce the response in Praxis.

Only `transform` and `local` operations enter owned-contract OpenAPI
comparison. All eight current Conversations operations are `local`.

### Rate-limit response ownership

The pinned OpenAI reference declares a `429` response on all eight
Conversations operations. Its `TooManyRequests` contract includes an optional
integer `Retry-After` header (minimum 1) and an `ErrorResponse` JSON body.

The Conversations filter owns an optional `rate_limit` admission path that
reuses the core request limiter. It runs after classification and trusted owner
resolution, before store access, and covers all eight local operations. Its
adapter adds the OpenAI JSON error envelope while preserving the core limiter's
`Retry-After` and rate-limit headers. The operation registry declares the shared
429 body and optional retry header from runtime contract types.

`mode: global` shares a budget across callers of that filter instance;
`mode: per_ip` groups requests by their TCP peer address, independently of the
Conversations store's owner isolation. The core configuration controls refill,
burst, IPv6 grouping, and shadow mode. Omitting the block disables admission.
Buckets are local to the pipeline instance and reset when it is rebuilt. This is
request admission, not a distributed or owner-specific quota guarantee.

The focused conformance suite requires a runtime rejection test for every
operation, validated against the pinned error schema. SDK tests exercise all
eight HTTP paths and verify that rejected mutations preserve existing state.
The default example demonstrates admission and has a functional rejection test.
The fatal proxy error formatter remains a separate response path.

### Other Conversations errors

The operation registry also declares the current pin's 16 non-admission error
responses: five `400`, seven `404`, and four `500` responses. Validation, missing
owner-scoped resources, and storage failures provide their runtime paths. Every
local error includes `message`, `type`, `code`, and `param`; the last two may be
null. Storage failures expose a generic message without backend details.
The shared schema includes optional `misalignment.review_target`, with its
nullable opaque-target constraints; locally generated errors omit misalignment.

The required runtime suite checks every declared status against the pinned
schema and verifies foreign-owner resources remain inaccessible. SDK coverage
checks all eight operations, unchanged state after rejected mutations, and real
SQL transaction rollback using injected write failures. Conversation creation
and initial items commit atomically; item deletion uses the metadata snapshot
read before mutation so a later read cannot report a failed committed deletion.
The example's functional suite exercises all eight admission paths and the
client-error envelopes. Existing evidence-backed Conversations exceptions
remain unchanged (11 entries); exact conformance is measured after them.

### Authentication ownership

The `openai_conversations` filter terminates matching requests but does not
authenticate clients. Authentication and authorization are deployment-owned:
operators must place the required security filter or trusted ingress boundary
before Conversations. The generated implementation document therefore does
not claim OpenAI bearer authentication.

Area projections exclude only inherited global security when an area declares
it deployment-owned. Operation-specific security remains in scope. The report
records this choice as `inherited_security: "deployment"`, so the comparison
boundary is explicit without hiding operation contracts.

## Tooling

The report requires exactly `oasdiff` 1.23.0. Install the pinned release with:

```console
go install github.com/oasdiff/oasdiff@v1.23.0
```

This release requires Go 1.26 or newer when installed from source.

The task rejects other versions, disables external references, flattens
path-level parameters, and lets `oasdiff` exclude documentation fields in
their OpenAPI context.

Generate the report:

```console
cargo xtask openai-conformance \
  --output-json docs/conformance/openai-conformance-report.json \
  --output-badges-dir docs/conformance/badges
```

The default selects all registered areas. Use `--area conversations` for a
focused run. Repeat `--area` to select multiple registered areas; `--area all`
is equivalent to the default.

This command also runs the focused runtime contract tests recorded in the
report. A failing or missing declared test makes the command fail after the
JSON result has been written.

Conversation input and output item unions are derived from the same pinned
document into the runtime artifact:

```console
cargo xtask openai-conversation-item-contracts
cargo xtask openai-conversation-item-contracts --check
```

When Conversations is selected, conformance generation performs the check
before comparing schemas, so the runtime validator and generated implementation
document cannot silently drift from the pinned item union.

## Reference Refresh

`cargo xtask openai-conformance` does not fetch upstream. It always uses the
checked-in complete OpenAI reference at
`docs/conformance/specs/openai-openapi.yaml` and verifies its provenance against
`docs/conformance/specs/openai-openapi-source.json` before comparison, then
creates area projections in memory. The report retains the reference revision
and SHA-256. There is no alternate reference-path option; `--openai-spec` is
rejected.
`--implementation-spec` remains available to override the local implementation
document for exactly one selected area, without changing the pinned reference.

Reference refresh is an independent command. To pin an intentional upstream
update and replace the complete spec:

```console
cargo xtask openai-conformance-reference --revision <40-character-commit>
```

To fetch the checked-in immutable revision, verify its source SHA-256, and
byte-compare it with the complete vendored document:

```console
cargo xtask openai-conformance-reference --check
```

Area projection keeps selected path items, path-level parameters, and the
recursive closure of local component references. Inherited global security
and referenced security schemes are included by default but excluded when
an area declares authentication as deployment-owned via
`without_inherited_security()`. Remote and non-component `$ref` values are
rejected.

## Reading the Report

Report schema version 3 records the complete reference once, gives every area
its own projection digest, implementation source, and `inherited_security`
ownership (`"owned"` when the area includes global security in its contract,
`"deployment"` when authentication is declared as deployment-owned and
excluded from comparison), and keeps three independent dimensions:

1. `capability_coverage` groups selected operations by handling mode and shows
   missing or stale support claims.
2. `owned_contract_conformance` compares `local` and `transform` contracts
   with pinned `oasdiff`. Operation drift and inherited area drift, such as
   global authentication, are reported separately.
3. `runtime_verification` records the focused commands, exact test evidence,
   and their result from this report run.

Do not collapse these dimensions into one percentage. A route can be covered
while its owned schema drifts, and exact operation schemas do not erase a
global security mismatch.

Use `owned_contract_conformance.fixes_required` as the actionable contract
backlog.

### Responses state verification

`responses_state_verification` reports focused runtime evidence for the native
Responses pipeline with local state filters configured. Its `groups` object has
stable keys used by the README badges. `--output-badges-dir` generates local SVG
assets from the same results, so badges render on branches and local README
previews without fetching the report from `main`. CI regenerates and checks
these assets alongside the report:


| Group | Checked behavior |
| --- | --- |
| `persistence` | Default persistence admission, `store:false` preventing buffered and streamed writes, buffered and streamed persistence, and persistence through the classified pipeline. |
| `rehydration` | History from `previous_response_id` and both `conversation` forms, missing or unfinished response rejection, owner isolation, and previous-response ID restoration in JSON and chunked SSE. |
| `lifecycle` | Retrieval and missing-resource errors, owner isolation, input-item cursor pagination, deletion, and ordered stored SSE replay with cursor and owner checks. |

Each group includes its scope, executed command, and required test evidence.
A group is `passed` only if it has declared checks and the complete Responses
runtime command succeeds with every required success sentinel present. Any
suite failure or missing sentinel marks all declared groups `failed`;
`not_verified` means the Responses suite was not run or the group has no checks.
The command fails on runtime verification failure, including missing tests.

These are test results, not percentages of the complete OpenAI contract.
The state checks exercise local SQLite storage and controlled test fixtures;
they do not certify every storage backend or a live vLLM deployment. The OpenAI
SDK suite separately covers persistence, continuation, and lifecycle behavior;
its results are not included in these badge statuses. Generation behavior belongs
to the backend, and Chat Completions translation is a separate profile. Owned
OpenAPI drift remains visible independently in `owned_contract_conformance`.
See the [state conformance boundary](openai-responses-state-conformance.md).

### Upstream specification exceptions

An area may remove an exact drift fingerprint from the actionable backlog only
when authoritative or live evidence shows the pinned upstream schema is
incomplete. Every exception records its operation, drift channel, exact detail,
rationale, and evidence under `upstream_spec_exceptions`. Generation fails when
a declared fingerprint no longer appears, so an upstream correction cannot be
masked by a stale exception. Unrelated drift on the same operation remains
active.

OpenAI Conversations responses were checked against `api.openai.com` on
2026-07-27. Omitted and explicitly null create metadata returned `{}`, while a
populated string map round-tripped unchanged. The refreshed upstream spec now
defines the string-map response schema. Four exceptions remain for Praxis's
metadata-key constraint, which that response schema omits.

The same probe covered the update request body. An absent body returned `400
missing_required_parameter` and null metadata returned `400 invalid_type`, so
the pinned upstream is incomplete: it omits `requestBody.required` and marks
`metadata` nullable. Praxis follows the verified runtime contract — a required
body with non-null string-map metadata — and records the seven resulting
`POST /conversations/{conversation_id}` request drift fingerprints as explicit
exceptions.

## CI Enforcement

CI always runs strict capability and owned-contract conformance after the
reference, runtime checks, and generated-artifact verification have passed.
Strict conformance records a failed step while known failures remain.

On pull requests, the gate compares normalized failure fingerprints with the
base branch's checked-in report. Existing failures and fixes need no label.
New missing operations or drift details fail the workflow unless a reviewer
applies `conformance-failure-acknowledged`. Adding or removing that label
reruns the workflow.

The label does not skip conformance. It acknowledges the exact new failure set
written to the generated report. Reference integrity, runtime verification,
and stale generated artifacts remain hard failures and cannot be acknowledged.

Merge queue and `main` runs do not have pull-request labels. They regenerate
the report and acknowledge strict failure only when it is byte-identical to
the checked-in report reviewed on the pull request. Any combined merge-queue
change that alters conformance therefore fails until its report is updated and
reviewed.

Follow-up fixes remove their fingerprints from the report without needing the
label. When no failures remain, the strict gate passes and no acknowledgement
step runs.

## Change Workflow

When OpenAI-compatible behavior changes:

1. Update the operation registry and the real runtime behavior.
2. Update the shared request or response contract types when Praxis owns them.
3. Add mode-appropriate runtime tests.
4. Run `cargo xtask openai-conformance-reference --check`.
5. Regenerate `openai-conformance-report.json`.
6. Commit code, the generated report, and reference artifacts only when the
   upstream pin was intentionally changed.

Future API areas should add one data-driven path selector, operation registry
adapter, generated owned-contract document, and focused runtime suite. They
reuse the global OpenAI pin, semantic projector, shared operation metadata,
and report pipeline rather than adding another reference artifact, scanning
Rust source, or maintaining a handwritten implementation specification.

## Anthropic Messages Reference Spec

`specs/anthropic-spec.json` is the vendored Anthropic API OpenAPI document,
used as the authoritative contract when reviewing changes to the Anthropic
Messages surfaces under `apis/src/anthropic/`. It serves two purposes: the
registry drift check (`cargo xtask check-anthropic-messages-registry`, wired
into `make lint`) verifies that every registered operation's method, path, and
operation ID match the pinned spec, and the automated PR review prompt
cross-references request and response schemas, field names, types, enums, and
optionality without a network fetch. Unlike the OpenAI spec, there is no
`oasdiff` structural comparison, capability projection, or runtime suite wired
to it.

### Source and provenance

`specs/anthropic-spec-source.json` pins the exact upstream origin:

- Repository `anthropics/anthropic-sdk-python`, at an immutable 40-character
  commit `revision`.
- Path `scripts/mock-spec.json.gz` — the mock-server OpenAPI document bundled
  in that repository. `encoding` is `gzip` and `source_sha256` is the digest of
  that upstream gzip.
- `anthropic-spec.json` is the verbatim gunzip of that pinned gzip, and
  `vendored_sha256` is the digest of the decompressed JSON checked in here.

Verify the vendored file against the manifest:

```console
shasum -a 256 docs/conformance/specs/anthropic-spec.json
# must equal vendored_sha256 in anthropic-spec-source.json
```

### Scope

The spec is the complete upstream Anthropic API document. Today the only
Anthropic path Praxis handles is `/v1/messages` and its subpaths
(`/v1/messages`, `/v1/messages/count_tokens`, `/v1/messages/batches`), so that
is where conformance review applies. The other stable surfaces in the spec
(`/v1/complete`, `/v1/models`, `/v1/files`) are in scope only if a change adds
Anthropic handling for them. Paths carrying a `?beta=true` query and the
platform or console surfaces (agents, deployments, environments, memory stores,
organizations, sessions, tunnels, vaults, skills, user profiles) are always out
of scope — Praxis does not implement them, so their absence in Praxis is never
a finding.

### Relationship to the Anthropic Messages operation registry

The Anthropic Messages operation registry in `apis/src/anthropic/routes.rs`
transcribes its methods, paths, operation IDs, and request-body shapes from this
pinned document, covering the non-beta `/v1/messages` surface named above.
`cargo xtask check-anthropic-messages-registry` (wired into `make lint`)
compares every registered operation's method, path, and operation ID against
the pinned spec and fails on divergence, so a spec refresh will flag a registry
that has fallen behind.

### Refresh

```console
curl -fsSL \
  https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/<revision>/scripts/mock-spec.json.gz \
  | gunzip > docs/conformance/specs/anthropic-spec.json
```

Then update `revision`, `source_sha256`, and `vendored_sha256` in
`specs/anthropic-spec-source.json`.
