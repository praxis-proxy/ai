# OpenAI Conformance

This directory tracks Praxis AI conformance against selected OpenAI API
surfaces: Conversations, Responses and Files. Use it when you change
OpenAI-compatible behavior and need to regenerate the report and pass the CI
gate.

See the [Responses support and qualification table](openai-responses-state-conformance.md#responses-support-and-qualification) for state parameters, SDK versions, and the native and translated vLLM test lanes.

For verified CLI edits, permission modes and compaction outcomes by model and API route, see [Coding harness compatibility](coding-harness-compatibility.md).

For the external OpenResponses runtime suite, see
[full acceptance](openresponses-acceptance.md) and the separately scoped
[translation coverage](openresponses-translation.md). The directory also
vendors the Anthropic Messages OpenAPI document; see
[Anthropic Messages Reference Spec](#anthropic-messages-reference-spec).

## Contract Sources

Conformance compares two independent truths:

- OpenAI's complete pinned contract is checked in at
  `specs/openai-openapi.yaml`, with its immutable commit and SHA-256 recorded
  in `specs/openai-openapi-source.json`. Every area is projected semantically
  from this one source during the conformance run.
- Praxis behavior comes from the Conversations and Responses operation
  registries, which declare each operation once: its route, handling mode,
  and, when Praxis owns the payload, request and response contract bindings.

The registry generates runtime matching, capability claims, and the local
implementation OpenAPI document. The report is derived output; editing it
does not add support.

Files is a Praxis-owned contract on the same pin. Its source chain is the
pinned full spec projected to the `/files` scope, then the `files_operations!`
registry (operation identity and declared contract), the DTOs in
`files/contracts.rs` that the registry references, and the generated
`files_openapi_json()` document. No trimmed or hand-maintained copy of the
Files spec exists.

```text
Pinned complete OpenAI spec -> semantic area projections ----------------+
                                                                         |
Area registry + Rust operation registry + shared contract types          |
  +-> runtime matcher -> area handlers -> focused runtime tests ----------+-> generated report
  +-> per-area implementation OpenAPI ------------> pinned oasdiff -------+
  +-> capability claims --------------------------------------------------+
```

## Code Map

- `apis/src/openai/conversations/routes.rs` declares each operation once and
  derives runtime matching, handling modes, capability claims, and contract
  bindings.
- `apis/src/openai/conversations/contracts.rs` contains the request and
  response types shared by runtime behavior and generated schemas.
- `apis/src/openai/conversations/openapi.rs` builds the implementation OpenAPI
  document from the registry.
- `apis/src/openai/conversations/filter.rs` and `handlers.rs` execute the
  matched operation.
- `apis/src/openai/files/routes/` (`mod.rs`), `contracts.rs`, `openapi.rs`, and
  `tests.rs` play the same roles for Files: the `files_operations!` registry,
  the wire DTOs, the generated implementation document, and the
  `conformance_files_` runtime checks.
- `apis/src/openai/operation.rs` provides handling modes and reusable
  parameter, media-type, request, response, and OpenAPI generation metadata.
- `xtask/src/openai_conformance/area.rs` registers each area selector,
  implementation adapter, support claims, and focused runtime suite.
- `xtask/src/openai_conformance/` verifies the full pinned reference, derives
  semantic area projections, runs `oasdiff` and runtime checks, and writes the
  report. `files_registry.rs` there implements
  `cargo xtask check-files-registry`.
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

Operations with an owned mode (`transform` or `local`) or a declared contract
enter owned-contract OpenAPI comparison. The generator emits every operation
that declares a contract, so the comparison covers exactly what is generated.
All eight current Conversations operations are `local`.

### Files contract

- **Handling mode versus contract.** All five Files operations are
  `passthrough`: Praxis forwards them without mutation. Each still declares a
  contract, and generation and oasdiff comparison follow that contract, not the
  handling mode.
- **Schema deviations from the pin**, recorded as `FILES_CONTRACT_EXCEPTIONS`
  in `xtask/src/openai_conformance/area.rs`: `evals` is added to
  `OpenAIFile.purpose` (it is an accepted upload purpose), and the generated
  `OpenAIFile` states `type: object` explicitly.
- **Serialization choices within the pinned schema** (not exceptions):
  `status` is always present (`processed`); `expires_at` and `status_details`
  are omitted when absent; an empty list has `first_id` and `last_id` set to
  `""`.
- **Error responses.** All pinned error responses (400/404/429/500, the shared
  `ErrorResponse` envelope, no `Retry-After` on 429) are declared and validated
  through DTO instances, like the success schemas. While operations are
  `passthrough`, upstream produces them and Praxis forwards them unchanged. The
  registry test blocks switching an operation to `local` until
  handler-produced rejection tests exist for every declared status.
- **Authentication.** Deployment-owned, as for Conversations. The Files scope
  uses `without_inherited_security()` so the pinned global bearer requirement
  is not compared against the passthrough routes, which do not authenticate.

### Rate-limit response ownership

The pinned reference declares a `429` on all eight Conversations operations:
`TooManyRequests` with an optional integer `Retry-After` header (minimum 1)
and an `ErrorResponse` JSON body.

The Conversations filter's optional `rate_limit` admission produces that
response for all eight local operations. It reuses the core request limiter,
runs after classification and trusted owner resolution and before store
access, and wraps the limiter's `Retry-After` and rate-limit headers in the
OpenAI JSON error envelope. The operation registry declares the shared 429
body and optional retry header from runtime contract types. This producer rule
applies to `local` operations; `passthrough` contracts (Files) describe the
forwarded wire contract. For `mode: global`, `mode: per_ip`, scope, and the
other settings, see the [`openai_conversations` filter](../filters/agents/openai_conversations.md).
This is request admission, not a distributed or owner-specific quota
guarantee.

The focused conformance suite requires a runtime rejection test for every
operation, validated against the pinned error schema. SDK tests exercise all
eight HTTP paths and verify that rejected mutations preserve existing state.
The default example demonstrates admission and has a functional rejection
test. The fatal proxy error formatter is a separate response path.

### Other Conversations errors

The registry also declares the pin's 16 non-admission error responses: five
`400`, seven `404`, and four `500`. Validation, missing owner-scoped
resources, and storage failures produce them at runtime. Every local error
includes `message`, `type`, `code`, and `param`; the last two may be null.
Storage failures expose a generic message without backend details. The shared
schema includes optional `misalignment.review_target`, with its nullable
opaque-target constraints; locally generated errors omit misalignment.

The required runtime suite checks every declared status against the pinned
schema and verifies that foreign-owner resources stay inaccessible. SDK
coverage checks all eight operations, unchanged state after rejected
mutations, and real SQL transaction rollback using injected write failures.
Conversation creation and initial items commit atomically. Item deletion uses
the metadata snapshot read before mutation, so a later read cannot report a
failed committed deletion. The example's functional suite exercises all eight
admission paths and the client-error envelopes.

### Authentication ownership

The `openai_conversations` filter terminates matching requests but does not
authenticate clients. Operators must place the required security filter or a
trusted ingress boundary before Conversations, so the generated implementation
document does not claim OpenAI bearer authentication.

When an area declares authentication deployment-owned, its projection drops
only inherited global security; operation-specific security stays in scope.
The report records this as `inherited_security: "deployment"`.

## Tooling

The report requires exactly `oasdiff` 1.23.0 (Go 1.26 or newer to install
from source):

```console
go install github.com/oasdiff/oasdiff@v1.23.0
```

The task rejects other versions, disables external references, flattens
path-level parameters, and lets `oasdiff` exclude documentation fields in
their OpenAPI context.

Generate the report:

```console
cargo xtask openai-conformance \
  --output-json docs/conformance/openai-conformance-report.json \
  --output-badges-dir docs/conformance/badges
```

All registered areas run by default (`--area all`). Use `--area conversations`
for a focused run, or repeat `--area` to select several. The command also runs
the focused runtime contract tests recorded in the report; a failing or
missing declared test fails the command after the JSON result is written.

Conversation input and output item unions are derived from the same pinned
document into the runtime artifact:

```console
cargo xtask openai-conversation-item-contracts
cargo xtask openai-conversation-item-contracts --check
```

When Conversations is selected, conformance generation runs this check before
comparing schemas, so the runtime validator and the generated implementation
document cannot drift from the pinned item union.

Check Files operation identity three ways (pinned specification, runtime
registry, generated document):

```console
cargo xtask check-files-registry
```

It compares `(method, path, operationId)`, fails if any source disagrees, and
runs in `make lint`; schemas are left to oasdiff. The Files report area also
runs the `conformance_files_` tests, which cover route dispatch, success
payloads against the generated response schemas, and a sensitivity check that
the schema validation rejects an invalid payload. Each test prints a
`PRAXIS_CONFORMANCE_OK files <check>` sentinel that the report requires as
evidence; a missing test or sentinel fails the report run.

## Reference Refresh

`cargo xtask openai-conformance` never fetches upstream. It uses the
checked-in `docs/conformance/specs/openai-openapi.yaml`, verifies its
provenance against `docs/conformance/specs/openai-openapi-source.json`, and
builds area projections in memory. The report keeps the reference revision
and SHA-256. `--openai-spec` is rejected; there is no alternate reference
path. `--implementation-spec` overrides the local implementation document for
exactly one selected area, without changing the pinned reference.

To pin an intentional upstream update and replace the complete spec:

```console
cargo xtask openai-conformance-reference --revision <40-character-commit>
```

To fetch the checked-in revision, verify its source SHA-256, and byte-compare
it with the vendored document:

```console
cargo xtask openai-conformance-reference --check
```

Area projection keeps selected path items, path-level parameters, and the
recursive closure of local component references. Inherited global security
and referenced security schemes are included unless the area declares
authentication deployment-owned via `without_inherited_security()`. Remote
and non-component `$ref` values are rejected.

## Reading the Report

Report schema version 3 records the complete reference once and gives every
area its own projection digest, implementation source, and
`inherited_security` value (`"owned"` when the area's contract includes global
security, `"deployment"` when authentication is deployment-owned and excluded
from comparison). It reports three independent dimensions:

1. `capability_coverage` groups selected operations by handling mode and shows
   missing or stale support claims.
2. `owned_contract_conformance` compares owned-mode and declared contracts
   with pinned `oasdiff`. Operation drift and inherited area drift, such as
   global authentication, are reported separately.
3. `runtime_verification` records the focused commands, exact test evidence,
   and their result from this report run.

Do not collapse these into one percentage. A route can be covered while its
owned schema drifts, and exact operation schemas do not erase a global
security mismatch. `owned_contract_conformance.fixes_required` is the
actionable contract backlog.

### Responses state verification

`responses_state_verification` reports focused runtime evidence for the
native Responses pipeline with local state filters configured. Its `groups`
object has stable keys used by the README badges. `--output-badges-dir`
generates local SVG badges from the same results, so they render on branches
and in local previews without fetching the report from `main`; CI regenerates
and checks them alongside the report.

| Group | Checked behavior |
| --- | --- |
| `persistence` | Default persistence admission, `store:false` preventing buffered and streamed writes, buffered and streamed persistence, and persistence through the classified pipeline. |
| `rehydration` | History from `previous_response_id` and both `conversation` forms, missing or unfinished response rejection, owner isolation, and previous-response ID restoration in JSON and chunked SSE. |
| `lifecycle` | Retrieval and missing-resource errors, owner isolation, input-item cursor pagination, deletion, and ordered stored SSE replay with cursor and owner checks. |

Each group includes its scope, executed command, and required test evidence.
A group is `passed` only if it has declared checks and the complete Responses
runtime command succeeds with every required success sentinel present. Any
suite failure or missing sentinel marks all declared groups `failed`;
`not_verified` means the Responses suite was not run or the group has no
checks. The command fails on runtime verification failure, including missing
tests.

These are test results, not percentages of the OpenAI contract. They use
local SQLite storage and controlled test fixtures, so they certify neither
every storage backend nor a live vLLM deployment. The OpenAI SDK suite covers
persistence, continuation, and lifecycle separately, and its results do not
feed these badges. Generation belongs to the backend, and Chat Completions
translation is a separate profile; see the
[state conformance boundary](openai-responses-state-conformance.md).

### Upstream specification exceptions

An area may remove an exact drift fingerprint from the actionable backlog only
when authoritative or live evidence shows the pinned upstream schema is
incomplete. Each exception records its operation, drift channel, exact
detail, rationale, and evidence under `upstream_spec_exceptions`. Generation
fails when a declared fingerprint no longer appears, so a stale exception
cannot mask an upstream correction. Unrelated drift on the same operation
stays active.

The Conversations exceptions (11 entries) come from probes of
`api.openai.com` on 2026-07-27:

| Request | Result |
| --- | --- |
| Create with no body or `{}` | `200` |
| Create with null `metadata` and `items` | `200` |
| Update with no body or `{}` | `400 missing_required_parameter` for `metadata` |
| Update with null `metadata` | `400 invalid_type` for `metadata` |
| Update with object `metadata` | `200` |

- **Metadata response.** Omitted and null create metadata returned `{}`; a
  populated string map round-tripped unchanged. The refreshed upstream spec
  now defines the string-map response schema. Four exceptions remain for
  Praxis's metadata-key constraint, which that schema omits.
- **Update request body.** The pinned and current upstream documents omit
  `requestBody.required` and mark `metadata` nullable, but the live endpoint
  requires the body and rejects null metadata. Praxis implements the verified
  contract (a required body with non-null string-map metadata) and records the
  seven resulting request drift fingerprints on
  `POST /conversations/{conversation_id}`.

Exact conformance is measured after these exceptions.

## CI Enforcement

CI always runs strict capability and owned-contract conformance after the
reference, runtime checks, and generated-artifact verification pass. Strict
conformance records a failed step while known failures remain.

On pull requests, the gate compares normalized failure fingerprints with the
base branch's checked-in report. Existing failures and fixes need no label.
New missing operations or drift details fail the workflow unless a reviewer
applies `conformance-failure-acknowledged`; adding or removing the label
reruns the workflow. The label does not skip conformance. It acknowledges the
exact new failure set written to the generated report. Reference integrity,
runtime verification, and stale generated artifacts remain hard failures and
cannot be acknowledged.

Merge queue and `main` runs have no pull-request labels. They regenerate the
report and accept strict failure only when it is byte-identical to the
checked-in report reviewed on the pull request, so a merge-queue combination
that changes conformance fails until its report is updated and reviewed.
When no failures remain, the strict gate passes and no acknowledgement step
runs.

## Change Workflow

When OpenAI-compatible behavior changes:

1. Update the operation registry and the real runtime behavior.
2. Update the shared request or response contract types when Praxis owns them.
3. Add mode-appropriate runtime tests.
4. Run `cargo xtask openai-conformance-reference --check`.
5. Regenerate `openai-conformance-report.json`.
6. Commit code, the generated report, and reference artifacts only when the
   upstream pin was intentionally changed.

### Changing the Files contract

1. Edit the registry entry in `files/routes/mod.rs` and/or the DTOs in
   `files/contracts.rs`.
2. Run `cargo xtask check-files-registry`.
3. Run the `conformance_files_` tests.
4. Regenerate `openai-conformance-report.json`. Oasdiff drift must be fixed or
   carry an evidence-backed contract exception.

### Refreshing the upstream pin for Files

1. Run `cargo xtask openai-conformance-reference --revision <sha>`.
2. Run `cargo xtask check-files-registry`; it flags operation drift.
3. Regenerate the report; oasdiff shows Files schema drift.
4. Update the DTOs, and update or remove any exception that the
   stale-exception rule flags.

A new API area adds one data-driven path selector, operation registry adapter,
generated owned-contract document, and focused runtime suite. It reuses the
global OpenAI pin, semantic projector, shared operation metadata, and report
pipeline instead of adding another reference artifact, scanning Rust source,
or maintaining a handwritten implementation specification.

## Anthropic Messages Reference Spec

`specs/anthropic-spec.json` is the vendored Anthropic API OpenAPI document,
the authoritative contract for changes to the Anthropic Messages surfaces
under `apis/src/anthropic/`. It has two consumers:

- The Anthropic Messages operation registry in `apis/src/anthropic/routes.rs`
  transcribes its methods, paths, operation IDs, and request-body shapes from
  this document. `cargo xtask check-anthropic-messages-registry` (wired into
  `make lint`) fails when a registered operation's method, path, or operation
  ID diverges from it, so a spec refresh flags a registry that fell behind.
- The automated PR review prompt (`.github/prompts/automated-review.md`) reads
  it when a change touches Anthropic surfaces, to cross-reference schemas,
  field names, types, enums, and optionality without a network fetch.

Unlike the OpenAI spec, it has no `oasdiff` structural comparison, capability
projection, or runtime suite.

### Source and provenance

`specs/anthropic-spec-source.json` pins the upstream origin:

- Repository `anthropics/anthropic-sdk-python`, at an immutable 40-character
  commit `revision`.
- Path `scripts/mock-spec.json.gz`, the mock-server OpenAPI document bundled
  in that repository. `encoding` is `gzip` and `source_sha256` is the digest of
  that upstream gzip.
- `anthropic-spec.json` is the verbatim gunzip of that gzip, and
  `vendored_sha256` is the digest of the decompressed JSON checked in here.

Verify the vendored file against the manifest:

```console
shasum -a 256 docs/conformance/specs/anthropic-spec.json
# must equal vendored_sha256 in anthropic-spec-source.json
```

### Scope

The spec is the complete upstream Anthropic API document. Praxis handles only
`/v1/messages` and its subpaths (`/v1/messages`, `/v1/messages/count_tokens`,
`/v1/messages/batches`), so conformance review applies there. The other
stable surfaces (`/v1/complete`, `/v1/models`, `/v1/files`) come into scope
only if a change adds Anthropic handling for them. Paths with a `?beta=true`
query and the platform or console surfaces (agents, deployments,
environments, memory stores, organizations, sessions, tunnels, vaults,
skills, user profiles) are always out of scope: Praxis does not implement
them, so their absence is never a finding.

### Refresh

```console
curl -fsSL \
  https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/<revision>/scripts/mock-spec.json.gz \
  | gunzip > docs/conformance/specs/anthropic-spec.json
```

Then update `revision`, `source_sha256`, and `vendored_sha256` in
`specs/anthropic-spec-source.json`.
