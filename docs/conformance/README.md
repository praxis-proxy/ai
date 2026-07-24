# OpenAI Conformance

This directory tracks Praxis AI conformance against selected OpenAI API
surfaces. The current scope is Conversations only.

## Contract Sources

Conformance compares two independent truths:

- OpenAI's complete pinned contract is checked in at
  `specs/openai-openapi.yaml`, with its immutable commit and SHA-256 recorded
  in `specs/openai-openapi-source.json`. Every area is projected semantically
  from this one source during the conformance run.
- Praxis behavior is declared by the Conversations operation registry. Each
  operation has one route, handling mode, and, when Praxis owns the payload,
  request and response contract bindings.

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
  --output-json docs/conformance/openai-conformance-report.json
```

The default selects all registered areas. Use `--area conversations` for a
focused run. Repeat `--area` to select multiple registered areas; `--area all`
is equivalent to the default.

This command also runs the focused runtime contract tests recorded in the
report. A failing or missing declared test makes the command fail after the
JSON result has been written.

## Reference Refresh

Normal conformance runs do not fetch upstream. They read the complete vendored
spec and create area projections in memory. To pin an intentional upstream
update and replace the complete spec:

```console
cargo xtask openai-conformance-reference --revision <40-character-commit>
```

To fetch the checked-in immutable revision, verify its source SHA-256, and
byte-compare it with the complete vendored document:

```console
cargo xtask openai-conformance-reference --check
```

Area projection keeps selected path items, inherited security, path-level
parameters, referenced security schemes, and the recursive closure of local
component references. Remote and non-component `$ref` values are rejected.

## Reading the Report

Report schema version 3 records the complete reference once, gives every area
its own projection digest and implementation source, and keeps three
independent dimensions:

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
