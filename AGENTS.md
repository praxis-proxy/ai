# AGENTS.md

This file provides guidance to coding agents when working
with code in this repository.

## Requirements

- Rust stable 1.96+
- Rust nightly (for `rustfmt`)
- CMake 3.31+ (for Pingora build via praxis dep)
- Docker 29.3.0+ or Podman (for container builds)
- Praxis core repo at `../praxis` (path dependency)

## Rust Data Ownership

Avoid cloning request, response, header, body, or SSE data
unless the copy is necessary for correctness. This is a
high-performance proxy, so prefer borrowing, moving, or
sharing data through existing ownership boundaries before
adding `.clone()`, `to_vec()`, `to_string()`, or full-body
buffering.

When a clone is necessary, keep it close to the boundary
that requires ownership and make the reason clear in the
surrounding code or test. Do not clone streaming chunks or
provider payloads just to satisfy local control flow; instead
reshape the code to move completed buffers, borrow parsed
fields, or process data incrementally.

## Quick Reference

```console
make setup-hooks    # install git pre-commit hook
make build          # workspace build
make test           # all tests
make fmt            # format with nightly rustfmt
make lint           # clippy + nightly fmt check
make doc            # rustdoc with -D warnings
make audit          # cargo audit + cargo deny check
make container      # build praxis-ai container image
```

Run a single test:

```console
cargo test -p praxis-ai-apis -- test_name
cargo test -p praxis-ai-filters -- test_name
cargo test -p praxis-ai-proxy -- test_name
```

## Architecture

**Crate dependency flow:**

```text
server (praxis-ai-proxy)
  -> filters (praxis-ai-filters)
  -> apis (praxis-ai-apis)
  -> praxis-filter (core, from ../praxis)
```

- **server** (`praxis-ai-proxy`): binary entry point,
  registers AI filters on top of core builtins,
  injects `ResponseStoreRegistry` as pipeline extension
- **apis** (`praxis-ai-apis`): provider-specific API
  types (OpenAI, Anthropic), request classification,
  response storage backends (SQLite, PostgreSQL),
  token usage extraction, SSE parsing
- **filters** (`praxis-ai-filters`): cross-cutting AI
  filter implementations (A2A, MCP, guardrails,
  inference routing, prompt enrichment, token usage
  header injection)

**Dependencies on Praxis core** (path deps to
`../praxis`): `praxis-filter` for `HttpFilter` trait,
pipeline, registry; `praxis-core` for config types;
`praxis-protocol` for HTTP/TCP adapters;
`praxis-tls` for TLS.

## Conventions

Follows the same conventions as
[praxis core](https://github.com/praxis-proxy/praxis).
See `docs/developing/conventions.md` for the full
coding style guide, including the
[PR review process](docs/developing/conventions.md#pr-review-process)
for handling `praxis-bot` automated review comments.

## Git Workflow

- **Never amend commits** — when addressing PR
  review feedback, add new fixup commits on top
  of the branch.
- PRs are merged with **squash and merge**, so the
  final commit on the target branch is always clean
  regardless of intermediate fixup commits.

## Test Requirements

New capabilities require:

1. Unit tests
2. Integration tests
3. Example config in `examples/configs/`
4. Functional integration test for the example config

## Adding a Filter

1. Create module under `filters/src/` or `apis/src/`
2. Implement `HttpFilter` from `praxis-filter`
3. Register in `server/src/lib.rs` via
   `register_filters!` macro
4. Add unit tests and doctests
5. Add example config in `examples/configs/`

## Key Patterns

- **Classify → route → branch**: classifier filters
  promote facts to internal headers
  (`x-praxis-ai-*`) and the router matches those
  headers to select clusters.
- **Do not buffer full streaming responses**:
  streaming and SSE filters should use
  `BodyMode::Stream` and process chunks
  incrementally.
- **Validate only proxy-needed fields**: let the
  backend handle parameter ranges, model
  availability, and role ordering.

## Filter Organization

- `apis/src/anthropic/` — Anthropic Messages API
- `apis/src/openai/` — OpenAI Responses, Conversations,
  SSE, model rewrite, store, rehydrate, validate, proxy
- `apis/src/classifier/` — AI request format detection
- `apis/src/store/` — ResponseStore trait, SQLite/Postgres
- `apis/src/token_usage/` — Multi-provider token counting
- `filters/src/agentic/` — A2A, MCP protocol filters
- `filters/src/guardrails/` — AI content safety (NeMo)
- `filters/src/inference/` — Model-to-header routing
- `filters/src/prompt_enrich/` — Prompt enrichment
- `filters/src/token_usage_headers.rs` — Token headers

## Dynamic Config Reload

Praxis swaps filter pipelines at runtime without
restarting. The AI server inherits this from
praxis-protocol. The `ResponseStoreRegistry` is
injected as a `PipelineExtension` and created fresh
per pipeline build.

## Pingora Boundary

See praxis core documentation. Pingora handles:
request smuggling prevention, H2 backpressure,
connection pool safety, HTTP/1.1 upgrade detection.

## CI Workflows

CI workflows that post PR comments must use the
`praxis-bot-app` GitHub App token (via
`actions/create-github-app-token`), not the default
`github.token`.
