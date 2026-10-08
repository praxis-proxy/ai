# Flow visualizers

A **flow visualizer** is a single, self-contained HTML file that explains one
Praxis config as an interactive diagram: the filter pipeline, the clusters and
external services it talks to, the scenarios it serves, and — for agentic
configs — the iterative request router (IRR) tool loop, complete with
on-the-wire request/response examples and step-by-step timelines.

The canonical example documents the Responses API full-flow agentic gateway:

- Config: `examples/configs/openai/responses/full-flow-agentic.yaml`
- Visualizer: `examples/configs/openai/responses/full-flow-agentic.visualizer.html`

The HTML is **generated**, not hand-maintained. This document explains how, and
how to change it safely.

## Exploring the pages

The curated page opens on **Topology**, with separate **Filters**, **Wire
exchange**, and **Timeline** views. Select a scenario to update all four views;
switching views preserves the selection. **Play trace** opens the filter view
and follows active filters in order. The phase control opens the topology and
emphasizes request or response arcs. These are illustrations of the selected
scenario, not live traffic monitoring.

Scenario controls and view tabs support arrow keys, Home, and End. Motion
respects the system's reduced-motion preference. On narrow screens, diagrams
scroll horizontally to keep their labels readable.

The generic config explorer provides section navigation and a filter search
covering names, types, branch details, and redacted configuration values.
Listeners and clusters stay visible while filtering. Open a document-model
JSON with the file picker or drag it onto the page to explore another config.

## The three inputs

The generated HTML is a deterministic function of three checked-in inputs plus
the pinned `praxis_core` parser:

| Input | File | Owns |
| --- | --- | --- |
| **Config** | `…/full-flow-agentic.yaml` | The *structure*: which filters run, in what order, the clusters/endpoints, and the scalar limits ("knobs"). This is the same file Praxis actually loads. |
| **Sidecar** | `…/full-flow-agentic.visualizer.yaml` | The *semantics*: synthetic filter ids, display names, groups, phase/promotes prose, per-scenario notes, wire payloads, timelines, and cluster/service annotations. |
| **Template** | `…/full-flow-agentic.visualizer.template.html` | The static shell: inline CSS, the vanilla-JS renderer, inline SVG, and six `@@BLOCK@@` placeholders the generator fills. No network, no build step. |

The generator ([`xtask/src/flow_generator.rs`](../../xtask/src/flow_generator.rs))
merges these into the checked-in HTML.

### Structure vs. semantics

The split is deliberate and enforced. **Structure is never invented by the
sidecar** — it is read from the config through the real parser
([`xtask/src/flow_graph.rs`](../../xtask/src/flow_graph.rs)) and cross-checked
against the sidecar at generation time:

- The flattened filter chain (main chain plus IRR inference steps) determines
  filter **count, order, and type**. If the sidecar's filter list disagrees at
  any position — an added, removed, reordered, relocated, or mistyped filter —
  generation **fails** with an actionable message.
- Every `load_balancer` cluster (however deeply nested, including inside branch
  chains and IRR steps) and its endpoints must match the sidecar's declared
  clusters.
- Every scalar limit surfaced as a knob (registered in
  [`xtask/src/flow_visualizer.rs`](../../xtask/src/flow_visualizer.rs)) must
  equal its authoritative config value.

Everything the config cannot express — the prose, the wire examples, the
timelines — comes from the sidecar verbatim. The generator does **not** fabricate
behavioral explanations, and it never silently ignores a structural change.

## Regenerating and verifying

```console
# Rewrite the checked-in HTML from the current config + sidecar.
cargo xtask sync-flow-visualizers --fix

# Verify the checked-in HTML is exactly what the inputs produce (no write).
cargo xtask sync-flow-visualizers
```

The verify form runs as part of `make lint`, so CI fails if:

- the config changed but the visualizer was not regenerated,
- the generated HTML was hand-edited, or
- the sidecar drifted from the config's structure (this always fails, even under
  `--fix`, because it is a real disagreement a human must resolve).

Generation is deterministic: every data block is emitted as order-preserving
pretty JSON, so `--fix` is a no-op when nothing changed and diffs stay legible.

## Making changes

- **Change the narration** (a scenario note, a wire example, a timeline step):
  edit the sidecar YAML, then run `cargo xtask sync-flow-visualizers --fix`.
- **Change the pipeline** (add/remove/reorder a filter, change a cluster or a
  limit): edit the config YAML. Regeneration will fail until you update the
  matching sidecar entry (id, name, group, prose, per-scenario states) so
  structure and semantics agree again, then run `--fix`.
- **Change the look or interactions**: edit the template HTML. Because the
  renderer reads the data blocks at runtime, appearance changes never touch the
  data.
- **Never hand-edit the generated `*.visualizer.html`.** It is overwritten on
  the next `--fix` and the lint gate rejects manual edits.

## Adding a new visualizer

1. Author a config (if it does not already exist) under `examples/configs/`.
2. Author a sidecar `<name>.visualizer.yaml` beside it and a template
   `<name>.visualizer.template.html` (start from the full-flow files).
3. Register the tuple in `VISUALIZERS` in
   [`xtask/src/flow_visualizer.rs`](../../xtask/src/flow_visualizer.rs): the
   config, template, sidecar, and output HTML paths, the pipeline chain name,
   and the knob checks that bind surfaced limits to their config source.
4. Run `cargo xtask sync-flow-visualizers --fix` to produce the HTML, then
   `make lint` to confirm the gate is green.

Sidecar files use the `*.visualizer.yaml` suffix, which is excluded from the
example-config integration-test requirement (they are documentation data, not
Praxis configs).

## Visualizing an arbitrary config

The curated full-flow visualizer above is bespoke: it pairs one config with a
hand-authored sidecar of prose, wire examples, and timelines. For **any other**
Praxis config — one without a sidecar — there is a generic renderer:

```console
# Render any config to a self-contained, offline HTML document.
cargo xtask visualize-config path/to/config.yaml --output /tmp/config.html

# Also emit the extracted "document model" as JSON for offline reuse (below).
cargo xtask visualize-config path/to/config.yaml \
  --output /tmp/config.html --json /tmp/config.model.json
```

This tool has **no sidecar**, so it shows only what can be derived truthfully:

- **Structure** — listeners, top-level clusters, and every filter chain flattened
  with IRR inference steps hoisted inline — comes from the same
  [`FlowGraph`](../../xtask/src/flow_graph.rs) parser the curated generator uses.
- **Semantics** — a one-line "what it does" description on a filter — is shown
  **only for filters this repository defines** (the catalog is
  [`filter_docs::filter_descriptions`](../../xtask/src/filter_docs.rs), the same
  `syn`-parsed source metadata that drives `docs/filters/`). A filter Praxis core
  defines (e.g. `trace_context`, `load_balancer`, `iterative_request_router`) is
  labelled **"semantics unavailable"**; its raw configuration is still shown so
  the structure is never hidden. The tool never fabricates behavior it cannot
  ground in this repository's source.

Configuration values whose keys look sensitive (token, secret, password,
api\_key, authorization, credential, private\_key, client\_secret, bearer, …) are
redacted to the `«redacted»` sentinel before embedding, and highlighted in the
output.

The template lives at
[`xtask/src/assets/generic_visualizer.html`](../../xtask/src/assets/generic_visualizer.html)
and is compiled into the binary with `include_str!`; the generator escapes the
model JSON and substitutes it into the single `@@MODEL@@` placeholder.

### Offline reuse (drag-and-drop) and why the CLI stays authoritative

A generated page includes a file picker and drag-and-drop target. They load a
**document-model `.json`** (the artifact of `--json` above) and re-render in the
browser — no server, no network. They deliberately do **not** accept a raw YAML
config.

The reason is a hard design constraint: structure extraction runs through
`praxis_core`, the real Rust parser, so the topology the page shows is exactly
what Praxis would load. Teaching the browser to parse YAML would mean a second,
independent parser that could disagree with the authoritative one — precisely the
drift this whole system exists to prevent. So YAML → model extraction remains the
CLI's job, and the browser only ever consumes a model the CLI already produced.
To visualize a new config offline, run `visualize-config … --json` once and drop
the resulting `.json` onto any generated page.
