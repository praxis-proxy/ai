# Flow visualizers

A **flow visualizer** is a self-contained HTML file that explains one Praxis
config as an interactive diagram: the filter pipeline, the clusters and
external services it calls, the scenarios it serves, and, for agentic configs,
the iterative request router (IRR) tool loop, with on-the-wire
request/response examples and step-by-step timelines.

The canonical example covers the Responses API full-flow agentic gateway:

- Config: `examples/configs/openai/responses/full-flow-agentic.yaml`
- Visualizer: `examples/configs/openai/responses/full-flow-agentic.visualizer.html`

The HTML is **generated**. This page explains how and how to change it.

## Exploring the pages

The curated page opens on **Topology**, with separate **Filters**, **Wire
exchange**, and **Timeline** views. Selecting a scenario updates all four
views, and switching views keeps the selection. **Play trace** opens the filter
view and steps through the active filters in order. The phase control opens
the topology and highlights request or response arcs. These illustrate the
selected scenario; they do not monitor live traffic.

Scenario controls and view tabs support arrow keys, Home, and End. Motion
follows the system's reduced-motion preference. On narrow screens, diagrams
scroll horizontally so labels stay readable.

The generic config explorer has section navigation and a filter search over
names, types, branch details, and redacted configuration values. Listeners and
clusters stay visible while filtering. To explore another config, open a
document-model JSON with the file picker or drag it onto the page.

## The three inputs

The generated HTML is a deterministic function of three checked-in inputs plus
the pinned `praxis_core` parser:

| Input | File | Owns |
| --- | --- | --- |
| **Config** | `…/full-flow-agentic.yaml` | The *structure*: which filters run, in what order, the clusters/endpoints, and the scalar limits ("knobs"). This is the same file Praxis loads. |
| **Sidecar** | `…/full-flow-agentic.visualizer.yaml` | The *semantics*: synthetic filter ids, display names, groups, phase/promotes prose, per-scenario notes, wire payloads, timelines, and cluster/service annotations. |
| **Template** | `…/full-flow-agentic.visualizer.template.html` | The static shell: inline CSS, the vanilla-JS renderer, inline SVG, and six `@@BLOCK@@` placeholders the generator fills. No network, no build step. |

The generator ([`xtask/src/flow_generator.rs`](../../xtask/src/flow_generator.rs))
merges these into the checked-in HTML.

### Structure vs. semantics

The sidecar never defines structure. Structure is read from the config by the
real parser ([`xtask/src/flow_graph.rs`](../../xtask/src/flow_graph.rs)) and
checked against the sidecar at generation time:

- The flattened listener pipeline (its referenced filter chains composed in
  order, each chain's inherited conditions folded in, plus IRR inference
  steps) sets filter **count, order, and type**. If the sidecar's filter list
  disagrees at any position (an added, removed, reordered, relocated, or
  mistyped filter), generation **fails** with an actionable message.
- Every `load_balancer` cluster, however deeply nested (including inside
  branch chains and IRR steps), and its endpoints must match the sidecar's
  declared clusters.
- Every scalar limit shown as a knob (registered in
  [`xtask/src/flow_visualizer.rs`](../../xtask/src/flow_visualizer.rs)) must
  equal its config value.

Everything the config cannot express (prose, wire examples, timelines) comes
from the sidecar verbatim. The generator does not invent behavior
descriptions and never ignores a structural change.

## Regenerating and verifying

```console
# Rewrite the checked-in HTML from the current config + sidecar.
cargo xtask sync-flow-visualizers --fix

# Verify the checked-in HTML is exactly what the inputs produce (no write).
cargo xtask sync-flow-visualizers
```

The verify form runs in `make lint`, so CI fails if:

- the config changed but the visualizer was not regenerated,
- the generated HTML was hand-edited, or
- the sidecar drifted from the config's structure. This fails even under
  `--fix`, because a human must resolve the disagreement.

Every data block is emitted as order-preserving pretty JSON, so `--fix` is a
no-op when nothing changed and diffs stay readable.

## Making changes

- **Narration** (a scenario note, a wire example, a timeline step): edit the
  sidecar YAML, then run `cargo xtask sync-flow-visualizers --fix`.
- **Pipeline** (add/remove/reorder a filter, change a cluster or a limit):
  edit the config YAML. Regeneration fails until you update the matching
  sidecar entry (id, name, group, prose, per-scenario states); then run
  `--fix`.
- **Look or interactions**: edit the template HTML. The renderer reads the data
  blocks at runtime, so appearance changes never touch the data.
- **Never hand-edit the generated `*.visualizer.html`.** The next `--fix`
  overwrites it and the lint gate rejects manual edits.

## Adding a new visualizer

1. Write a config under `examples/configs/` if one does not exist.
2. Write a sidecar `<name>.visualizer.yaml` beside it and a template
   `<name>.visualizer.template.html` (start from the full-flow files).
3. Register the tuple in `VISUALIZERS` in
   [`xtask/src/flow_visualizer.rs`](../../xtask/src/flow_visualizer.rs): the
   config, template, sidecar, and output HTML paths, the listener whose
   composed chains the HTML flattens, and the knob checks that bind shown
   limits to their config source.
4. Run `cargo xtask sync-flow-visualizers --fix` to produce the HTML, then
   `make lint` to confirm the gate passes.

Sidecar files use the `*.visualizer.yaml` suffix, which is excluded from the
example-config integration-test requirement because they are documentation
data, not Praxis configs.

## Visualizing an arbitrary config

For any config without a sidecar, use the generic renderer:

```console
# Render any config to a self-contained, offline HTML document.
cargo xtask visualize-config path/to/config.yaml --output /tmp/config.html

# Also emit the extracted "document model" as JSON for offline reuse (below).
cargo xtask visualize-config path/to/config.yaml \
  --output /tmp/config.html --json /tmp/config.model.json
```

With no sidecar, it shows only what it can derive from source:

- **Structure** (listeners, top-level clusters, and every filter chain
  flattened with IRR inference steps inline) comes from the same
  [`FlowGraph`](../../xtask/src/flow_graph.rs) parser the curated generator
  uses.
- **Semantics**, a one-line description per filter, appear **only for filters
  this repository defines**. The catalog is
  [`filter_docs::filter_descriptions`](../../xtask/src/filter_docs.rs), the
  same `syn`-parsed source metadata that drives `docs/filters/`. A filter
  defined by Praxis core (e.g. `trace_context`, `load_balancer`,
  `iterative_request_router`) is labelled **"semantics unavailable"**, and its
  raw configuration is still shown.

Configuration values whose keys look sensitive (token, secret, password,
api\_key, authorization, credential, private\_key, client\_secret, bearer, …)
are replaced with the `«redacted»` sentinel before embedding and highlighted in
the output.

The template,
[`xtask/src/assets/generic_visualizer.html`](../../xtask/src/assets/generic_visualizer.html),
is compiled into the binary with `include_str!`; the generator escapes the
model JSON and substitutes it into the single `@@MODEL@@` placeholder.

### Offline reuse (drag-and-drop) and why the CLI stays authoritative

A generated page has a file picker and drag-and-drop target that load a
**document-model `.json`** (the `--json` output above) and re-render in the
browser with no server or network. They do **not** accept a raw YAML config.

Structure extraction runs through `praxis_core`, the real Rust parser, so the
page shows exactly the topology Praxis would load. A YAML parser in the
browser would be a second parser that could disagree with it. YAML → model
extraction therefore stays in the CLI. To visualize a new config offline, run
`visualize-config … --json` once and drop the resulting `.json` onto any
generated page.
