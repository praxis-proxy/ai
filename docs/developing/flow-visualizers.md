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
