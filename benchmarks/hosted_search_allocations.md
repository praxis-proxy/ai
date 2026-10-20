# Hosted search allocation evidence

These deterministic unit fixtures measure allocation count and peak live bytes
with `allocation_counter::measure`. Each compares the current implementation
with the former construction pattern on the same input. The assertions cover
both metrics and, for formatting, identical output.

Run each fixture with Rust 1.96 or newer:

```console
for fixture in \
  high_cardinality_decode_avoids_full_result_tree_allocation \
  high_cardinality_context_reuses_chunk_storage \
  citation_heavy_rewrite_skips_unused_offset_ranges \
  large_web_result_format_avoids_buffer_growth; do
  cargo test -p praxis-ai-apis --features full,store-sqlite "$fixture" -- --nocapture
done
```

| Fixed input | Former path | Current path | Allocations, former → current | Peak live bytes, former → current |
| --- | --- | --- | ---: | ---: |
| 10,000 vector store rows, retain 50 | Decode every row into an owned JSON tree | Borrow row slices and decode only the 50 retained rows | 80,016 → 3 | 9,530,008 → 15,208 |
| 2,048 file citation chunks, 256 bytes each | Allocate each chunk annotation and copy the finished context into a wrapper | Reuse one annotation buffer and finish in the wrapper allocation | 16,400 → 4,117 | 1,958,951 → 1,149,562 |
| 2,048 citation markers without existing annotations | Stage an offset range for every marker | Skip offsets when no annotation needs remapping | 22,549 → 22,539 | 1,970,488 → 1,937,720 |
| 64 web results with 4 KiB snippets | Start with `results.len() * 200` capacity and grow | Reserve the formatted length once | 6 → 1 | 614,400 → 264,673 |

These are allocation probes for specific hot paths, not process RSS or a
capacity benchmark. Request budget tests separately exercise below, at, and
above admission boundaries and verify that exhaustion cannot publish a
successful hosted result.

## Combined peak

The per-stage probes above measure one hot path at a time. This combined
fixture holds all four file-search staging artifacts live together — the
parallel raw response bodies, the bounded-decoded results, the private
model-context text, and the staged citations — so the peak includes the raw
bodies and reflects their coexistence. The per-body marker text and file
metadata are shared fixture inputs prepared before measurement. It compares the
current borrow/reuse path against a deep-copy baseline that additionally keeps
a full `serde_json::Value` tree per body, builds model context from all retained
results using owned per-chunk annotations copied into the wrapper, and stages
citations on the legacy path that collects a removal offset range per marker.
Each workload retains the top 5 results per body and stages 5 citation markers
per body; the assertions cover both allocation count and peak live bytes
(`bytes_max`).

Run with:

```console
cargo test -p praxis-ai-apis --features full,store-sqlite --lib \
  combined_hosted_search_staging_peak_beats_deep_copy_baseline -- --nocapture
```

| Fixed input | Deep-copy baseline | Current path | Required result |
| --- | --- | --- | --- |
| Many small: 256 bodies × 32 results × 4 chunks × 64 B, retain 5 | Full `Value` tree per body + per-chunk owned context + legacy citation offsets | Bounded decode + reused context buffer + offset-skipping citations | Fewer allocations and lower peak live bytes |
| Few large: 4 bodies × 8 results × 16 chunks × 4,096 B, retain 5 | Full `Value` tree per body + per-chunk owned context + legacy citation offsets | Bounded decode + reused context buffer + offset-skipping citations | Fewer allocations and lower peak live bytes |

Both workloads carry the same 2 MiB total result-content payload (256 × 32 × 4 ×
64 B and 4 × 8 × 16 × 4,096 B). The test prints the measured values with
`--nocapture`; the assertions require the current path to stay below the
deep-copy baseline on both allocation count and peak live bytes.
