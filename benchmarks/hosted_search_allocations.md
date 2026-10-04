# Hosted search allocation evidence

The tests below use `allocation_counter::measure` on fixed, deterministic
inputs. Each test includes the former construction path as a local baseline
and checks that the current path returns identical output where applicable.
The assertions compare both allocation count (`count_total`) and peak live
bytes (`bytes_max`); no wall-clock threshold is used.

Run with Rust 1.96 or newer:

```console
RUSTC_WRAPPER= cargo test -p praxis-ai-apis --features full --lib allocation -- --nocapture
```

| Fixture | Baseline path | Current path | Count: baseline → current | Peak live bytes: baseline → current |
| --- | --- | --- | ---: | ---: |
| 10,000 vector-store result objects, retain 50 | Parse all results into an owned `Value` tree before selecting 50 | Borrow raw result slices and decode only retained candidates | 80,018 → 3 | 9,537,608 → 15,208 |
| 2,048 file citation chunks of 256 bytes | Allocate each annotation and copy the complete context into its wrapper | Reuse annotation scratch and finish in the wrapper allocation | 16,400 → 4,117 | 1,958,951 → 1,149,562 |
| 2,048 citation markers without existing annotations | Stage one offset range per marker | Omit offset ranges when nothing needs remapping | 22,549 → 22,539 | 1,970,488 → 1,937,720 |
| File-search public item with a 512 KiB previous result | Clone the whole item to size its replacement | Size changed members without cloning the item | 16 → 0 | 525,600 → 0 |
| 64 web queries of 4 KiB each | Clone the query array into an intermediate JSON tree | Serialize from borrowed queries | 77 → 8 | 661,953 → 394,656 |
| 64 web results with 4 KiB snippets | Start with `results.len() * 200` output capacity and grow | Reserve the exact formatted length once | 6 → 1 | 614,400 → 264,673 |

These are isolated allocation probes, not process RSS measurements. The
request-wide budget still charges every independently retained payload owner
before accepting a hosted result, and functional tests exercise the proxy
error path and the normal cited result path.
