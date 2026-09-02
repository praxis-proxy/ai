# Migrating to 0.3.0

Version 0.3.0 normalizes the failure-policy keys
used by outbound callout filters.

## Proxy configuration

### Outbound callout failure policy

Callout filters previously spelled the same fail-open/fail-closed
choice three different ways. They now share one key, `on_failure`,
with unchanged `open` / `closed` values and an unchanged `closed`
default. Rename the key in place:

| Filter | 0.2.x key | 0.3.0 key |
| --- | --- | --- |
| `anthropic_web_search` | `provider_failure_mode` | `on_failure` |
| `openai_web_search` | `provider_failure_mode` | `on_failure` |
| `openai_responses_compact` | `callout_failure_mode` | `on_failure` |
| `openai_file_search_callout` | `callout_failure_mode` | `on_failure` |
| `http_callout` | `on_failure` | `on_failure` (unchanged) |

The old keys are not accepted as aliases. Because these filters use
`deny_unknown_fields`, a stale key fails validation at startup with a
message naming the offending field.

`openai_file_resolve`'s `on_missing` key is **not** part of this
rename and keeps its `continue` / `reject` values, and `continue`
remains its default.

## Rust API

<!-- Documentation for any 0.3.x API changes -->