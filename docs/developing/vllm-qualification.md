# vLLM Responses gateway qualification evidence

The existing `vLLM Integration` workflow runs the complete Responses SDK suite
on a real GPU for scheduled runs, `run_live_vllm=true` dispatches, and the
`vllm-full-suite` PR label. A hosted summary job writes `qualification.json`
and an Actions summary even if GPU provisioning, setup, or runner registration
fails. The raw `vllm-gpu-responses-raw` artifact retains per-case JSON, pytest
output with focused HTTP/SSE failure assertions, and available service logs for
30 days. The canonical report is also retained for 30 days.
The release workflow attaches its chosen report to the GitHub Release for
longer term access.

This is the GPU-specific report. Releases also publish a broader
[CI evidence report](../release.md#changelog) with exact-commit workflow, job,
and step outcomes in `release-qualification.md` and `release-qualification.json`.
An unavailable GPU result does not hide successful unit, integration, SDK,
security, or other CI results. The CI report never counts simulator or CPU
jobs as live GPU qualification.

## Report version and interpretation

The report has `schema_version: 3` and
`profile: responses-gateway-live-vllm-gpu`. The combined GPU execution records
native `/v1/responses` and Responses-to-Chat `/v1/chat/completions` results
separately. `gateway.checkout_sha` comes from `git rev-parse HEAD` inside the
GPU checkout. The workflow SHA and attempt are
recorded separately. `gateway.binary_sha256` identifies the locally built
debug/full gateway binary; it is **not** the separately published release
container.

`backend.local_image_id` is the immutable ID of the model-baked vLLM image the
suite served, and its mutable tag is recorded separately. The suite reuses the
build published by `vllm-gpu-container.yaml` when it matches the checkout and
builds from source otherwise, so the report states which:

| Field | Meaning |
| --- | --- |
| `backend.image_source` | `registry` for a pulled image, `local-build` for one built on the runner. |
| `backend.registry_digest` | The pulled image's digest. Null for a local build, which has none. |
| `backend.built_from_commit` | Commit the image was built from, read from its `org.opencontainers.image.revision` label. |
| `backend.containerfile_sha256` | Hash of the `Containerfile` the **image** was built from, read from its label — not a hash of the working tree, which may describe a build that never happened on this runner. |
| `backend.base_image` | The `FROM` reference the image was built on, read from its label. |

The model ID is a build input; the resolved model revision is null when the
image build cannot expose it. Null provenance means unavailable, not an
inferred value.

| Field | Meaning |
| --- | --- |
| `schema_version`, `profile`, `status`, `reason` | Versioned Responses gateway profile and combined qualification result. |
| `workflow` | Run ID, attempt, URL, event, and workflow SHA. |
| `gateway`, `backend`, `model`, `configuration`, `runner` | Tested binary, resolved dependencies, local image, model, config, storage, and GPU provenance. |
| `phases` | Provisioning, setup, and Responses suite execution results. |
| `suites` | Separate native, translation, supporting, and credentialed pytest results, plus simulator and client acceptance entries. |
| `limitations`, `diagnostics` | Supported boundaries, linked issues, and retained artifact locations. |

`status` is `passed`, `failed`, `incomplete`, `needs_review`, `skipped`,
`not_requested`, or `unavailable`. A `passed` combined result requires at least
one passing native case and one passing translation case, no failed or
unexpectedly passing selected cases, and the tested binary, Praxis version,
vLLM version/local image, model ID, config hash, and SDK version to be recorded.
One passing translation case cannot turn a skipped native suite into native
proof. Supporting checks run in the same process but do not contribute to
either live path's counters; failures in those checks still fail qualification.
Expected failures remain visible; strict unexpected passes classify as
`needs_review` even though pytest exits nonzero.

`configuration` records the reference full-flow config path and hash,
additional config hashes used by the suite, the live/model/database overrides,
and PostgreSQL storage. Selected and deselected tests are separate arrays.
Selected cases carry a `profile` (`native`, `translation`, or `supporting`),
`passed`, `failed`, `skipped`, `xfailed`, `xpassed`, or `unexecuted`, with reasons.
The pytest plugin determines the profile from the client fixture that selects
the pipeline. New paths without known attribution are reported as
`unclassified` and prevent a passing qualification. `xfailed` marks a known
expectation, not automatically a Praxis defect. `xpassed` calls for contract
review. A skipped credentialed provider case provides no evidence that the
provider path works. The separately executed credentialed Tavily case uses the
translation path and remains outside the main translation counters.
The native suite, translation suite, supporting checks, credentialed tools,
simulator, and optional Claude/Codex acceptance are separate entries. Optional
acceptance records individual scenario step outcomes and available client logs. Anthropic
SDK scenarios additionally include selected/deselected pytest cases and
reasons. This report qualifies selected text and tool behavior over native and
translated HTTP/SSE paths, not the complete OpenAI API. Streamed Conversation
append is tracked by
[#410](https://github.com/praxis-proxy/ai/issues/410); background,
WebSockets, and multimodal models are excluded.

## Local reproduction

On a GPU host with the same vLLM image, PostgreSQL, OGX, and configuration as
the workflow, build the gateway and run the existing suite with the reporting
plugin:

```console
cargo build -p praxis-ai-proxy --features full,openai-all,store-postgres
PRAXIS_QUALIFICATION_RESULTS=/tmp/responses-cases.json \
  DATABASE_URL=postgres://praxis:praxis@127.0.0.1:5432/praxis \
  VLLM_MODEL=Qwen/Qwen3-8B VLLM_TEST_BACKEND=live \
  uv run tests/integration/sdk/openai/test_openai_responses_vllm.py -s \
    -p qualification_plugin -k 'not live_tavily_web_search_returns_real_sources'
```

Run the credentialed Tavily case separately with
`-k live_tavily_web_search_returns_real_sources`, a valid `TAVILY_API_KEY`,
and `PRAXIS_TEST_REQUIRE_LIVE_WEB_SEARCH=1`. Use a separate result path. The
workflow invokes `.github/scripts/vllm_qualification.py capture` after both
test steps and `finish` on a hosted runner. The script's focused tests run with
`python3 -m unittest discover -s tests/qualification`.

## Release selection from recent nightlies

For a release tag, the workflow peels annotated tags with
`git rev-parse 'HEAD^{commit}'`. It reuses the latest successful scheduled
GPU nightly on `main` started within the preceding 48 hours, provided its tested
commit is an ancestor of the release commit (or the same commit). Release
publication does not launch or wait for a new GPU run.

All required GPU acceptance, startup, teardown, and summary jobs must succeed,
and the downloaded qualification report must be complete and passing. The
report is verified against its original run ID, attempt, and tested SHA.
Unavailable or invalid reports are not accepted as successful qualification.
The search can reuse an earlier successful nightly, but reports every newer
eligible failed, incomplete, or unfinished nightly alongside it.

The published report preserves the tested checkout SHA, date, versions, and
Actions link, and separately identifies the release SHA. Changes after the
tested checkout are not claimed to have passed GPU validation. If no acceptable
recent ancestor nightly exists, GPU evidence is unavailable; release publication
continues with that explicit status. CI evidence remains scoped to the exact
release commit.

The release job inserts marked CI and GPU qualification sections, attaches all
three reports, and verifies their presence. Rerun Release on the tag to refresh
this publication-time snapshot. Repository release immutability settings may
reject an update, which is surfaced as a workflow error.

A synthetic, reviewable notes fixture lives in
`tests/qualification/release-notes.fixture.md`. It demonstrates the rendered
section and idempotent composition; it is not real GPU qualification evidence.
The broader CI summary has a separate synthetic fixture at
`tests/qualification/release-ci-notes.fixture.md`.
