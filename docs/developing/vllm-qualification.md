# vLLM Responses gateway qualification evidence

The `vLLM Integration` workflow runs the complete Responses SDK suite on a real
GPU for scheduled runs, `run_live_vllm=true` dispatches, and the
`vllm-full-suite` PR label. A hosted summary job writes `qualification.json`
and an Actions summary even if GPU provisioning, setup, or runner registration
fails. The raw `vllm-gpu-responses-raw` artifact keeps per-case JSON, pytest
output with focused HTTP/SSE failure assertions, and available service logs
for 30 days, as does the canonical report. The release workflow attaches its
chosen report to the GitHub Release for longer-term access.

This is the GPU report. Releases also publish a broader
[CI evidence report](release.md#changelog) (`release-qualification.md` and
`release-qualification.json`) with exact-commit workflow, job, and step
outcomes. A missing GPU result does not hide other CI results, and the CI
report never counts simulator or CPU jobs as live GPU qualification.

The **Live vLLM CPU Decisions and System One** job appears in that CI evidence
under **vLLM integration**. It records separate step outcomes for OpenAI SDK
requests to `POST /v1/decisions` and HTTP requests to `POST /v1/systemone`,
both through Praxis to a pinned vLLM CPU image serving a pinned revision of
`Qwen/Qwen3-0.6B`. The Decisions example sends the former to the classified
route and the latter to its default passthrough route. The tests validate
typed answers and token usage without requiring a particular answer.

## Report version and interpretation

The report has `schema_version: 3` and
`profile: responses-gateway-live-vllm-gpu`. It records native `/v1/responses`
and Responses-to-Chat `/v1/chat/completions` results separately.
`gateway.checkout_sha` comes from `git rev-parse HEAD` in the GPU checkout;
the workflow SHA and attempt are recorded separately.
`gateway.binary_sha256` identifies the locally built debug/full gateway
binary, **not** the published release container.

`backend.local_image_id` is the immutable ID of the model-baked vLLM image the
suite served; its mutable tag is recorded separately. The suite reuses the
build published by `vllm-gpu-container.yaml` when it matches the checkout and
builds from source otherwise (see
[Reusing published images](gpu-nightly-suite.md#reusing-published-images)):

| Field | Meaning |
| --- | --- |
| `backend.image_source` | `registry` for a pulled image, `local-build` for one built on the runner. |
| `backend.registry_digest` | The pulled image's digest. Null for a local build, which has none. |
| `backend.built_from_commit` | Commit the image was built from, read from its `org.opencontainers.image.revision` label. |
| `backend.containerfile_sha256` | Hash of the `Containerfile` the **image** was built from, read from its label. It is not a hash of the working tree, which may describe a build that never happened on this runner. |
| `backend.base_image` | The `FROM` reference the image was built on, read from its label. |

The model ID is a build input; the resolved model revision is null when the
image build cannot expose it. Null provenance means unavailable, never an
inferred value.

Each client lane using baked weights also records the image its own container
served under `suites.<client>_acceptance.backend`: the tag, local image ID,
model, revision, base image, and `image_source` (`registry`, `runner` for an
image an earlier job prepared, or `build`), plus the registry digest when that
job pulled it. Matching local image IDs show the suite and the clients ran
against the same image.

| Field | Meaning |
| --- | --- |
| `schema_version`, `profile`, `status`, `reason` | Versioned Responses gateway profile and combined qualification result. |
| `workflow` | Run ID, attempt, URL, event, and workflow SHA. |
| `gateway`, `backend`, `model`, `configuration`, `runner` | Tested binary, resolved dependencies, local image, model, config, storage, and GPU provenance. |
| `phases` | Provisioning, setup, and Responses suite execution results. |
| `suites` | Separate native, translation, supporting, and credentialed pytest results, plus simulator and client acceptance entries. |
| `limitations`, `diagnostics` | Supported boundaries, linked issues, and retained artifact locations. |

`status` is `passed`, `failed`, `incomplete`, `needs_review`, `skipped`,
`not_requested`, or `unavailable`. A `passed` result requires:

- at least one passing native case and one passing translation case;
- no failed or unexpectedly passing selected cases;
- a recorded tested binary, Praxis version, vLLM version/local image, model
  ID, config hash, and SDK version.

A passing translation case cannot stand in for a skipped native suite.
Supporting checks run in the same process but do not count toward either live
path; their failures still fail qualification. Expected failures stay
visible. Strict unexpected passes classify as `needs_review` even though
pytest exits nonzero.

`configuration` records the reference full-flow config path and hash, other
config hashes the suite used, the live/model/database overrides, and
PostgreSQL storage. Selected and deselected tests are separate arrays. Each
selected case has a `profile` (`native`, `translation`, or `supporting`) set by
the pytest plugin from the client fixture that selects the pipeline, and a
result (`passed`, `failed`, `skipped`, `xfailed`, `xpassed`, or `unexecuted`)
with its reason. A new path the plugin cannot attribute is reported as
`unclassified` and blocks a passing qualification.

- `xfailed` marks a known expectation, not necessarily a Praxis defect.
- `xpassed` calls for contract review.
- A skipped credentialed provider case is no evidence that the provider path
  works. The separately run credentialed Tavily case uses the translation path
  but stays outside the main translation counters.

The native suite, translation suite, supporting checks, credentialed tools,
simulator, and optional Claude/Codex acceptance are separate entries. Optional
acceptance records each scenario's step outcomes and available client logs;
Anthropic SDK scenarios also include selected/deselected pytest cases and
reasons. The report qualifies selected text and tool behavior over native and
translated HTTP/SSE paths, not the complete OpenAI API. Streamed Conversation
append is tracked by [#410](https://github.com/praxis-proxy/ai/issues/410);
background, WebSockets, and multimodal models are excluded.

The top-level backend, model, and configuration describe the Responses suite.
To reproduce a client acceptance case, use the run in `workflow.url` and the
[GPU job and pin guide](gpu-nightly-suite.md#jobs-in-the-main-suite-vllm-integrationyaml)
at the revision in `workflow.run_sha`.

## Local reproduction

On a GPU host with the workflow's vLLM image, PostgreSQL, OGX, and
configuration, build the gateway and run the suite with the reporting plugin:

```console
cargo build -p praxis-ai-proxy --features full,openai-all,store-postgres
PRAXIS_QUALIFICATION_RESULTS=/tmp/responses-cases.json \
  DATABASE_URL=postgres://praxis:praxis@127.0.0.1:5432/praxis \
  VLLM_MODEL=Qwen/Qwen3-8B VLLM_TEST_BACKEND=live \
  uv run tests/integration/sdk/openai/test_openai_responses_vllm.py -s \
    -p qualification_plugin -k 'not live_tavily_web_search_returns_real_sources'
```

Run the credentialed Tavily case separately, with a separate result path,
using `-k live_tavily_web_search_returns_real_sources`, a valid
`TAVILY_API_KEY`, and `PRAXIS_TEST_REQUIRE_LIVE_WEB_SEARCH=1`. The workflow
runs `.github/scripts/vllm_qualification.py capture` after both test steps and
`finish` on a hosted runner. Test the script with
`python3 -m unittest discover -s tests/qualification`.

## Release selection from recent nightlies

For a release tag, the workflow peels annotated tags with
`git rev-parse 'HEAD^{commit}'`. It reuses the latest successful scheduled GPU
nightly on `main` started within the preceding 48 hours, if its tested commit
is the release commit or an ancestor of it. Release publication does not
start or wait for a new GPU run.

All required GPU acceptance, startup, teardown, and summary jobs must succeed,
and the downloaded report must be complete and passing. The report is verified
against its original run ID, attempt, and tested SHA; an unavailable or
invalid report does not count as qualification. When the search reuses an
earlier successful nightly, it also reports every newer eligible failed,
incomplete, or unfinished nightly.

The published report keeps the tested checkout SHA, date, versions, and
Actions link, and identifies the release SHA separately. Changes after the
tested checkout are not claimed to have passed GPU validation. If no recent
ancestor nightly qualifies, GPU evidence is unavailable and publication
continues with that status. CI evidence stays scoped to the exact release
commit.

The release job inserts marked CI and GPU qualification sections, attaches all
three reports, and verifies they are present. Rerun Release on the tag to
refresh this snapshot. Repository release immutability settings may reject the
update, which shows as a workflow error.

`tests/qualification/release-notes.fixture.md` is a synthetic notes fixture
showing the rendered section and idempotent composition; it is not real GPU
evidence. The CI summary has its own synthetic fixture at
`tests/qualification/release-ci-notes.fixture.md`.

## CPU embeddings

The [vLLM Embeddings Qualification workflow](../../.github/workflows/vllm-embeddings.yaml)
serves a pinned Granite embedding model in pooling mode on a standard runner.
The OpenAI SDK calls `/v1/embeddings` through Praxis and checks batch order
against individual requests, 768-dimensional finite vectors, token usage, and
unknown-model handling. The workflow records the image and model revision in
its summary and publishes JUnit results. Release qualification lists the
latest eligible run for the release commit.
