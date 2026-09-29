# vLLM GPU Container

Pre-built [vLLM](https://docs.vllm.ai/) GPU container image with an
inference model baked in at build time, eliminating runtime download
latency.

Published to `ghcr.io/praxis-proxy/vllm-gpu`.

## Requirements

- Docker with [BuildKit](https://docs.docker.com/build/buildkit/) enabled
  (default in Docker 23+), or Podman 4+; required for the `# syntax=`
  directive and `--secret` mounts used below.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  on the host, so `docker run --gpus all` (or
  `podman run --device nvidia.com/gpu=all`) can expose the GPU to the
  container.

## Default model

CI defaults to [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), the same
model the full GPU suite in `vllm-integration.yaml` serves (`VLLM_GPU_MODEL`).
The suite builds the image from its checked-out commit and tests that build.

The 8.19 B bf16 checkpoint is ~15.3 GiB of weights. It needs a card with enough
VRAM left over for a KV cache; the CI runner is a `g5.xlarge` (A10G, 24 GiB,
compute capability 8.6), which also gives it bfloat16 and FlashAttention-2.
`INFERENCE_MODEL` builds any other HuggingFace ID — on a smaller card, a
quantized variant such as
[`Qwen/Qwen3-8B-AWQ`](https://huggingface.co/Qwen/Qwen3-8B-AWQ) (~5.7 GiB int4)
fits where the bf16 checkpoint does not.

## Building

```console
docker build \
  --build-arg INFERENCE_MODEL=Qwen/Qwen3-8B \
  --tag vllm-gpu:Qwen3-8B \
  --file vllm/Containerfile \
  .
```

For gated models that require a HuggingFace token:

```console
docker build \
  --build-arg INFERENCE_MODEL=meta-llama/Llama-3.1-8B \
  --secret id=hf_token,env=HF_TOKEN \
  --tag vllm-gpu:Llama-3.1-8B \
  --file vllm/Containerfile \
  .
```

To bake a specific model revision, pass `INFERENCE_REVISION` with the full
Hugging Face commit hash. The Anthropic vision GPU workflow does this for
`Qwen/Qwen3-VL-4B-Instruct` and runs its SDK image tests against the local
built image.

```console
docker build \
  --build-arg INFERENCE_MODEL=Qwen/Qwen3-VL-4B-Instruct \
  --build-arg INFERENCE_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --tag vllm-gpu:Qwen3-VL-4B-Instruct \
  --file vllm/Containerfile \
  .
```

## Running

```console
docker run --gpus all -p 8000:8000 vllm-gpu:Qwen3-8B \
  --model /opt/vllm/models/Qwen/Qwen3-8B \
  --max-model-len 4096 \
  --served-model-name Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9
```

Note the asymmetry between the two model names: the **image tag** uses the
short form (`Qwen3-8B`, the HuggingFace ID with its org prefix stripped),
while `--served-model-name` uses the **full** ID (`Qwen/Qwen3-8B`). The
integration suite defaults `VLLM_MODEL` to the full ID, so serving under the
short name would make every test request 404.

The flags above match what CI serves, and come from the shared
[`start-vllm`](../.github/actions/start-vllm/action.yml) action —
`--enable-auto-tool-choice`/`--tool-call-parser` are required by the tool-loop
tests, and `--reasoning-parser` by the reasoning-content tests.

Additional flags for larger GPUs:

```console
--max-model-len 32768
--enable-chunked-prefill
--enable-prefix-caching
```

## Health check

```console
curl http://localhost:8000/health
```

## Verifying the image on a GPU node

A `/health` probe only proves the server booted. To prove the image actually
serves Praxis traffic, run the repository's live-vLLM coverage against it —
the same `critical_vllm` subset CI gates on:

```console
# 1. Start the image on the GPU (see "Running" above), then:
cargo build -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite

# 2. OGX backs the file_search / file_resolve tests in the critical set.
uv run --with-requirements tests/integration/ogx-constraints.txt \
  ogx run starter --insecure &

# 3. Drive real Responses API traffic through Praxis into the GPU image.
VLLM_TEST_BACKEND=live VLLM_MODEL=Qwen/Qwen3-8B \
  uv run tests/integration/sdk/openai/test_openai_responses_vllm.py \
  -s -m "critical_vllm"
```

Drop `-m "critical_vllm"` to run the complete live Responses suite. The
nightly GPU job builds this image from its checkout and runs that full suite
with a PostgreSQL response store.

## CI

The `.github/workflows/vllm-gpu-container.yaml` workflow builds, tests, and
publishes the image on a GPU runner, in that order — a failing test skips the
push, so an image is never published untested.

The test stage is not a bespoke smoke check: it starts the freshly built image
on the GPU, builds Praxis, starts OGX, and runs
`tests/integration/sdk/openai/test_openai_responses_vllm.py -m critical_vllm`
with `VLLM_TEST_BACKEND=live`. That is the same suite, same marker, and same
[`start-vllm`](../.github/actions/start-vllm/action.yml) /
[`wait-vllm`](../.github/actions/wait-vllm/action.yml) actions the
`vllm-live-cpu` job in `vllm-integration.yaml` uses against the CPU image, so
GPU and CPU images are held to one standard.

The scheduled and `vllm-full-suite` label runs in `vllm-integration.yaml` also
build this image from the checked-out commit and exercise the complete live
Responses suite against that local build. The PR image job and full-suite job
are separate builds; neither publishes an image from a PR.

Triggers:

- **Push to `main`** (when `Containerfile`, the GPU build action, the vLLM
  start/wait actions, the EC2 runner actions, or the workflow changes) —
  builds, tests, and pushes both
  `:<model>` (e.g. `:Qwen3-8B`) and `:latest`.
- **Same-repository pull request** / **merge queue** — builds and tests only;
  nothing is pushed. Fork and Dependabot PRs skip the secret-backed GPU runner.
- **Manual (`workflow_dispatch`)** — accepts an `inference_model` input to
  build an arbitrary HuggingFace model and a `gpu_instance_type` input to
  override the EC2 instance type; pushes `:<model>` but not `:latest`. Defaults
  to `Qwen/Qwen3-8B`.

The image tag is derived from the model ID with the org prefix stripped
(`Qwen/Qwen3-8B` → `Qwen3-8B`). Gated models are supported via the
`HUGGING_FACE_HUB_TOKEN` repository secret.

### GPU runner

No GPU runner is permanently registered for this repository, so the workflow
provisions one for the run, exactly as the nightly GPU jobs in
`vllm-integration.yaml` do:

1. `gpu-start-runner` launches an ephemeral EC2 instance via AWS OIDC using the
   shared [`start-ec2-runner`](../.github/actions/start-ec2-runner/action.yml)
   action, and exports the dynamically generated runner label.
2. `build-test` sets `runs-on` to that label.
3. `gpu-stop-runner` runs with `if: always()` and terminates the instance via
   [`stop-ec2-runner`](../.github/actions/stop-ec2-runner/action.yml), so a
   failed or cancelled build never leaks an instance.

The default instance type is `g5.xlarge` (A10G, 24 GiB), matching
`GPU_INSTANCE_TYPE` in `vllm-integration.yaml`; `gpu_instance_type` overrides it
for transient capacity shortages. The build job still checks `nvidia-smi` before
the multi-gigabyte build starts, and detects Docker or Podman to select
`--gpus all` or `--device nvidia.com/gpu=all` accordingly.

## Dependabot

The `FROM` directive in `Containerfile` is monitored by Dependabot for
weekly base image updates (see `.github/dependabot.yaml`).
