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

CI defaults to [`Qwen/Qwen3-8B-AWQ`](https://huggingface.co/Qwen/Qwen3-8B-AWQ),
the official AWQ int4 build of Qwen3-8B. The variant is picked by the CI GPU,
not by preference — the runner is a 16 GiB Tesla T4:

| Checkpoint | Weights | Fits a 16 GiB T4? |
| --- | --- | --- |
| `Qwen/Qwen3-8B` (bf16) | 8.19 B params, 15.3 GiB | No — no room left for a KV cache |
| `Qwen/Qwen3-8B-FP8` | ~8.2 GiB | No — FP8 needs sm_89+; the T4 is sm_75 |
| `Qwen/Qwen3-8B-AWQ` (int4) | ~5.7 GiB | Yes — ~7 GiB left for the KV cache |

AWQ is a
[Turing-supported quantization in vLLM](https://docs.vllm.ai/en/latest/features/quantization/),
so the T4 runs it without a source build. `INFERENCE_MODEL` overrides this for
any host with more VRAM; on Ampere or newer, `Qwen/Qwen3-8B` fits unquantized.

## Building

```console
docker build \
  --build-arg INFERENCE_MODEL=Qwen/Qwen3-8B-AWQ \
  --tag vllm-gpu:Qwen3-8B-AWQ \
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

## Running

```console
docker run --gpus all -p 8000:8000 vllm-gpu:Qwen3-8B-AWQ \
  --model /opt/vllm/models/Qwen/Qwen3-8B-AWQ \
  --max-model-len 4096 \
  --served-model-name Qwen/Qwen3-8B-AWQ \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --dtype float16 \
  --gpu-memory-utilization 0.9
```

`--dtype float16` is for the T4 CI runner specifically: Turing (sm_75) has no
bfloat16, so vLLM would otherwise downcast the bf16 checkpoint with a warning.
On Ampere or newer, drop the flag and let vLLM default to `auto`.

Note the asymmetry between the two model names: the **image tag** uses the
short form (`Qwen3-8B-AWQ`, the HuggingFace ID with its org prefix stripped),
while `--served-model-name` uses the **full** ID (`Qwen/Qwen3-8B-AWQ`). The
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
cargo build -p praxis-ai-proxy --no-default-features --features store-sqlite

# 2. OGX backs the file_search / file_resolve tests in the critical set.
uv run --with-requirements tests/integration/ogx-constraints.txt \
  ogx run starter --insecure &

# 3. Drive real Responses API traffic through Praxis into the GPU image.
VLLM_TEST_BACKEND=live VLLM_MODEL=Qwen/Qwen3-8B-AWQ \
  uv run tests/integration/sdk/openai/test_openai_responses_vllm.py \
  -s -m "critical_vllm"
```

Drop `-m "critical_vllm"` to run the complete live Responses suite — the same
thing the nightly `vllm-integration.yaml` run does against the CPU image.

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

Triggers:

- **Push to `main`** (when `Containerfile`, the vLLM start/wait actions, or the
  workflow changes) — builds, tests, and pushes both `:<model>` (e.g.
  `:Qwen3-8B-AWQ`) and `:latest`.
- **Pull request** / **merge queue** — builds and tests only; nothing is
  pushed.
- **Manual (`workflow_dispatch`)** — accepts an `inference_model` input to
  build an arbitrary HuggingFace model and a `runner` input to override the GPU
  runner label; pushes `:<model>` but not `:latest`. Defaults to
  `Qwen/Qwen3-8B-AWQ`.

The image tag is derived from the model ID with the org prefix stripped
(`Qwen/Qwen3-8B-AWQ` → `Qwen3-8B-AWQ`). Gated models are supported via the
`HF_TOKEN` repository secret.

### GPU runner

The job runs on `gpu-t4-4-core` by default; a `workflow_dispatch` run can point
it at another label via the `runner` input. The runner must expose a GPU
(`nvidia-smi` is checked before the multi-gigabyte build starts) and provide
either Docker or Podman — the job detects which and selects `--gpus all` or
`--device nvidia.com/gpu=all` accordingly.

`gpu-t4-4-core` is a 16 GiB Tesla T4, which is what constrains the default
model (see [Default model](#default-model)) and forces `--dtype float16`. A
`workflow_dispatch` run against a larger card can pass an unquantized
`inference_model` alongside the `runner` override.

## Dependabot

The `FROM` directive in `Containerfile` is monitored by Dependabot for
weekly base image updates (see `.github/dependabot.yaml`).
