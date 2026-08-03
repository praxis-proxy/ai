# vLLM GPU Container

Pre-built [vLLM](https://docs.vllm.ai/) GPU container image with an
inference model baked in at build time, eliminating runtime download
latency.

Published to `ghcr.io/praxis-proxy/vllm-gpu`.

## Requirements

- Docker with [BuildKit](https://docs.docker.com/build/buildkit/) enabled
  (default in Docker 23+); required for the `# syntax=` directive and
  `--secret` mounts used below.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  on the host, so `docker run --gpus all` can expose the GPU to the
  container.

## Building

```console
docker build \
  --build-arg INFERENCE_MODEL=Qwen/Qwen3-0.6B \
  --tag vllm-gpu:Qwen3-0.6B \
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
docker run --gpus all -p 8000:8000 vllm-gpu:Qwen3-0.6B \
  --model /opt/vllm/models/Qwen/Qwen3-0.6B \
  --max-model-len 4096 \
  --served-model-name Qwen3-0.6B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Additional flags for larger GPUs:

```console
--gpu-memory-utilization 0.92
--enable-chunked-prefill
--enable-prefix-caching
```

## Health check

```console
curl http://localhost:8000/health
```

## CI

The `.github/workflows/vllm-gpu-container.yaml` workflow builds, smoke
tests, and publishes the image on a GPU runner:

- **Push to `main`** (when `Containerfile` or the workflow changes) —
  builds, tests, and pushes both `:<model>` (e.g. `:Qwen3-0.6B`) and
  `:latest`.
- **Pull request** / **merge queue** — builds and tests only; nothing is
  pushed.
- **Manual (`workflow_dispatch`)** — accepts an `inference_model` input to
  build an arbitrary HuggingFace model; pushes `:<model>` but not
  `:latest`. Defaults to `Qwen/Qwen3-0.6B`.

The image tag is derived from the model ID with the org prefix stripped
(`Qwen/Qwen3-0.6B` → `Qwen3-0.6B`). Gated models are supported via the
`HF_TOKEN` repository secret.

## Dependabot

The `FROM` directive in `Containerfile` is monitored by Dependabot for
weekly base image updates (see `.github/dependabot.yaml`).
