# vLLM GPU Container

Pre-built [vLLM](https://docs.vllm.ai/) GPU container image with an
inference model baked in at build time, eliminating runtime download
latency.

Published to `ghcr.io/praxis-proxy/vllm-gpu`.

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

## Dependabot

The `FROM` directive in `Containerfile` is monitored by Dependabot for
weekly base image updates (see `.github/dependabot.yaml`).
