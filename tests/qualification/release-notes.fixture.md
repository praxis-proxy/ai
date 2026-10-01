## Changelog

- Example change

<!-- praxis:native-vllm-qualification:start -->
### Native vLLM qualification

Native vLLM qualification: **passed**. selected cases completed

Tested checkout: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`; locally built gateway binary `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` (debug/full); Praxis core `0.7.2`.
Backend: vLLM `0.30.0`, local image `sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`; model `Qwen/Qwen3-8B` (revision `unavailable`).
SDK: OpenAI Python `2.9.0`; config `examples/configs/openai/responses/full-flow-agentic.yaml` (`dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd`); storage `postgresql`.
Native selected cases: passed 2, failed 0, skipped 0, xfailed 1, xpassed 0, unexecuted 0.
Credentialed Tools: skipped.
Claude Acceptance: success.
Codex Acceptance: skipped.

[Workflow run](https://github.com/praxis-proxy/ai/actions/runs/17) and [report/diagnostic artifacts](https://github.com/praxis-proxy/ai/actions/runs/17#artifacts)

Limitations: selected native text HTTP/SSE behavior only; [streamed Conversation append](https://github.com/praxis-proxy/ai/issues/410) and multimodal/background/WebSocket paths are excluded. See the attached `qualification.json` for case reasons and separate client/tool results.
<!-- praxis:native-vllm-qualification:end -->
