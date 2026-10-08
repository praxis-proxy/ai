## Changelog

- Example change

<!-- praxis:native-vllm-qualification:start -->
### vLLM Responses gateway qualification

vLLM Responses gateway qualification: **passed**. native and translation cases completed

Tested checkout: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`; locally built gateway binary `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` (debug/full); Praxis core `0.7.2`.
Backend: vLLM `0.30.0`, local image `sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`; model `Qwen/Qwen3-8B` (revision `unavailable`).
SDK: OpenAI Python `2.9.0`; config `examples/configs/openai/responses/full-flow-agentic.yaml` (`dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd`); storage `postgresql`.
Native Responses: passed; selected cases: passed 1, failed 0, skipped 0, xfailed 0, xpassed 0, unexecuted 0.
Responses-to-Chat translation: passed; selected cases: passed 1, failed 0, skipped 0, xfailed 0, xpassed 0, unexecuted 0.
Supporting: passed; selected cases: passed 1, failed 0, skipped 0, xfailed 0, xpassed 0, unexecuted 0.
Credentialed Tools: skipped.
Claude Acceptance: success.
Codex Acceptance: skipped.

[Workflow run](https://github.com/praxis-proxy/ai/actions/runs/17) and [report/diagnostic artifacts](https://github.com/praxis-proxy/ai/actions/runs/17#artifacts)

Limitations: selected text and tool behavior over native Responses and Responses-to-Chat HTTP/SSE paths only; [streamed Conversation append](https://github.com/praxis-proxy/ai/issues/410) and multimodal/background/WebSocket paths are excluded. See the attached `qualification.json` for case reasons and separate client/tool results.
<!-- praxis:native-vllm-qualification:end -->
