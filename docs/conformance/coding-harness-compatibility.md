# Coding harness compatibility

Can a developer edit a file through Praxis on a self-hosted model? The November 25, 2026 upstream nightly ran these clients against vLLM on an NVIDIA A10G. Each result below applies to the client, route, configuration and Praxis build identified on its card. [GPU run 45546536176][run] · [Tested Praxis build 55e32ba][build]

| Harness | Client API | Native vLLM route | Translated vLLM route |
| --- | --- | --- | --- |
| [Codex CLI 0.170.0](#codex-cli) | Responses over HTTP/SSE | Verified file edit | Verified file edit via Chat Completions |
| [Claude Code 2.1.278](#claude-code) | Anthropic Messages | Edits in acceptEdits and auto mode | Same two modes via Chat Completions |

**Reading a result.** **Supported** means the recorded configuration passed a real-client GPU or release-cluster scenario for that capability. **Supported with limitations** means it passed within the stated boundary. **Not supported** requires an observed failure on that configuration. **Not validated** means the cited evidence leaves the outcome open. The GPU run used the CLIs directly. OpenShell deployment remains unvalidated.

## Codex CLI

**Tested configuration.** The two Codex edit scenarios and the native compaction scenario used a custom Responses provider over HTTP/SSE with full-access sandbox mode and `web_search = "disabled"`. Both edit scenarios selected Codex's `gpt-5.5` tool catalog and rewrote the inference model to `Qwen/Qwen3-8B` (served as `qwen3-8b`). vLLM 0.32.0 used the `hermes` tool parser and `deepseek_r1` reasoning parser on EC2 g5.xlarge. The report records the image digest and no resolved weights revision. The gateway runs `openai_stream_events` ahead of `openai_client_tool_compat`. See the tested [native configuration][codex-native], [translated configuration][codex-chat] and [run artifact][run] for the ordered filter chains and runtime overrides.

| Developer capability | Native Responses | Responses → Chat | What the CLI demonstrated or what remains open |
| --- | --- | --- | --- |
| Start a session and receive a usable tool call | Supported | Supported | The pinned CLI starts the edit with its grammar-based patch tool. Exhaustive first-request coverage of namespace, shell, tool search and developer-configured MCP declarations together is unvalidated. |
| Apply a patch, send the tool result and run a shell verifier | Supported | Supported | Codex created the exact expected file, sent the patch output through Praxis and verified the workspace. This exercises a client-owned tool loop. |
| Replay full history with `store:false`; obey developer instructions across turns | Not validated | Not validated | The edit includes multiple exchanges. Independent assertions on replayed history and instruction retention are needed for these claims. |
| Return reasoning usage on an edit | Supported | Not validated | The native edit requires positive reasoning-token usage on a completed turn. |
| Trigger client self-compaction | Supported | Not validated | The observer saw the client's inline summarization request cross Praxis. |
| Continue sending requests after compaction | Supported | Not validated | Native requires continued traffic with compacted history. |
| Finish and verify the task after compaction | Not validated | Not validated | On this Qwen3-8B nightly the native test permits skipping the final file and verifier assertions. |
| Use stored continuation with `previous_response_id` | Not validated | Not validated | The [Responses parameter table](openai-responses-state-conformance.md#responses-support-and-qualification) covers gateway storage and SDK evidence. |
| Invoke developer-configured MCP; edit inside OpenShell | Not validated | Not validated | Each needs its own client scenario: consume the MCP result, or run the edit and verifier inside the sandbox. |
| Hosted web search from Codex | Not validated | Not validated | This baseline predates the dedicated Codex search cases. New runs record native and translated results independently and require a final answer citing a URL returned by Tavily. The [Claude Code](#claude-code) card covers its built-in WebSearch tool. |
| Image input or WebSocket sessions | Not validated | Not validated | The edit used text and HTTP. The vision workflow exercises SDK image input. |

## Claude Code

**Tested configuration.** The pinned Linux CLI ran directly on the GPU runner against the same Qwen checkpoint and vLLM 0.32.0 image version. Claude used `hermes` for tool calls and `qwen3` for reasoning. The weights revision was unspecified. Both routes accept Messages at the gateway. The [pin manifest][claude-pin] records the executable, image digest, launch flags and chat template settings. See the tested [native Messages configuration][claude-native] and [Messages-to-Chat configuration][claude-chat].

| Developer capability | Native Messages | Messages → Chat | What the CLI demonstrated or what remains open |
| --- | --- | --- | --- |
| Complete the coding task in acceptEdits mode | Supported | Supported | Both edit scenarios completed under the deterministic permission mode. |
| Complete the coding task in auto mode | Supported | Supported | Both client-initiated auto-mode scenarios completed. |
| Finish read-only planning | Supported | Not validated | The native scenario requires convergence and keeps reasoning-channel text out of the visible answer. |
| Trigger client self-compaction | Supported | Not validated | The CLI emitted its auto-compaction boundary. A successful pre-compaction read carrying the marker traversed Praxis. |
| Make a tool call after compaction | Not validated | Not validated | On Qwen3-8B the test checks ordering if the model emits a call; a missing call is an expected limitation. The report does not establish it occurred in this run. |
| Complete the task after compaction | Not validated | Not validated | The Qwen3-8B scenario permits skipping the post-compaction file verification. |
| Resolve built-in WebSearch and deliver a usable final answer | Not validated | Supported with limitations | The managed Tavily scenario checks any observed exchange and tool suppression. Qwen3-8B can omit the call or finish without a usable answer. |
| Enforce a total-token limit | Not validated | Not validated | Requires a Claude session whose input plus output usage triggers the configured limit. |
| Authenticate with bearer and x-api-key credentials | Not validated | Not validated | Requires successful Claude requests under each credential form independently. |
| Handle provider beta headers | Not validated | Not validated | Requires capturing the CLI's anthropic-beta header and the selected provider's response. |

## Faults, accounting and deployment

These outcomes are **Not validated** by the cited harness baseline for either client. A deterministic fault test can supplement a qualified configuration when it uses the same pinned CLI and gateway build and observes the client-visible outcome.

| Capability | Evidence needed |
| --- | --- |
| Normal stream completion | Observe the protocol's terminal event and successful client exit. |
| Failed stream or provider error | Inject 4xx, 5xx and mid-stream failure. Record the CLI-visible code, terminal behavior and time to exit. Include context-length, model and quota errors. |
| Interrupted or idle stream | Cancel or stall a request. Verify bounded termination and the configured idle timeout. |
| Usage accounting | Compare gateway-metered usage with the known provider usage separately for completed, failed and interrupted calls. |
| Oversized requests | Exceed the configured body limit and capture the error the CLI receives. |
| Guardrails and isolation | Exercise allow, block and guardrail failure; check that separate API keys cannot retrieve each other's stored responses. |
| Release and version drift | Run the pinned CLI and checkpoint against the Praxis release build, record direct and OpenShell results separately, and track latest-stable CLI runs separately from pinned results. |

## Read the evidence for your build

The [upstream run][run] publishes qualification.json. Inspect the individual cases under suites.codex_acceptance and suites.claude_acceptance. Codex case capabilities and provenance identify the executable version, model, parsers, image and gateway filters. Claude scenarios use step outcomes and the [manifest at the tested commit][claude-pin]. For a Praxis release, use its attached support-report.md and support-report.json and check the live-validation commit. The [qualification guide](../developing/vllm-qualification.md) explains ancestor evidence; the [GPU inventory](../developing/gpu-nightly-suite.md) covers OpenCode, pi and SDK lanes.

[run]: https://github.com/praxis-proxy/ai/actions/runs/45546536176
[build]: https://github.com/praxis-proxy/ai/tree/55e32bae0ed0f6e4b7e1a8cf38a8ad033be9ac19
[claude-pin]: https://github.com/praxis-proxy/ai/blob/55e32bae0ed0f6e4b7e1a8cf38a8ad033be9ac19/tests/integration/fixtures/claude-code-cli/pin.toml
[codex-native]: https://github.com/praxis-proxy/ai/blob/55e32bae0ed0f6e4b7e1a8cf38a8ad033be9ac19/examples/configs/openai/responses/client-tool-compat.yaml
[codex-chat]: https://github.com/praxis-proxy/ai/blob/55e32bae0ed0f6e4b7e1a8cf38a8ad033be9ac19/examples/configs/openai/responses/client-tool-compat-chat-completions.yaml
[claude-native]: https://github.com/praxis-proxy/ai/blob/55e32bae0ed0f6e4b7e1a8cf38a8ad033be9ac19/examples/configs/anthropic/messages-native-vllm.yaml
[claude-chat]: https://github.com/praxis-proxy/ai/blob/55e32bae0ed0f6e4b7e1a8cf38a8ad033be9ac19/examples/configs/anthropic/messages-to-openai-vllm.yaml
