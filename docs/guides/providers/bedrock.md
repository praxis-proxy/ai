# Amazon Bedrock Converse from Chat Completions

Translate OpenAI Chat Completions requests to the Amazon Bedrock Converse API
and sign them with AWS SigV4.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow),
which also has the validate and start commands for `provider-local.yaml`.

## Configure, run, and verify

Build the standard binary and copy the full [Bedrock example](../../../examples/configs/bedrock/chat-completions-to-converse.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/bedrock/chat-completions-to-converse.yaml provider-local.yaml
export AWS_ACCESS_KEY_ID='<authorized-access-key-id>'
export AWS_SECRET_ACCESS_KEY='<secret-access-key>'
```

Pick a Bedrock Runtime region where your model is available. In the copy, bind
the listener to loopback, set the signer's `region`, and set its `host` to
`bedrock-runtime.<region>.amazonaws.com`. Replace the local cluster endpoint
with that hostname on port 443 and add a matching `tls.sni`. Keep the
translator before routing and the signer after load balancing, as in the
example, so the signature covers the translated body and path. Remove
`insecure_options` for the public endpoint.

The signer reads only the environment variables you configure, not the ambient
AWS credential chain. If your credentials include a session token, set
`session_token_env_var: AWS_SESSION_TOKEN` on `aws_sigv4_sign` and export
`AWS_SESSION_TOKEN='<session-token>'` in the proxy shell. See the
[signer reference](../../filters/providers/aws_sigv4_sign.md); scope the
credentials to the intended service and refresh them before they expire.

Validate and start, then send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<available-bedrock-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions JSON mapped from Converse. With `"stream":true` and
`curl -N`, the adapter decodes AWS EventStream frames into Chat SSE ending in
`[DONE]`. Gaps in reasoning, cache accounting, and history are listed in the
[matrix](../../conformance/translation-matrix.md). The translator has no
generated reference page yet because of a
[documentation tooling gap](../../developing/documentation-audit.md#remaining-gaps-and-separate-blockers);
the filter itself exists.

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```

Then finish the [shared cleanup](README.md#cleanup-and-qualification).

## Filter reference

- `openai_chat_completions_to_bedrock_converse` (no generated page yet; see the
  [example configuration](../../../examples/configs/bedrock/chat-completions-to-converse.yaml))
- [`aws_sigv4_sign`](../../filters/providers/aws_sigv4_sign.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
