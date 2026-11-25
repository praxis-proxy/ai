# Amazon Bedrock Converse from Chat Completions

Translate OpenAI Chat Completions requests to the Amazon Bedrock Converse API
and sign them with AWS SigV4.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow)
for provider guides: account access, credentials, the validate/start commands
for `provider-local.yaml`, and how to read provider errors.

## Configure, run, and verify

Build standard and copy the full [Bedrock example](../../../examples/configs/bedrock/chat-completions-to-converse.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/bedrock/chat-completions-to-converse.yaml provider-local.yaml
export AWS_ACCESS_KEY_ID='<authorized-access-key-id>'
export AWS_SECRET_ACCESS_KEY='<secret-access-key>'
```

Choose a Bedrock Runtime region with access to your requested model. Edit the
listener to loopback, set the signer's `region`, and replace its `host` with
`bedrock-runtime.<region>.amazonaws.com`. Replace the local cluster endpoint
with that same hostname on port 443 and add matching `tls.sni`. Keep the
translator before routing and the signer after load balancing as in the full
example, so it signs the translated body/path. Remove local `insecure_options`
for a public endpoint.

The signer reads explicitly configured environment sources; it does not use
an ambient AWS credential chain. If your credentials include a session token,
set `session_token_env_var: AWS_SESSION_TOKEN` on `aws_sigv4_sign` and export
`AWS_SESSION_TOKEN='<session-token>'` in the proxy shell. Follow the
[signer reference](../../filters/providers/aws_sigv4_sign.md), keep credentials limited to
the intended service, and refresh them before expiry. Signing alone does not
implement Bedrock protocol translation.

Validate/start, then send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<available-bedrock-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions JSON, mapped from Converse. With `"stream":true`
and `curl -N`, the adapter decodes AWS EventStream frames into Chat SSE and a
terminal `[DONE]`. Reasoning, cache accounting, and history gaps are documented
in the [matrix](../../conformance/translation-matrix.md). The separate
[generated-reference omission](../../developing/documentation-audit.md#remaining-gaps-and-separate-blockers)
is a documentation tooling blocker, not proof of absent implementation.

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```

Delete your disposable `provider-local.yaml`, and revoke credentials created
only for testing according to your account policy. See
[cleanup and qualification](README.md#cleanup-and-qualification) for what these
steps do and do not establish.

## Filter reference

- `openai_chat_completions_to_bedrock_converse` (no generated page yet; see the
  [example configuration](../../../examples/configs/bedrock/chat-completions-to-converse.yaml))
- [`aws_sigv4_sign`](../../filters/providers/aws_sigv4_sign.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
