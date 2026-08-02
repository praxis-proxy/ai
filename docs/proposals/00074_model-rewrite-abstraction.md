---
issue: https://github.com/praxis-proxy/ai/issues/74
discussion: https://github.com/praxis-proxy/praxis/discussions/838
status: proposed
authors:
  - usize
graduation_criteria:
  - What? and Why? accepted by stakeholders
  - How? section with requirements and design
stakeholders:
  - shaneutt
  - leseb
  - bentito
  - cnuland
  - caldeirav
  - franciscojavierarceo
---

# Model Rewrite Abstraction

## What?

Separate model selection from and the mechanisms necessary to route to a model in the
filter pipeline. These mechanisms include alias resolution, fact promotion to
headers and/or metadata, safety validation, and policy enforcement.

Today, any filter that wants to route a request to a different model must reimplement that machinery (or skip it). This proposal introduces a shared rewrite mechanism that any upstream filter can invoke by writing a model suggestion to filter
metadata, leaving the mutation and policy enforcement to dedicated downstream machinery.

### Goals

- Define an API contract between model-selecting
  filters and model-routing machinery.
- Provide a final policy gate before model rewriting
  so that misconfigured or buggy selectors cannot
  route requests to disallowed models.
- Expose policy configuration as a library to filter authors so
  selectors can make informed suggestions.

### Non-Goals

- Classification and selection strategy. Model selection is treated as a black box so
  operators can integrate whichever approach fits their needs. 
- Cross-request state. Managing state across the turns of a conversation (routing momentum, context dilution, session coherence) is out of scope. Selectors that need cross-request memory build on the state primitives from proposals #412 (storage layer) and #432 (request extensions).

## Why?

### Motivation

Mixture-of-models routing is a core goal of the AI
gateway (issue #74). Multiple filters will need to
select models: semantic routers, complexity
classifiers, cost-aware routers, latency-aware
routers, LoRA adapter routers, and more. Each of
these filters solves a different selection problem,
but all share the same downstream need: rewrite the
`model` field in the request body and enforce
organizational policy on which models are permitted.

There is no policy gate anywhere in Praxis today that can answer "is this
model allowed for this request?" PR #446 demonstrated the problem concretely. That
PR implemented a semantic router filter that could not enforce any model access policies. It circumvented the difficulties of model re-writing via cluster selection--introducing an assumption that a particular cluster represents a particular model.

Without a shared rewrite mechanism, every new
model-selecting filter must either:

1. Reimplement the selection-facing half of the
   rewrite pipeline (alias resolution, header and
   metadata promotion, safety validation, policy
   checks).
2. Bypass the model field entirely and select a
   cluster directly--conflating model identity
   with backend topology
3. Depend on the existing `ModelRewriteFilter`,
   which is Responses-API-specific and gates on `POST /v1/responses`.

None of these options scale to the routing
strategies enumerated in #74.

#### Why not Guardrails?

Guardrails filters can not close the
policy gap. Guardrails match request content
against configured patterns; they cannot evaluate
"is this model allowed for this request" at the
moment a model is chosen, and they cannot feed
policy back to a selector so it avoids suggesting
models that will be rejected. The gate belongs at
the rewrite decision point, and the policy
configuration must be consumable as a library by
upstream selectors.

### Prior Art

AI gateways and routing projects fall into three
architectural categories on this question. Where
the boundary lands depends on whether the system
mutates in-flight requests (as Praxis does) or
constructs fresh outbound requests.

**SDK-style gateways ([LiteLLM][litellm-arch],
[Portkey][portkey-plugins], [MLflow][mlflow-gw])**
do not separate selection from rewriting. These
projects construct fresh outbound HTTP requests
for each provider call, so there is no in-flight
body to mutate.

Portkey is a partial exception: its
`beforeRequestHooks` mechanism runs after target
selection but before the provider call, allowing
guardrail hooks to deny or transform requests.
This is the closest any SDK-style gateway comes
to an intervention point between selection and
execution.

**Recommendation-only projects ([Not Diamond][notdiamond],
[RouteLLM][routellm], [Aurelio Semantic Router][semantic-router])**
achieve total separation by scope limitation -- they
only select, never rewrite. Not Diamond returns a
model recommendation; the caller handles everything
else. RouteLLM is a binary strong/weak classifier
that delegates to LiteLLM for the actual API call.
Aurelio Semantic Router classifies intent via
embedding similarity and returns a route name. None
of these projects touch request bodies.

**Proxy-native projects ([Gateway API Inference
Extension][gie], [Envoy AI Gateway][envoy-ai-gw])**
are the direct analogues to Praxis, and both
implement explicit separation between selection
and rewriting:

- The Gateway API Inference Extension (GIE)
  decomposes the problem into four distinct
  components: a Body-Based Router that extracts
  the model name and promotes it to a header (pure
  extraction, no mutation), HTTPRoute rules that
  route based on that header, an Endpoint Picker
  that selects a specific pod based on runtime
  metrics (communicating its choice via the
  `x-gateway-destination-endpoint` header), and
  InferenceModelRewrite that handles model name
  rewriting in the body as a [separate CRD][gie-rewrite].

- The Envoy AI Gateway uses [two ext_proc
  phases][envoy-ai-gw-dataplane]: a router-level
  phase that extracts the model name and sets
  `x-ai-eg-model` for routing, and an
  upstream-level phase that handles schema
  translation, credential injection, and [model
  name override][envoy-ai-gw-vnm] via the
  `modelNameOverride` field on `AIGatewayRoute`
  backend refs.

The suggest-then-apply pattern -- where one
component suggests a model or endpoint via
metadata and a separate component applies the
mutation -- is the established architecture in the
proxy-native space. It exists because proxies that
mutate in-flight requests face real per-site
engineering costs (JSON parse, re-serialize,
content-length fixup, header promotion) that
SDK-style gateways avoid by constructing fresh
requests.

### Policy: Gate and Library

A recurring concern with separating selection from
rewriting is that the selector seemingly *needs*
to know the policy: if requests flagged with PII
must not reach external models, doesn't the thing
picking the model need to know that? Established
policy systems answer with a consistent split:
enforce authoritatively at a single gate, and
expose the same policy to callers as a library or
discovery API so they can make informed choices.
Correctness never depends on the caller's
cooperation; caller awareness exists to avoid
wasted work.

- **[Kubernetes admission control][k8s-admission]**
  runs mutating webhooks first and validating
  webhooks only after all mutation completes, so
  policy always evaluates the final object no
  matter which component mutated it. Clients can
  pre-flight a check through the
  `SelfSubjectAccessReview` API
  ([`kubectl auth can-i`][k8s-authz]);
  enforcement at the API server is unaffected by
  whether clients bother.
- **[The Kubernetes scheduler][k8s-scheduler]**
  separates hard constraints from preferences:
  filter plugins eliminate nodes that cannot run a
  pod, then score plugins rank the survivors. Any
  scorer composes with any filter set because
  constraint satisfaction is not the scorer's job.
  Under this proposal, model selection is a
  preference; model policy is a constraint.
- **Open Policy Agent** runs the same policy
  document in both positions: [embedded as a
  library][opa-integration] so callers evaluate
  decisions locally (including [partial
  evaluation][opa-partial-eval] to derive the set
  of permitted options up front), and deployed as
  a boundary enforcement point ([Envoy
  `ext_authz`][opa-envoy], Kubernetes admission).
  Selector and gate cannot drift apart because
  they share one source of truth.
- **LiteLLM's proxy** enforces key- and
  team-level [model access groups][litellm-access]
  at call time and exposes the same policy through
  model discovery (`/models`), so a caller can
  enumerate what it may use before choosing.

The selector does not need to know the policy for
correctness -- the gate guarantees that -- but it
should consult the policy for quality, since
rejected suggestions waste requests. That is why
this proposal treats the gate and the library as
one deliverable rather than trusting selectors to
get policy right on their own.

### User Stories

- As a filter author building a semantic router, I
  want to suggest a model for a request without
  reimplementing alias resolution, header and
  metadata promotion, and policy checks so that I
  can focus on classification logic.
- As a platform engineer, I want a single policy
  gate that enforces which models are permitted for
  a request regardless of which upstream filter
  selected the model, so that organizational access
  controls cannot be bypassed by a misconfigured
  classifier.
- As a proxy operator deploying multiple
  model-selecting strategies (semantic routing,
  cost-aware routing, complexity classification), I
  want these strategies to compose with a single
  rewrite mechanism so that I do not need to
  configure body mutation independently for each
  one.
- As a filter author, I want access to the policy
  configuration (which models are allowed for which
  users, groups, or request properties) as a
  library so that my selector can make informed
  suggestions rather than suggesting models that
  will be rejected downstream.

## How?

> **Note:** do not include this section in the first PR.
> Submit What? and Why? first. Add How? in a follow-up
> PR after the proposal direction is accepted.

<!-- reference links -->

[litellm-arch]: https://docs.litellm.ai/docs/proxy/architecture
[portkey-plugins]: https://github.com/Portkey-AI/gateway/blob/main/plugins/README.md
[mlflow-gw]: https://mlflow.org/docs/latest/genai/governance/ai-gateway/
[notdiamond]: https://docs.notdiamond.ai/docs/quickstart-routing
[routellm]: https://github.com/lm-sys/RouteLLM
[semantic-router]: https://github.com/aurelio-labs/semantic-router
[gie]: https://gateway-api-inference-extension.sigs.k8s.io/
[gie-rewrite]: https://gateway-api-inference-extension.sigs.k8s.io/guides/adapter-rollout/
[envoy-ai-gw]: https://aigateway.envoyproxy.io/docs/concepts/architecture/
[envoy-ai-gw-dataplane]: https://aigateway.envoyproxy.io/docs/concepts/architecture/data-plane/
[envoy-ai-gw-vnm]: https://aigateway.envoyproxy.io/docs/capabilities/traffic/model-name-virtualization/
[k8s-admission]: https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/
[k8s-authz]: https://kubernetes.io/docs/reference/access-authn-authz/authorization/
[k8s-scheduler]: https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/
[opa-integration]: https://www.openpolicyagent.org/docs/integration
[opa-partial-eval]: https://www.openpolicyagent.org/docs/filtering/partial-evaluation
[opa-envoy]: https://www.openpolicyagent.org/docs/envoy
[litellm-access]: https://docs.litellm.ai/docs/proxy/model_access
