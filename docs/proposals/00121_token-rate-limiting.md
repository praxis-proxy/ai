---
issue: https://github.com/praxis-proxy/ai/issues/121
discussion: https://github.com/praxis-proxy/ai/issues/121
status: proposed
authors:
  - shaneutt
graduation_criteria:
  - How? section with requirements and design
stakeholders:
  - leseb
  - mkoushni
  - crstrn13
  - eoinfennessy
  - alexsnaps
---

# Token Rate Limiting

## What?

Token rate limiting adds quota enforcement denominated
in tokens to Praxis. Request-count quotas cannot
express constraints like "this team may consume 1M
tokens per hour" because a single LLM call can vary
wildly in terms of token cost. This capability lets
operators define budgets in the unit that actually
drives inference cost.

The system requires reservation-based admission: requests
are admitted by reserving an estimated cost up front,
then the reservation is reconciled against actual usage
reported by the provider after the response completes.
The estimation method must be configurable because
different deployments have different cost models.

Different token types must be accounted for separately
with configurable weights. A cached input token does
not carry the same cost as an uncached token, and
quotas should reflect that difference. We also need
to support multiple modular strategies for what to do
about input token counting, output token counting,
estimation, what to do with unused tokens, etc.

### Goals

**Must have (MVP):**

- **[M1]** Bucketed `Application` concept which applies
  `TokenBudget` to a specific application (or, by default
  treats all requests as one `default` application.
- **[M1]** Reservation-based admission: admit requests
  by reserving an estimated token cost, then reconcile
  against actual provider-reported usage after the
  response.
- **[M2]** Configurable estimation: how estimated cost
  is calculated from request metadata must be
  operator-defined, not a single hard-coded formula.
- **[M3]** Token-type-aware accounting: different token
  types (input, output, cached, uncached) tracked
  separately with configurable weights so quotas
  reflect real cost differences.
- **[M4]** Flexible bucket keys: quotas keyed by
  request headers, model identity, or compound keys
  so different clients and models get independent
  budgets. (TBD - this one may need to be teased out more)
- **[M5]** Hard deny with 429 when a budget is
  exhausted, with standard rate limit response
  headers (`Retry-After`, `X-RateLimit-*`).
- **[M7]** Observability: metrics and tracing
  distinguishing admitted vs. limited requests,
  including estimated cost, actual cost, and
  remaining budget. The ability to log tokenomic
  results distinctly for accounting.
- **[M7]** Soft limits: usage tiers that modify request
  headers instead of rejecting, enabling downstream
  systems (e.g. llm-d `InferenceObjective`) to
  degrade gracefully as a client approaches its
  budget.
- **[M8]** Batch workload awareness: separate quota
  rules or deferred accounting for batch/async API
  patterns so bulk jobs do not starve interactive
  traffic.
- **[M9]** Exact token metering: an accounting path
  that records precise actual-usage counts for billing
  and chargeback, independent of rate-limit counters.
- **[M10]** Reservation refund on lost requests: release
  reserved capacity when a request times out, is
  dropped, or otherwise never completes.

### Non-Goals

- Replacing request-count rate limiting. Token and
  request-count quotas are independent concerns;
  operators may use both.
- Identity resolution. This capability assumes client
  identity has already been resolved to a request
  header by an upstream component. We can re-assess
  needs around this in later iterations.

These are non-goals for this _iteration_ but are otherwise
long term capabilities we do want.

- Usage-based dynamic rates: adjust quotas or
  refill rates based on observed usage patterns over
  a sliding window.
- Input message tokenization: fully tokenize request
  content before admission to use a real input token
  count rather than relying on estimations, `max_tokens`
  as a proxy, etc.

## Why?

### Motivation

AI inference is the first workload where request-count
rate limiting is meaningfully wrong. Inference cost
scales with token count, and a single request can vary
by orders of magnitude. An operator limiting a team to
100 req/min has no control over whether those requests
consume 1,000 tokens or 10,000,000.

Three realities shape the requirements:

1. **Precise token counts are _generally_ only available
   after the response.** Providers report usage in the
   response body or headers. By the time actual counts
   are known, the tokens have been consumed. Admission
   decisions must therefore rely on estimates, with
   reconciliation after the fact. This is why a
   reservation-based model is necessary rather than
   simple post-hoc accounting. (Caveat: in the future we
   may be able to eliminate this by running tokenizers
   early).

2. **Not all tokens cost the same.** Providers charge
   significantly less for cached input tokens than
   uncached (roughly 10x for Anthropic). A quota
   system that treats all tokens equally will
   over-restrict users who benefit from caching or
   under-restrict those who do not. Token-type-aware
   weighting is essential for fair quotas.

3. **AI workloads span clusters.** Enterprise
   deployments run multiple proxy instances across
   availability zones. Per-instance budgets cause
   effective quota to scale with instance count,
   the opposite of the intended control.

Beyond hard enforcement, operators need graduated
controls. When a team approaches its budget the
platform should be able to signal downstream
schedulers to route to cheaper models or lower-priority
queues, rather than rejecting outright. Hard deny is
the backstop, not the first line of defense.

### User Stories

- As a **platform operator**, I need per-team token
  budgets so shared inference infrastructure remains
  fair across consumers.

- As a **platform operator**, I need different models
  to carry different quota weights so budgets reflect
  actual inference cost.

- As a **platform operator**, I need cached tokens to
  consume less quota than uncached tokens so teams are
  not penalized for efficient caching.

- As a **platform operator**, I need quota state
  shared across proxy instances so teams cannot bypass
  limits by hitting different endpoints.

- As a **platform operator**, I need graduated
  controls that signal downstream systems at usage
  thresholds so the platform can degrade gracefully
  before hard deny.

- As a **FinOps engineer**, I need accurate per-team,
  per-model token consumption records for cost
  attribution and chargeback.

- As a **platform operator running batch workloads**,
  I need batch calls accounted for without starving
  real-time traffic.

## Open Questions

### Estimation configurability

The estimation method must be operator-defined.
What is the right level of flexibility? Named
strategies with parameters cover common patterns but
may be too rigid. A general expression language is
maximally flexible but adds complexity. The choice
affects usability and the range of cost models that
can be expressed.

### Soft limit tier model

We describe graduated controls via header
injection. How should tiers be defined? As a list of
usage thresholds with associated headers? Should
multiple tiers be active simultaneously (e.g. warning
at 80%, degraded at 95%, deny at 100%)?

### Batch workload accounting

We must handle batch workloads differently
from interactive traffic. Should batch jobs draw from
a separate budget, share the main budget with lower
priority, or use deferred accounting that settles
after completion?

### Lost request handling

Lost requests require releasing reservations for
requests that never complete. What conditions qualify
(timeout, connection reset, upstream 5xx)? How long
should a reservation be held before it is considered lost?
This needs to be accounted for in architecture, but
potentially out-of-scope for MVP.

[#210]: https://github.com/praxis-proxy/ai/issues/210
