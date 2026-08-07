---
issue: https://github.com/praxis-proxy/ai/issues/121
discussion: https://github.com/praxis-proxy/ai/issues/121
status: proposed
authors:
  - shaneutt
graduation_criteria:
  - How? section with requirements and design
stakeholders:
  - jland-redhat
  - leseb
  - mkoushni
  - crstrn13
  - eoinfennessy
  - alexsnaps
---

# Tokenomics + Token Rate Limiting

## What?

Token rate limiting adds quota enforcement denominated
in tokens to Praxis. Request-count quotas cannot
express constraints like "this team may consume 1M
tokens per hour" because a single LLM call can vary
wildly in terms of token cost. This capability lets
operators define budgets in the unit that actually
drives inference cost.

The system requires reservation-based admission:
requests are admitted by reserving an estimated cost
up front, then the reservation is reconciled against
actual usage reported by the provider after the
response completes. The estimation method must be
configurable because different deployments have
different cost models.

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
  a `TokenBudget` to a specific application (or, by
  default, treats all requests as one `default`
  application).
- **[M2]** Reservation-based admission: admit requests
  by reserving an estimated token cost, then reconcile
  against actual provider-reported usage after the
  response.
- **[M3]** Configurable estimation: how estimated cost
  is calculated from request metadata must be
  operator-defined, not a single hard-coded formula.
- **[M4]** Token-type-aware accounting: different token
  types (input, output, cached, uncached) tracked
  separately with configurable weights so quotas
  reflect real cost differences.
- **[M5]** Flexible bucket keys: quotas keyed by
  request headers, model identity, or compound keys
  so different clients and models get independent
  budgets. (TBD - this one may need to be teased
  out more)
- **[M6]** Hard deny with 429 when a budget is
  exhausted, with standard rate limit response
  headers (`Retry-After`, `X-RateLimit-*`).
- **[M7]** Observability: metrics and tracing
  distinguishing admitted vs. limited requests,
  including estimated cost, actual cost, and
  remaining budget. The ability to log tokenomic
  results distinctly for accounting.
- **[M8]** Soft limits: usage tiers that modify request
  headers instead of rejecting, enabling downstream
  systems (e.g. llm-d `InferenceObjective`) to
  degrade gracefully as a client approaches its
  budget.
- **[M9]** Batch workload awareness: separate quota
  rules or deferred accounting for batch/async API
  patterns so bulk jobs do not starve interactive
  traffic.
- **[M10]** Exact token metering: an accounting path
  that records precise actual-usage counts for billing
  and chargeback, independent of rate-limit counters.
- **[M11]** Reservation refund on lost requests: release
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

These are non-goals for this _iteration_ but are
otherwise long term capabilities we do want.

- Usage-based dynamic rates: adjust quotas or
  refill rates based on observed usage patterns over
  a sliding window.
- Input message tokenization: fully tokenize request
  content before admission to use a real input token
  count rather than relying on estimations,
  `max_tokens` as a proxy, etc.

## Why?

### Motivation

AI inference is the first workload where request-count
rate limiting is meaningfully wrong. Inference cost
scales with token count, and a single request can vary
by orders of magnitude. An operator limiting a team to
100 req/min has no control over whether those requests
consume 1,000 tokens or 10,000,000.

Three realities shape the requirements:

1. **Precise token counts are _generally_ only
   available after the response.** Providers report
   usage in the response body or headers. By the time
   actual counts are known, the tokens have been
   consumed. Admission decisions must therefore rely
   on estimates, with reconciliation after the fact.
   This is why a reservation-based model is necessary
   rather than simple post-hoc accounting. (Caveat: in
   the future we may be able to eliminate this by
   running tokenizers early).

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

## How?

### Requirements

- Token budgets assigned per application with
  configurable matching rules
- Pluggable estimation strategies for request-time
  cost prediction
- Per-type token weights applied during
  reconciliation when actual usage by type is known
- Graduated soft limit tiers with header injection
  before hard deny
- Separate accounting for batch vs. interactive
  workloads
- Exact metering records independent of rate limit
  counters
- Reservation cleanup on request failure or timeout

### Design

#### Application and Token Budget

An `Application` groups requests by matching rules
(headers, model, path) and binds them to a
`TokenBudget`. Unmatched requests fall to a `default`
application. Each application maintains its own
independent budget.

```yaml
filter: token_rate_limit
applications:
  - name: team-alpha
    match:
      headers:
        x-api-key: team-alpha-key
    token_budget:
      capacity: 1_000_000
      window: 1h

  - name: default
    token_budget:
      capacity: 100_000
      window: 1h
```

#### Threshold Modes

When an application's budget is under pressure, two
modes control what happens to the request:

- **Hard ceiling**: cap `max_tokens` on the request to
  the application's remaining budget. The model
  generates at most what the budget can afford.
- **Overcharge**: allow the full `max_tokens`. The
  request proceeds at its original size, but any
  usage beyond the budget is tracked as overcharge
  and reported.

Hard deny (429) is a separate control that applies
when the budget is fully exhausted and no overcharge
is permitted.

#### Request Lifecycle

Each request passes through five phases:

1. **Admission** - Match the request to an
   application, compute an estimated cost using the
   configured strategy, and check the budget. If the
   budget has soft limit tiers, inject the appropriate
   tier headers. If the budget is exhausted and
   overcharge is not permitted, reject with 429.
   Estimation at admission can be optionally disabled
   in favor of response-only accounting.

2. **Threshold check** - Check the application's
   remaining budget. In **hard ceiling** mode, cap
   `max_tokens` to the remaining budget. In
   **overcharge** mode, leave `max_tokens` at the
   operator-configured default.

3. **Inference** - The request is forwarded upstream
   with the (possibly adjusted) `max_tokens`. The
   provider performs inference and returns token
   usage.

4. **Reconciliation** - Actual provider-reported token
   usage replaces the estimate. If overcharge mode is
   active, the overage is calculated and reported.
   The difference between estimated and actual cost is
   logged. (In future iterations, we may refund the
   estimate-vs-actual difference back to the budget;
   for MVP, budgets consume at the estimated rate.)

5. **Cleanup** - If a request is lost (timeout,
   connection reset, upstream failure), release the
   reservation after a configurable hold period.

#### Estimation Strategies

Estimation is pluggable. Each application selects a
named strategy with parameters. Strategies operate
on request metadata available before forwarding
(e.g. `max_tokens`, model identity, content length).

Built-in strategies:

| Strategy | Basis | Use case |
|---|---|---|
| `max_tokens` | `max_tokens` field | Simple upper bound |
| `input_plus_max_tokens` | content size + `max_tokens` | Conservative full-cost |
| `fixed` | constant per request | Uniform cost model |
| `model_scaled` | `max_tokens` * model multiplier | Model-aware budgets |

```yaml
token_budget:
  capacity: 1_000_000
  window: 1h
  estimation:
    strategy: input_plus_max_tokens
    multiplier: 1.2
```

The fixed set of named strategies covers common
patterns. Additional strategies can be added as
requirements emerge without changing the
configuration surface.

#### Token Type Accounting

Token weights are multipliers applied during
reconciliation when the provider reports actual usage
broken down by type. All types default to 1.0. A
weight below 1.0 means that token type consumes
proportionally less budget.

```yaml
token_budget:
  capacity: 1_000_000
  window: 1h
  weights:
    input: 1.0
    output: 1.0
    cached_input: 0.1
    cached_output: 0.1
```

Weighted cost at reconciliation:

```
cost = (input * w_input)
     + (output * w_output)
     + (cached_input * w_cached_input)
     + (cached_output * w_cached_output)
```

At admission, the estimation strategy produces a
single cost number (it does not know the per-type
breakdown yet). Weights take effect at reconciliation
when the provider reports actual counts by type.

#### Soft Limit Tiers

Graduated thresholds inject headers as usage
approaches budget capacity. Multiple tiers can be
active simultaneously. When overcharge is not
permitted, hard deny (429) is the final tier at 100%.

```yaml
token_budget:
  capacity: 1_000_000
  window: 1h
  soft_limits:
    - at_percent: 80
      inject:
        header: X-Token-Tier
        value: warning
    - at_percent: 95
      inject:
        header: X-Token-Tier
        value: degraded
```

Downstream systems (e.g. llm-d) read these headers
to adjust scheduling or priority without the proxy
needing to understand the downstream API.

#### Batch Workloads

Batch workloads can be a specific `Application`,
therefore getting their own budget, or we can provide
sub-application support within a single
`Application`, sharing the budget but with
(optionally) different rules.

```yaml
applications:
  - name: team-alpha
    match:
      headers:
        x-api-key: team-alpha-key
    token_budget:
      capacity: 1_000_000
      window: 1h
    batch:
      match:
        path_prefix: /v1/batches
      token_budget:
        capacity: 5_000_000
        window: 24h
```

When no separate batch budget is configured, batch
traffic shares the application's main budget.

> **Note**: Sub-applications can also be used
> entirely for accounting purposes when no separate
> budget is provided.

#### Metering

Rate limiting and metering serve different purposes.
Rate limiting enforces quotas using estimates and
approximations. Metering records exact actual usage
for billing and chargeback.

The metering path emits records after reconciliation
containing: application name, model, exact token
counts by type (from the provider), weighted cost,
and timestamp. These records are independent of rate
limit counters and can be consumed by external
billing systems.

#### Observability

The system emits:

- **Metrics**: tokens reserved, reconciled, and
  refunded; budget remaining; requests admitted vs.
  denied; soft limit tier activations; overcharge
  amounts
- **Tracing**: per-request spans with estimated cost,
  actual cost, application, and admission decision
- **Accounting logs**: structured records at a
  dedicated log target for tokenomic auditing,
  separate from operational logs

## Open Questions

### Estimation configurability

The estimation method must be operator-defined ([M3]).
What is the right level of flexibility? Named
strategies with parameters cover common patterns but
may be too rigid. A general expression language is
maximally flexible but adds complexity.

### Estimate reconciliation

Should the estimate-vs-actual difference be refunded
to the budget at MVP, or is logging the difference
sufficient initially? Refunding makes budgets more
accurate but adds complexity to the token accounting
lifecycle.

### Batch workload accounting

Should batch workloads be modeled as separate
applications with independent budgets, as
sub-applications sharing a parent budget with
different rules, or should both options be available?

### Lost request handling

What conditions qualify a request as lost (timeout,
connection reset, upstream 5xx)? How long should a
reservation be held before it is considered lost?

[#210]: https://github.com/praxis-proxy/ai/issues/210
