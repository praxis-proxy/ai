# Contributing

Thank you for your interest in contributing! Start by
reading the [development conventions]. Submissions that
do not follow the conventions will be rejected.

[development conventions]: https://github.com/praxis-proxy/conventions/blob/main/docs/conventions.md

## Getting Started

1. Fork the repository and clone your fork
2. Install pre-commit hooks: `make setup-hooks`
3. Build and test: `make build && make test`
4. Run every gate locally before pushing: `make all`

Requirements are listed in the [development setup guide].

[development setup guide]: docs/developing/getting-started.md#requirements

`make lint-clippy` and `make fmt` refresh the lint toolchains before running,
matching CI: stable comes from `rust-toolchain.toml`, and rustfmt uses the
current nightly. Clippy runs through `rustup` so another `cargo` on `PATH`
cannot select a different compiler. Refresh requires network access; a failed
refresh stops the target rather than accepting results from an old nightly.

## Picking Up an Issue

Only issues a maintainer has triaged (given a milestone
and added to a project board) are open for contributors
to take, and only at `Medium` or `Low` priority. Urgent
and high-priority work is assigned by maintainers. If you
self-assign something outside these rules, a bot unassigns
it and points you back here. See [Picking Up Work] for the
full policy.

[Picking Up Work]: docs/developing/project-management.md#bot-generated-issues

## Larger Changes

Features that span multiple PRs, introduce new
architectural patterns, or affect the public interface
go through the [proposal process].

[proposal process]: https://github.com/praxis-proxy/enhancements/blob/main/docs/process.md

## Pull Request Gates

CI enforces reviewability on every PR:

- A maximum added lines count of production code (tests, docs, examples excluded)
- No more than three open PRs per contributor in this repository. Drafts count.
  The Praxis bot comments on and fails a PR that takes its author over the
  limit and skips automatic reviewer assignment; it will receive review
  attention once enough other PRs are merged to bring the author's open count
  back to three or fewer. Edit or update the PR to refresh the check.
- A link to the issue the PR addresses, using `Closes #123` when the PR
  completes the issue or `Refs #123` when the issue remains open.
- A description with non-empty **Why**, **What**, and **How** sections. Explain
  why the change is needed, what problem it fixes, and how the implementation
  addresses it. The Praxis bot check fails when a section is missing or empty,
  or when the PR has no issue reference.
  PRs that add a filter parameter must also justify why the parameter is
  needed, what behavior it controls, and why existing configuration is
  insufficient.
- Maintainer review starts only after all required CI checks pass. A maintainer
  assignment signals that review has started; contributors should wait for CI
  to be green before requesting or expecting that assignment.
- A `Signed-off-by`
  trailer on every commit (`git commit -s`)
- Cryptographically signed commits (GPG or SSH)
- Human authorship: commits authored or signed-off by tools are rejected
- Conventional commit subjects (`type(scope): summary`, ≤72 chars)

See the [PR conventions] section for details and override labels.

[PR conventions]: https://github.com/praxis-proxy/conventions/blob/main/docs/conventions.md#pull-request-conventions
