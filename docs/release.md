# Release Process

## Versioning

Praxis AI uses [Semantic Versioning][semver]. The
workspace version is the single source of truth, defined
in `workspace.package.version` in the root `Cargo.toml`.
All workspace crates inherit this version.

[semver]: https://semver.org/

Releases before `1.0.0` and versions carrying a SemVer pre-release suffix
(such as `1.0.0-rc.1`) are marked as **pre-release** on GitHub. The API may
still change until the first stable major release, and GitHub's "Latest"
release must not advertise an unstable build. Stable releases from `1.0.0`
onward are published normally.

## Distribution

The supported release artifact is the Praxis AI container image on GHCR.
Workspace crates are implementation packages and are not published to
crates.io independently.

## Pre-release Checklist

Before tagging a release:

- [ ] Lints are clean (`make lint`)
- [ ] All tests pass locally (`make test`)
- [ ] Dependency audit passes (`make audit`)
- [ ] Benchmarks have been run; performance is similar
      or better than the previous release
- [ ] Version in root `Cargo.toml` is bumped
- [ ] `Cargo.lock` is regenerated with the new version
- [ ] `.github/SECURITY.md` lists the new minor version
- [ ] Pull request labels produce useful generated release notes
- [ ] Inspect CI runs for the final release commit; the release report records
      missing, skipped, failed, and unfinished evidence as well as successes
- [ ] Inspect the latest successful GPU nightly within 48 hours and any newer
      failures; release publishing reuses ancestor evidence and records its
      tested SHA separately from the release SHA
- [ ] If native GPU qualification is desired, dispatch the **vLLM
      Integration** workflow at the final release commit or tag with
      `run_live_vllm` enabled; it defaults to false and skips the complete
      live suite. The release reuses a verified run at that exact commit for
      live validation and renders `not validated` without one, while the CI
      qualification section keeps reporting ancestor nightly evidence

## Tagging a Release

Tags follow the format `v<MAJOR>.<MINOR>.<PATCH>` (e.g.
`v0.1.0`). Push the tag to the repository:

```console
git tag v0.1.0
git push origin v0.1.0
```

## Publishing Container Images

Container images are published to
[GitHub Container Registry][ghcr] (GHCR).

Pushing a valid release tag triggers the **Release** workflow. It verifies
that the tag matches `workspace.package.version`, runs the test suite,
builds the multi-stage Alpine image, pushes it to
`ghcr.io/praxis-proxy/ai`, and creates the GitHub Release.

Reviewers can manually dispatch the **Publish** workflow when a container
image is needed without creating a tagged GitHub Release. Its optional
`commit` input selects the exact commit to build; when omitted, it builds the
commit associated with the selected dispatch ref.

The same workflow runs nightly from the latest commit on `main`, publishing
the rolling `nightly`, `nightly-experimental`, and `nightly-fips` tags.

[ghcr]: https://ghcr.io/praxis-proxy/ai

### Image Tags

The release workflow produces these tags per run:

| Pattern | Example | Description |
| --------- | --------- | ------------- |
| `sha-<hash>` | `sha-abc1234` | Git commit SHA |
| `<version>` | `0.1.0` | Full semver (from git tag) |
| `<major>.<minor>` | `0.1` | Major.minor shorthand |
| `latest` | `latest` | Most recent tagged release |
| `nightly` | `nightly` | Latest scheduled build from `main` |
| `<any of the above>-experimental` | `0.1.0-experimental` | Same runs, for the experimental image |
| `<any of the above>-fips` | `0.1.0-fips` | Same runs, for the FIPS image |

The workflow also publishes a `sha-<hash>` tag for traceability.

Every run that pushes a standard image also pushes the experimental image
under the same tag with an `-experimental` suffix. It is built from the same
`Containerfile` with `PRAXIS_AI_FEATURES=full,openai-all,store-all`, adding
the experimental OpenAI groups (Responses, agentic loop, response store,
Conversations, compaction, file resolve, MCP tools) and the SQLite and
PostgreSQL store backends. Other experimental filters are not included.

Every run that pushes a standard image also pushes the FIPS image, built
from `Containerfile.fips` on UBI 9 with Red Hat's toolchain and OpenSSL,
under the same tag with a `-fips` suffix. The pinned Red Hat base images are
verified to be signed by Red Hat before every such build
(`make fips-verify-image`). See [FIPS 140-3](fips.md) for what that image
contains and how to run it.

## Changelog

Praxis AI uses [GitHub Releases][gh-releases] for changelogs. The release
workflow creates each release with generated notes. Review pull request
labels before tagging so entries fall into the categories configured in
`.github/release.yml`. There is no separate `CHANGELOG.md` file.

Every release includes a **Release CI evidence** table with the scope, result,
job counts, and run link for each tracked CI workflow at the release commit.
The release's own unit, schema, and integration test gate is included. Missing
runs remain visible, including workflows with path filters or optional triggers.

The release attaches five reports:

- `release-qualification.md`: readable CI job and step outcomes plus GPU details
- `release-qualification.json`: versioned CI snapshot and the GPU report
- `qualification.json`: the original vLLM GPU qualification report
- `support-report.md`: readable release support report
- `support-report.json`: versioned support report with per-cell evidence

CI evidence is a snapshot at publication time, not a claim of complete API
coverage. Skipped jobs and steps remain visible even in successful workflows;
job counts are not individual test counts. The report selects the newest
eligible run for the exact release commit and its latest attempt. PR runs are
excluded because they may check out a synthetic merge. No results are borrowed
from another commit or an older green run. Rerunning Release refreshes the
snapshot and replaces the marked sections and their report assets.
The CI workflow inventory and its scope descriptions live in
`.github/scripts/release_qualification.py`; update them when adding or changing
a validation workflow. Run `python3 -m unittest discover -s tests/qualification`
to check selection, rendering, and the report-generation command.

The separate **vLLM Responses gateway qualification** section records live GPU
evidence. Missing or unverifiable GPU evidence is stated as unavailable, with
its reason and instructions to collect it; it does not block the release.
The additional CI inventory is informational and does not change existing
release gates. See [native vLLM qualification](developing/vllm-qualification.md)
for the GPU schema and recent ancestor nightly selection policy.

## Release support report

Every release also carries a **Release support report** assembled from the
committed conformance reports and this release's exact-commit evidence:

- Provenance: release commit, gateway binary, Praxis core, vLLM, model,
  reference config, and pinned client versions.
- Native API surfaces and the translation support matrix, including every
  known gap and its blocker.
- OpenAI operation conformance and OpenResponses translation coverage. The
  OpenAI percentage states its non-beta, non-deprecated scope and the number of
  operations that scope excludes.
- Live qualification status with client acceptance suite status, reasons, and
  per-scenario results, when a verified GPU run exists at the release commit.
  A client suite that failed before producing its scenario artifact is still
  listed. The report never substitutes the ancestor nightly the CI
  qualification section uses for the release commit.
- Exact-commit CI workflow outcomes.
- Translation cell status changes since the release published before this one,
  when that release's `support-report.json` is available.

The report is intentionally narrower than a product support matrix. It covers
the Praxis AI gateway and the vLLM backend the qualification suite uses; the
model gateway, API server, per-field API support, and live hosted-provider
validation are named as out of scope in the report itself. Missing live
evidence renders as `not validated` rather than being omitted. The report also
refuses evidence collected at another commit: a qualification report for a
different checkout or one that cannot name its checkout, or CI evidence
recording a different release SHA, renders as `provenance mismatch` and copies
no results.

Generation reads `docs/conformance/translation-matrix.json` and
`docs/conformance/openresponses-coverage.json`, which
`cargo xtask translation-matrix --fix` and
`cargo xtask openresponses-coverage --fix` generate and verify beside their
Markdown documents. `.github/scripts/support_report.py` assembles the report;
run `python3 -m unittest tests.qualification.test_support_report` to check
rendering, missing-evidence handling, and diffing.

[gh-releases]: https://github.com/praxis-proxy/ai/releases

## Release Branches

Release branches are optional and created from tags when
backports are needed. The naming convention is
`release/v<MAJOR>.<MINOR>.x` (e.g. `release/v0.1.x`).

Fixes are cherry-picked onto the release branch and a new patch tag is
created from it. The tag triggers the release workflow as usual.

## Container Details

The production image is a minimal Alpine container:

- musl build with LTO, single codegen unit and stripped symbols, linked
  dynamically against Alpine's OpenSSL (all cryptography goes through the
  system library)
- Runs as non-root user (`praxis`)
- Exposes ports `8080` (proxy) and `9901` (admin)
- Built-in health check at
  `http://127.0.0.1:9901/healthy`
- Config directory and working directory: `/etc/praxis` (root-owned; mount the
  configuration read-only)
- Writable state directory: `/var/lib/praxis`, owned by the `praxis` user, for
  backends that keep local files such as the SQLite response store
- Cargo features: `full` — experimental Responses, agentic loops and stores
  require a custom build with explicit [feature selection](features.md#cargo-features)

The `-fips` image is the same binary's FIPS feature set on
`ubi9/ubi-minimal`, built with Red Hat's `rust-toolset` and linked against
UBI's OpenSSL; see [FIPS 140-3](fips.md).

> **Note**: This is subject to change.
