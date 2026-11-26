# Release Process

## Versioning

Praxis AI uses [Semantic Versioning][semver]. The single source of truth is
`workspace.package.version` in the root `Cargo.toml`, which all workspace
crates inherit.

[semver]: https://semver.org/

Releases before `1.0.0` and versions with a SemVer pre-release suffix (such
as `1.0.0-rc.1`) are marked **pre-release** on GitHub, so GitHub's "Latest"
release never points at an unstable build. Stable releases from `1.0.0` on are
published normally.

## Distribution

The supported release artifact is the Praxis AI container image on GHCR.
Workspace crates are not published to crates.io.

## Pre-release Checklist

Before tagging a release:

- [ ] Lints are clean (`make lint`)
- [ ] All tests pass locally (`make test`)
- [ ] Dependency audit passes (`make audit`)
- [ ] Benchmarks show performance similar to or better than the previous
      release
- [ ] Version in root `Cargo.toml` is bumped
- [ ] `Cargo.lock` is regenerated with the new version
- [ ] `.github/SECURITY.md` lists the new minor version
- [ ] Pull request labels produce useful generated release notes
- [ ] CI runs for the final release commit are inspected; the release report
      records missing, skipped, failed, and unfinished evidence as well as
      successes
- [ ] The latest successful GPU nightly within 48 hours, and any newer
      failures, are inspected; publishing reuses this ancestor evidence and
      records its tested SHA separately from the release SHA
      ([selection rules](vllm-qualification.md#release-selection-from-recent-nightlies))
- [ ] For native GPU qualification, the **vLLM Integration** workflow is
      dispatched at the final release commit or tag with `run_live_vllm`
      enabled (it defaults to false, which skips the complete live suite).
      The release uses a verified run at that exact commit for live
      validation and renders `not validated` without one; the CI
      qualification section still reports ancestor nightly evidence
- [ ] FIPS compliance evidence is current:
  - The `fips-host` job succeeded for the release commit. The release
    workflow's `fips-proof` gate checks this; a missing or failed run blocks
    the release unless `skip-fips-proof` is dispatched
  - The release CI evidence table shows the FIPS workflow as `success`
  - The Conversations and Responses mTLS tests ran in `test-postgres-fips-host`
  - `make fips-deps` passes locally for the release commit
  - The compliance report (`make fips-check`) shows no new findings
  - The FIPS host attestation artifact is attached to the fips-host run

## Tagging a Release

Tags use the format `v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v0.1.0`). Push the
tag:

```console
git tag v0.1.0
git push origin v0.1.0
```

## Releasing during a FIPS runner outage

When the FIPS runner is unavailable, a maintainer can bypass FIPS host
validation by dispatching the Release workflow on the release tag:

```console
gh workflow run release.yaml --repo praxis-proxy/ai --ref v0.12.1 \
  -f skip-fips-proof=true
```

This skips the recorded FIPS proof gate and the post-publish FIPS host
attestation, but not the FIPS image build or hosted image checks. The dispatch
input records the decision; the release CI report for such a release is not
evidence of FIPS host validation.

## Recovering Publication

If tests and container publishing passed but GitHub Release creation failed,
leave the tag and published images alone. Rerunning the original workflow
uses the scripts at the tag, so it cannot pick up a fix on `main`. Merge the
tooling fix, then dispatch **Recover release publication** from `main` with
the existing tag and the original Release run ID:

```console
gh workflow run release-recovery.yaml --repo praxis-proxy/ai --ref main \
  -f tag=v0.11.0 -f release-run-id=43437245497
```

Recovery verifies the original run's repository, tag, commit, latest attempt,
and its successful validation, test, container, and FIPS proof gates. Only a
failure in `github-release` is recoverable. Recovery runs the corrected
publication scripts but reads the support inventory from the tagged source.
CI evidence still references the original release run and commit, and GPU
evidence keeps its actual tested checkout. Recovery does not rebuild images or
move tags.

## Publishing Container Images

Pushing a valid release tag triggers the **Release** workflow. It verifies
that the tag matches `workspace.package.version`, runs the test suite, builds
the multi-stage Alpine image, pushes it to `ghcr.io/praxis-proxy/ai` on
[GitHub Container Registry][ghcr] (GHCR), and creates the GitHub Release. The
test job has a 90-minute limit; its unit, schema, and integration test steps
must all pass before container publication.

Reviewers can manually dispatch the **Publish** workflow to get an image
without a tagged GitHub Release. Its optional `commit` input selects the
commit to build; by default it builds the dispatch ref's commit. The same
workflow runs nightly from the latest commit on `main`, publishing the rolling
`nightly`, `nightly-experimental`, and `nightly-fips` tags.

[ghcr]: https://ghcr.io/praxis-proxy/ai

### Image Tags

Each run produces these tags:

| Pattern | Example | Description |
| --------- | --------- | ------------- |
| `sha-<hash>` | `sha-abc1234` | Git commit SHA |
| `<version>` | `0.1.0` | Full semver (from git tag) |
| `<major>.<minor>` | `0.1` | Major.minor shorthand |
| `latest` | `latest` | Most recent tagged release |
| `nightly` | `nightly` | Latest scheduled build from `main` |
| `<any of the above>-experimental` | `0.1.0-experimental` | Same runs, for the experimental image |
| `<any of the above>-fips` | `0.1.0-fips` | Same runs, for the FIPS image |

Every run that pushes a standard image also pushes two variants under the
same tag:

- `-experimental`: the same `Containerfile` built with
  `PRAXIS_AI_FEATURES=full,openai-all,store-all`. This adds the experimental
  OpenAI groups (Responses, agentic loop, response store, Conversations,
  compaction, file resolve, MCP tools) and the SQLite and PostgreSQL store
  backends, but no other experimental filters.
- `-fips`: built from `Containerfile.fips` on UBI 9 with Red Hat's toolchain
  and OpenSSL. Every build first verifies that the pinned Red Hat base images
  are signed by Red Hat (`make fips-verify-image`). See
  [FIPS 140-3](../operations/fips.md) for its contents and how to run it.

## Changelog

Changelogs are [GitHub Releases][gh-releases] with generated notes; there is
no `CHANGELOG.md`. Review pull request labels before tagging so entries land
in the categories configured in `.github/release.yml`.

Every release includes a **Release CI evidence** table with the scope, result,
job counts, and run link for each tracked CI workflow at the release commit,
including the release's own unit, schema, and integration test gate. Missing
runs stay visible, including workflows with path filters or optional
triggers.

The release attaches five reports:

- `release-qualification.md`: readable CI job and step outcomes plus GPU details
- `release-qualification.json`: versioned CI snapshot and the GPU report
- `qualification.json`: the original vLLM GPU qualification report
- `support-report.md`: readable release support report
- `support-report.json`: versioned support report with per-cell evidence

CI evidence is a publication-time snapshot, not a claim of complete API
coverage. Skipped jobs and steps stay visible even in successful workflows,
and job counts are not test counts. The report takes the newest eligible run
for the exact release commit and its latest attempt. PR runs are excluded
because they may check out a synthetic merge, and no results are borrowed
from another commit or an older green run. Rerunning Release refreshes the
snapshot and replaces the marked sections and their report assets. The CI
inventory is informational and does not change release gates.

The CI workflow inventory and its scope descriptions live in
`.github/scripts/release_qualification.py`; update them when adding or
changing a validation workflow. Run
`python3 -m unittest discover -s tests/qualification` to check selection,
rendering, and the report-generation command.

The separate **vLLM Responses gateway qualification** section records live
GPU evidence. Missing or unverifiable GPU evidence is shown as unavailable,
with its reason and how to collect it, and does not block the release. See
[native vLLM qualification](vllm-qualification.md) for the GPU schema and
ancestor nightly selection.

## Release support report

Every release also carries a **Release support report** built from the
committed conformance reports and the release's exact-commit evidence:

- Provenance: release commit, gateway binary, Praxis core, vLLM, model,
  reference config, and pinned client versions.
- Native API surfaces and the translation support matrix, including every
  known gap and its blocker.
- OpenAI operation conformance and OpenResponses translation coverage. The
  OpenAI percentage states its non-beta, non-deprecated scope and the number of
  operations that scope excludes.
- Live qualification status, with client acceptance suite status, reasons,
  and per-scenario results, when a verified GPU run exists at the release
  commit. A client suite that failed before producing its scenario artifact
  is still listed. The report never substitutes the ancestor nightly that the
  CI qualification section uses.
- Exact-commit CI workflow outcomes.
- Translation cell status changes since the release published before this one,
  when that release's `support-report.json` is available.

The report covers only the Praxis AI gateway and the vLLM backend the
qualification suite uses. It names the model gateway, API server, per-field
API support, and live hosted-provider validation as out of scope. Missing live
evidence renders as `not validated` instead of being omitted. Evidence from
another commit is refused: a qualification report for a different checkout,
or one that cannot name its checkout, or CI evidence recording a different
release SHA renders as `provenance mismatch` and copies no results.

Generation reads `docs/conformance/translation-matrix.json` and
`docs/conformance/openresponses-coverage.json`, which
`cargo xtask translation-matrix --fix` and
`cargo xtask openresponses-coverage --fix` generate and verify beside their
Markdown documents. `.github/scripts/support_report.py` assembles the report;
run `python3 -m unittest tests.qualification.test_support_report` to check
rendering, missing-evidence handling, and diffing.

[gh-releases]: https://github.com/praxis-proxy/ai/releases

## Release Branches

Release branches are optional and are created from tags when backports are
needed, named `release/v<MAJOR>.<MINOR>.x` (e.g. `release/v0.1.x`).
Cherry-pick fixes onto the branch and tag a new patch release from it; the tag
triggers the release workflow as usual.

## Container Details

The production image is a minimal Alpine container (subject to change):

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
- Cargo features: `full`. Experimental Responses, agentic loops, and stores
  need a custom build with explicit [feature selection](../features.md#cargo-features)

The `-fips` image is the same binary's FIPS feature set on
`ubi9/ubi-minimal`, built with Red Hat's `rust-toolset` and linked against
UBI's OpenSSL; see [FIPS 140-3](../operations/fips.md).
