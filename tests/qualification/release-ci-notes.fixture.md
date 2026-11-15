<!-- praxis:release-ci-qualification:start -->
### Release CI evidence

Release commit: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`. Snapshot: 2026-10-20T10:00:00+00:00.

Results below record CI jobs and steps. A successful workflow can include skipped tests; job counts are not test counts.

| CI suite | Scope | Result | Jobs / missing evidence | Run |
| --- | --- | --- | --- | --- |
| Tests | Rust unit tests and lint | success | 1 skipped, 1 success | [run 17, attempt 2](https://github.com/praxis-proxy/ai/actions/runs/17) |
| Vision GPU | Anthropic SDK image requests | not_run | No eligible workflow run for the release commit | Unavailable |

See attached `release-qualification.md` for job and step outcomes, `release-qualification.json` for machine-readable CI and GPU evidence, and `qualification.json` for the original GPU report.

Selection: Newest exact-commit run and latest attempt; no older green fallback.

- Job counts are not test counts; skipped steps remain visible.

GPU evidence: **unavailable**. no completed Responses GPU attempt for this commit.
GPU evidence requires a complete successful ancestor nightly within 48 hours. Rerun Release to refresh this snapshot after the nightly completes.
<!-- praxis:release-ci-qualification:end -->
