#!/usr/bin/env bash
# Publish reports for the checked-out release using a verified release test run.
set -euo pipefail
sha=$(git rev-parse 'HEAD^{commit}')
mkdir -p "$RUNNER_TEMP/release-qualification"
python3 .github/scripts/vllm_qualification.py release-evidence \
  --repo "$GH_REPO" --sha "$sha" \
  --output-dir "$RUNNER_TEMP/release-qualification"

# The qualification section reports ancestor nightly evidence. The
# support report only claims live validation for the release commit
# itself, which requires a vLLM Integration run dispatched at that
# commit. A missing run leaves the live section "not validated".
exact_dir="$RUNNER_TEMP/support-exact"
python3 .github/scripts/vllm_qualification.py exact-evidence \
  --repo "$GH_REPO" --sha "$sha" --output-dir "$exact_dir" || true

if gh release view "$TAG" --repo "$GH_REPO" --json body \
  --jq .body > "$RUNNER_TEMP/release-qualification/base-notes.md" 2>/dev/null; then
  existing=true
else
  existing=false
  gh api -X POST "repos/$GH_REPO/releases/generate-notes" \
    -f tag_name="$TAG" -f target_commitish="$sha" \
    --jq .body > "$RUNNER_TEMP/release-qualification/base-notes.md"
fi
python3 .github/scripts/release_qualification.py \
  --repo "$GH_REPO" --sha "$sha" \
  --run-id "${RELEASE_RUN_ID:-$GITHUB_RUN_ID}" --attempt "${RELEASE_RUN_ATTEMPT:-$GITHUB_RUN_ATTEMPT}" \
  --gpu-report "$RUNNER_TEMP/release-qualification/qualification.json" \
  --existing-notes "$RUNNER_TEMP/release-qualification/base-notes.md" \
  --output-dir "$RUNNER_TEMP/release-qualification"
cat "$RUNNER_TEMP/release-qualification/release-qualification.md" >> "$GITHUB_STEP_SUMMARY"

# Best-effort previous release for a status diff. A rerun of an older
# tag compares against the release published before it, not the newest
# release overall, so it cannot present regressions backwards. A
# missing asset is not an error; the report has no comparison section.
prev_dir="$RUNNER_TEMP/support-previous"
mkdir -p "$prev_dir"
if [ "$existing" = true ]; then
  released_at=$(gh release view "$TAG" --repo "$GH_REPO" \
    --json createdAt --jq .createdAt)
else
  released_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
fi
prev_tag=$(gh api "repos/$GH_REPO/releases?per_page=100" \
  --jq "[.[] | select(.draft == false and .created_at < \"$released_at\")]
        | max_by(.created_at) | .tag_name // empty" \
  2>/dev/null || true)
if [ -n "${prev_tag:-}" ]; then
  gh release download "$prev_tag" --repo "$GH_REPO" \
    --pattern support-report.json --dir "$prev_dir" 2>/dev/null || true
fi
prev_args=()
if [ -f "$prev_dir/support-report.json" ]; then
  prev_args=(--previous "$prev_dir/support-report.json")
fi
python3 .github/scripts/support_report.py generate \
  --tag "$TAG" --sha "$sha" \
  --qualification "$exact_dir/qualification.json" \
  --release-qualification "$RUNNER_TEMP/release-qualification/release-qualification.json" \
  "${prev_args[@]}" \
  --existing-notes "$RUNNER_TEMP/release-qualification/release-notes.md" \
  --output-dir "$RUNNER_TEMP/release-qualification"
cat "$RUNNER_TEMP/release-qualification/support-report.md" >> "$GITHUB_STEP_SUMMARY"

if [ "$existing" = true ]; then
  gh release edit "$TAG" --repo "$GH_REPO" \
    --prerelease="$PRERELEASE" \
    --notes-file "$RUNNER_TEMP/release-qualification/release-notes.md"
else
  gh release create "$TAG" --repo "$GH_REPO" --verify-tag \
    --prerelease="$PRERELEASE" \
    --notes-file "$RUNNER_TEMP/release-qualification/release-notes.md"
fi
gh release upload "$TAG" \
  "$RUNNER_TEMP/release-qualification/qualification.json" \
  "$RUNNER_TEMP/release-qualification/release-qualification.json" \
  "$RUNNER_TEMP/release-qualification/release-qualification.md" \
  "$RUNNER_TEMP/release-qualification/support-report.json" \
  "$RUNNER_TEMP/release-qualification/support-report.md" \
  --repo "$GH_REPO" --clobber
gh release view "$TAG" --repo "$GH_REPO" --json body,assets \
  > "$RUNNER_TEMP/release-qualification/published.json"
python3 - "$RUNNER_TEMP/release-qualification/published.json" <<'PY'
import json, pathlib, sys
release = json.loads(pathlib.Path(sys.argv[1]).read_text())
body = release.get("body") or ""
names = {asset["name"] for asset in release.get("assets", [])}
if body.count("<!-- praxis:native-vllm-qualification:start -->") != 1 \
        or body.count("<!-- praxis:native-vllm-qualification:end -->") != 1 \
        or body.count("<!-- praxis:release-ci-qualification:start -->") != 1 \
        or body.count("<!-- praxis:release-ci-qualification:end -->") != 1 \
        or body.count("<!-- praxis:support-report:start -->") != 1 \
        or body.count("<!-- praxis:support-report:end -->") != 1 \
        or not {"qualification.json", "release-qualification.json",
                "release-qualification.md", "support-report.json",
                "support-report.md"} <= names:
    raise SystemExit("release is missing its unique qualification sections or report assets")
PY
