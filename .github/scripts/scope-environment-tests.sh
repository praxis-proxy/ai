#!/usr/bin/env bash
set -euo pipefail

scope="${1:?usage: scope-environment-tests.sh environment|suite}"
case "$scope" in
  environment|suite) ;;
  *) echo "unknown environment test scope: $scope" >&2; exit 2 ;;
esac

if [[ "$EVENT_NAME" != "pull_request" ]]; then
  echo "run=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

paths=(
  'integrations/'
  'tests/environment/'
  'tests/utils/'
  'Cargo.toml'
  'Cargo.lock'
  'Makefile'
  '.github/workflows/integration.yaml'
  '.github/scripts/scope-environment-tests.sh'
)
if [[ "$scope" == "suite" ]]; then
  paths+=('tests/integration/tests/suite/examples/llmd_ext_proc.rs')
fi

changed="$(git diff --name-only "$PR_BASE_SHA" HEAD -- "${paths[@]}")"
if [[ -n "$changed" ]]; then
  echo "run=true" >> "$GITHUB_OUTPUT"
else
  echo "run=false" >> "$GITHUB_OUTPUT"
  echo "No ext-proc-relevant changes; skipping $scope tests." >> "$GITHUB_STEP_SUMMARY"
fi
