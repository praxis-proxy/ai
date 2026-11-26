#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run one credentialed GPU case and publish its per-assertion evidence.
set -euo pipefail
case "$1" in
  native|translated) route="$1" ;;
  *) exit 2 ;;
esac
export PRAXIS_TEST_CODEX_EVIDENCE
PRAXIS_TEST_CODEX_EVIDENCE="$(mktemp -d "$RUNNER_TEMP/codex-search.XXXXXX")/evidence.json"
echo "CODEX_${route^^}_SEARCH_EVIDENCE=$PRAXIS_TEST_CODEX_EVIDENCE" >> "$GITHUB_ENV"
log="$RUNNER_TEMP/codex-acceptance-${route}_search.log"
sudo --preserve-env=TAVILY_API_KEY,PRAXIS_TEST_CODEX_EVIDENCE,PRAXIS_TEST_CODEX_BIN,PRAXIS_TEST_CODEX_VLLM_BASE_URL,PRAXIS_TEST_CODEX_VLLM_MODEL,PRAXIS_TEST_CODEX_DATABASE_URL,CODEX_BACKEND_TOKEN,PRAXIS_TEST_CODEX_NETNS,PRAXIS_TEST_CODEX_LISTEN_ADDRESS,PRAXIS_TEST_CODEX_REQUIRE_LIVE,PRAXIS_TEST_REQUIRE_EGRESS_ISOLATION \
  "$TEST_BIN" "codex_http::pinned_codex_${route}_live_web_search" --exact --nocapture 2>&1 | tee "$log"
grep -F "test result: ok. 1 passed; 0 failed; 0 ignored;" "$log"
test -s "$PRAXIS_TEST_CODEX_EVIDENCE"
