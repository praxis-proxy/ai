#!/usr/bin/env bash

set -euo pipefail

OGX_URL="${OGX_URL:-http://127.0.0.1:8321}"
OGX_EMBEDDING_MODEL="${OGX_EMBEDDING_MODEL:-ollama/all-minilm:l6-v2}"
OGX_EMBEDDING_DIMENSION="${OGX_EMBEDDING_DIMENSION:-384}"
OGX_PROVIDER_ID="${OGX_PROVIDER_ID:-faiss}"

for command in curl jq; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "error: ${command} is required" >&2
    exit 1
  fi
done

curl_args=(--fail-with-body --silent --show-error)
if [[ -n "${OGX_API_KEY:-}" ]]; then
  curl_args+=(--header "Authorization: Bearer ${OGX_API_KEY}")
fi

tmp_dir="$(mktemp -d)"
vector_store_id=""
file_id=""

cleanup() {
  if [[ -n "${vector_store_id}" ]]; then
    curl "${curl_args[@]}" --request DELETE \
      "${OGX_URL}/v1/vector_stores/${vector_store_id}" >/dev/null || true
  fi
  if [[ -n "${file_id}" ]]; then
    curl "${curl_args[@]}" --request DELETE \
      "${OGX_URL}/v1/files/${file_id}" >/dev/null || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cat >"${tmp_dir}/quarterly-report.txt" <<'EOF'
Praxis Demo Quarterly Report

Revenue grew 37 percent year over year in the second quarter.
The finance team attributed the result to enterprise renewals in EMEA.
Customer support hours remain Monday through Friday, 08:00 to 18:00 UTC.
EOF

echo "Creating an OGX vector store"
create_store_body="$(
  jq --null-input --compact-output \
    --arg model "${OGX_EMBEDDING_MODEL}" \
    --argjson dimension "${OGX_EMBEDDING_DIMENSION}" \
    --arg provider "${OGX_PROVIDER_ID}" \
    '{
      name: "praxis-file-search-demo",
      embedding_model: $model,
      embedding_dimension: $dimension,
      provider_id: $provider
    }'
)"
vector_store_id="$(
  curl "${curl_args[@]}" \
    --header 'Content-Type: application/json' \
    --data "${create_store_body}" \
    "${OGX_URL}/v1/vector_stores" | jq --exit-status --raw-output '.id'
)"

echo "Uploading the demo document"
file_id="$(
  curl "${curl_args[@]}" \
    --form 'purpose=assistants' \
    --form "file=@${tmp_dir}/quarterly-report.txt;type=text/plain" \
    "${OGX_URL}/v1/files" | jq --exit-status --raw-output '.id'
)"

echo "Attaching and indexing the document"
attach_body="$(
  jq --null-input --compact-output \
    --arg file_id "${file_id}" \
    '{
      file_id: $file_id,
      attributes: {department: "finance", region: "emea"}
    }'
)"
curl "${curl_args[@]}" \
  --header 'Content-Type: application/json' \
  --data "${attach_body}" \
  "${OGX_URL}/v1/vector_stores/${vector_store_id}/files" >/dev/null

status="in_progress"
for _ in $(seq 1 60); do
  attachment="$(
    curl "${curl_args[@]}" \
      "${OGX_URL}/v1/vector_stores/${vector_store_id}/files/${file_id}"
  )"
  status="$(jq --raw-output '.status' <<<"${attachment}")"
  case "${status}" in
    completed)
      break
      ;;
    failed | cancelled)
      jq '.last_error' <<<"${attachment}" >&2
      echo "error: OGX indexing ${status}" >&2
      exit 1
      ;;
  esac
  sleep 1
done

if [[ "${status}" != "completed" ]]; then
  echo "error: timed out waiting for OGX indexing" >&2
  exit 1
fi

search() {
  local filters="${1:-null}"
  local request_body

  request_body="$(
    jq --null-input --compact-output \
      --arg query 'What was the year-over-year revenue growth?' \
      --argjson filters "${filters}" \
      '{
        query: $query,
        filters: $filters,
        max_num_results: 5,
        rewrite_query: false
      }'
  )"
  curl "${curl_args[@]}" \
    --header 'Content-Type: application/json' \
    --data "${request_body}" \
    "${OGX_URL}/v1/vector_stores/${vector_store_id}/search"
}

echo "Checking an unfiltered search"
result="$(search)"
jq --exit-status \
  'any(.data[]?.content[]?; ((.text // "") | ascii_downcase | contains("37 percent")))' \
  <<<"${result}" >/dev/null

echo "Checking a matching metadata filter"
finance_filter='{"type":"eq","key":"department","value":"finance"}'
result="$(search "${finance_filter}")"
jq --exit-status '.data | length > 0' <<<"${result}" >/dev/null

echo "Checking a non-matching metadata filter"
engineering_filter='{"type":"eq","key":"department","value":"engineering"}'
result="$(search "${engineering_filter}")"
jq --exit-status '.data | length == 0' <<<"${result}" >/dev/null

echo "OGX file-search demo passed (vector store ${vector_store_id})"
