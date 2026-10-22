#!/usr/bin/env bash
set -euo pipefail

engine=${CONTAINER_ENGINE:-docker}
image=${1:-praxis-ai:ci}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$repo_root/.context"
workspace=$(mktemp -d "$repo_root/.context/container-sqlite.XXXXXX")
network="praxis-ai-sqlite-$$"
backend="${network}-backend"
proxy="${network}-proxy"

cleanup() {
  "$engine" stop "$proxy" "$backend" >/dev/null 2>&1 || true
  "$engine" network rm "$network" >/dev/null 2>&1 || true
  rm "$workspace/praxis.yaml" "$workspace/backend.sh" 2>/dev/null || true
  rmdir "$workspace" 2>/dev/null || true
}
trap cleanup EXIT

# The example's relative database path lands under root-owned /etc/praxis in
# the image. Keep the example's filter chain while selecting its writable state
# directory and a backend reachable on the test network.
sed \
  -e 's|address: "127.0.0.1:8080"|address: "0.0.0.0:8080"|' \
  -e 's|database_url: "sqlite://responses.db?mode=rwc"|database_url: "sqlite:///var/lib/praxis/responses.db?mode=rwc"|' \
  -e "s|127.0.0.1:8000|$backend:8000|" \
  "$repo_root/examples/configs/openai/responses/response-store.yaml" > "$workspace/praxis.yaml"
# The mock backend has a private address on this disposable container network.
printf '  allow_private_upstreams: true\n' >> "$workspace/praxis.yaml"
grep -Fq 'sqlite:///var/lib/praxis/responses.db?mode=rwc' "$workspace/praxis.yaml"
grep -Fq "$backend:8000" "$workspace/praxis.yaml"

cat > "$workspace/backend.sh" <<'SH'
#!/bin/sh
IFS= read -r request_line || exit 1
case "$request_line" in
  'POST /v1/responses '*) ;;
  *) exit 1 ;;
esac
content_length=0
carriage_return=$(printf '\r')
while IFS= read -r header; do
  header=${header%"$carriage_return"}
  [ -n "$header" ] || break
  case "$header" in
    Content-Length:\ *|content-length:\ *) content_length=${header#*: } ;;
  esac
done
if [ "$content_length" -gt 0 ]; then
  dd bs=1 count="$content_length" of=/dev/null 2>/dev/null
fi
body='{"id":"resp_container_smoke","created_at":1000,"model":"gpt-4.1","object":"response","input":"Hello","output":[{"type":"message","content":[{"type":"output_text","text":"Hi there"}]}]}'
printf 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' "${#body}" "$body"
SH
chmod 755 "$workspace" "$workspace/backend.sh"

"$engine" network create "$network" >/dev/null
# The Alpine runtime image supplies BusyBox nc with the -e option.
"$engine" run -d --rm --network "$network" --name "$backend" \
  -v "$workspace/backend.sh:/tmp/backend.sh:ro" \
  --entrypoint sh "$image" -c 'nc -lk -p 8000 -e /tmp/backend.sh' >/dev/null
"$engine" run --rm --network "$network" \
  -v "$workspace/praxis.yaml:/etc/praxis/praxis.yaml:ro" \
  "$image" --validate -c /etc/praxis/praxis.yaml
"$engine" run -d --rm --network "$network" --name "$proxy" \
  -v "$workspace/praxis.yaml:/etc/praxis/praxis.yaml:ro" \
  "$image" -c /etc/praxis/praxis.yaml >/dev/null

"$engine" exec "$proxy" sh -c 'test -w /var/lib/praxis && test ! -w /etc/praxis'

post=''
for _ in {1..30}; do
  if post=$("$engine" exec "$proxy" wget -qO- -T 5 \
    --header 'Content-Type: application/json' \
    --post-data '{"model":"gpt-4.1","input":"Hello"}' \
    http://127.0.0.1:8080/v1/responses 2>/dev/null); then
    break
  fi
  sleep 1
done
if [ -z "$post" ]; then
  "$engine" logs "$proxy"
  "$engine" logs "$backend"
  echo '::error::container could not serve a Responses POST with SQLite' >&2
  exit 1
fi
python3 -c 'import json,sys; response=json.load(sys.stdin); assert response["id"] == "resp_container_smoke", response' <<< "$post"

get=$("$engine" exec "$proxy" wget -qO- -T 5 \
  http://127.0.0.1:8080/v1/responses/resp_container_smoke)
python3 -c 'import json,sys; response=json.load(sys.stdin); assert response["id"] == "resp_container_smoke", response' <<< "$get"
"$engine" exec "$proxy" test -s /var/lib/praxis/responses.db
echo 'SQLite response persisted and retrieved from the non-root container image'
