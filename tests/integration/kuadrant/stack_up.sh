#!/usr/bin/env bash
# Kuadrant MCP Gateway acceptance — stand up the minimal, no-full-Kubernetes real
# gateway stack (Envoy -> ext_proc(router) -> ext_authz(Authorino, backed by a bare
# etcd+kube-apiserver) -> broker -> fake MCP backend, plus a static-JWKS IdP) on a
# dedicated container network, and emit `.stack-env` for the acceptance driver.
#
# Engine-parametric: honors CONTAINER_ENGINE, else probes podman then docker.
# Idempotent: re-running wipes and rebuilds containers (images are cached). Tears
# nothing down at the end — run teardown.sh to clean up.
#
# Promoted from the proven spike (.context/kuadrant-spike/scripts/run_acceptance.sh);
# the inline assertion matrix is dropped (the pytest acceptance drives the matrix).
set -u
cd "$(dirname "$0")"
BASE="$(pwd)"
SCRIPTS="$BASE/scripts"; CONF="$BASE/configs"
RUN="$BASE/.run"; CP="$RUN/cp"; KEYS="$RUN/keys"; IDP="$RUN/idp"; LOGS="$RUN/logs"
mkdir -p "$CP" "$KEYS" "$IDP" "$LOGS"

ENGINE="${CONTAINER_ENGINE:-}"
[ -z "$ENGINE" ] && ENGINE="$(command -v podman || command -v docker || true)"
[ -z "$ENGINE" ] && { echo "FATAL: no container engine (set CONTAINER_ENGINE)" >&2; exit 1; }

NET=kuadrant-accept
PY=docker.io/library/python:3.12-slim
IMG_BROKER=ghcr.io/kuadrant/mcp-gateway:v1.0.2
IMG_AUTHORINO=quay.io/kuadrant/authorino:v0.28.0
IMG_ETCD=quay.io/coreos/etcd:v3.5.17
IMG_APISERVER=registry.k8s.io/kube-apiserver:v1.33.6
IMG_ENVOY=docker.io/envoyproxy/envoy:v1.35.0

say(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die(){ printf '\033[31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }
wait_ok(){ # url desc [maxtries]
  local url="$1" desc="$2" max="${3:-60}" i=0
  while [ $i -lt "$max" ]; do curl -sk --max-time 3 "$url" >/dev/null 2>&1 && { echo "   ready: $desc"; return 0; }; sleep 1; i=$((i+1)); done
  die "timeout waiting for $desc ($url)"
}

say "0. clean slate (engine: $ENGINE)"
"$ENGINE" rm -f envoy authorino kube-apiserver etcd mcp-broker weather-backend jwks >/dev/null 2>&1
"$ENGINE" network rm "$NET" >/dev/null 2>&1
"$ENGINE" network create "$NET" >/dev/null || die "network create"

say "1. keys + PKI (fresh)"
# ES256/P-256 wristband keypair (Authorino signs with priv; broker trusts pub)
openssl ecparam -genkey -name prime256v1 -noout -out "$KEYS/priv.pem" 2>/dev/null
openssl ec -in "$KEYS/priv.pem" -pubout -out "$KEYS/pub.pem" 2>/dev/null
# apiserver serving cert + SA keys + client CA/cert (AlwaysAllow resets anon->401, so use client-cert)
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$CP/apiserver.key" -out "$CP/apiserver.crt" -days 7 \
  -subj "/CN=kube-apiserver" -addext "subjectAltName=DNS:kube-apiserver,DNS:localhost,IP:127.0.0.1" 2>/dev/null
openssl genrsa -out "$CP/sa.key" 2048 2>/dev/null; openssl rsa -in "$CP/sa.key" -pubout -out "$CP/sa.pub" 2>/dev/null
openssl genrsa -out "$CP/ca.key" 2048 2>/dev/null
openssl req -x509 -new -nodes -key "$CP/ca.key" -days 7 -out "$CP/ca.crt" -subj "/CN=accept-ca" 2>/dev/null
openssl genrsa -out "$CP/client.key" 2048 2>/dev/null
openssl req -new -key "$CP/client.key" -out "$CP/client.csr" -subj "/CN=admin/O=system:masters" 2>/dev/null
openssl x509 -req -in "$CP/client.csr" -CA "$CP/ca.crt" -CAkey "$CP/ca.key" -CAcreateserial -days 7 -out "$CP/client.crt" 2>/dev/null
# IdP issuer keypair + JWKS
uv run --quiet "$SCRIPTS/gen_issuer_keys.py" --out "$IDP" || die "gen issuer keys"
KID=$(cat "$IDP/issuer_kid.txt")
echo "   wristband kid=wristband-signing-key ; issuer kid=$KID"

say "2. weather-backend (exec counter)"
"$ENGINE" run -d --name weather-backend --network "$NET" -p 9001:9001 \
  -e PORT=9001 -e MCP_RESPONSE_MODE=json -v "$SCRIPTS:/scripts:ro" \
  "$PY" python /scripts/fake_mcp_backend.py >/dev/null || die "weather-backend"
wait_ok http://127.0.0.1:9001/stats "weather-backend /stats"

say "3. jwks IdP"
"$ENGINE" run -d --name jwks --network "$NET" -p 8089:8089 \
  -e PORT=8089 -e JWKS_FILE=/idp/jwks.json -e ISSUER=http://jwks:8089 \
  -v "$SCRIPTS:/scripts:ro" -v "$IDP:/idp:ro" \
  "$PY" python /scripts/jwks_server.py >/dev/null || die "jwks"
wait_ok http://127.0.0.1:8089/jwks "jwks /jwks"

say "4. mcp-broker + router (private-host=envoy:8080, enforce-capability-filtering)"
"$ENGINE" run -d --name mcp-broker --network "$NET" -p 8080:8080 -p 50051:50051 \
  -e GATEWAY_SIGNING_KEY="$(openssl rand -hex 32)" \
  -e TRUSTED_HEADER_PUBLIC_KEY="$(cat "$KEYS/pub.pem")" \
  -v "$CONF/broker-config.yaml:/config/config.yaml:ro" \
  "$IMG_BROKER" ./mcp_gateway --mcp-gateway-config=/config/config.yaml \
    --mcp-gateway-public-host=mcp.local:8080 \
    --mcp-gateway-private-host=envoy:8080 \
    --mcp-broker-public-address=0.0.0.0:8080 \
    --mcp-router-address=0.0.0.0:50051 \
    --enforce-capability-filtering --log-level=-4 >/dev/null || die "mcp-broker"

say "5. etcd + kube-apiserver"
"$ENGINE" run -d --name etcd --network "$NET" "$IMG_ETCD" /usr/local/bin/etcd \
  --name etcd0 --data-dir /etcd-data \
  --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://etcd:2379 >/dev/null || die "etcd"
"$ENGINE" run -d --name kube-apiserver --network "$NET" -p 6443:6443 -v "$CP:/pki:ro" \
  --entrypoint kube-apiserver "$IMG_APISERVER" \
    --etcd-servers=http://etcd:2379 \
    --tls-cert-file=/pki/apiserver.crt --tls-private-key-file=/pki/apiserver.key \
    --client-ca-file=/pki/ca.crt \
    --service-account-key-file=/pki/sa.pub --service-account-signing-key-file=/pki/sa.key \
    --service-account-issuer=https://kubernetes.default.svc \
    --authorization-mode=AlwaysAllow --bind-address=0.0.0.0 --secure-port=6443 >/dev/null || die "apiserver"

# kubeconfigs (host via 127.0.0.1, authorino via container DNS name)
CLICRT=$(base64 < "$CP/client.crt" | tr -d '\n'); CLIKEY=$(base64 < "$CP/client.key" | tr -d '\n')
for v in host authorino; do
  [ "$v" = host ] && S="https://127.0.0.1:6443" || S="https://kube-apiserver:6443"
  cat > "$CP/kubeconfig-$v.yaml" <<EOF
apiVersion: v1
kind: Config
clusters: [{name: bare, cluster: {server: $S, insecure-skip-tls-verify: true}}]
contexts: [{name: bare, context: {cluster: bare, user: admin}}]
current-context: bare
users: [{name: admin, user: {client-certificate-data: $CLICRT, client-key-data: $CLIKEY}}]
EOF
done
chmod 644 "$CP/kubeconfig-authorino.yaml"
export KUBECONFIG="$CP/kubeconfig-host.yaml"
wait_ok https://127.0.0.1:6443/livez "kube-apiserver /livez"
for i in $(seq 1 30); do kubectl --request-timeout=5s get ns >/dev/null 2>&1 && break; sleep 1; done

say "6. apply CRD + signing Secret + AuthConfig (BEFORE authorino)"
kubectl --request-timeout=20s apply -f "$CONF/authconfigs-crd.yaml" >/dev/null 2>&1 \
  || kubectl --request-timeout=20s apply -f https://raw.githubusercontent.com/Kuadrant/authorino/v0.28.0/install/crd/authorino.kuadrant.io_authconfigs.yaml >/dev/null || die "CRD"
kubectl --request-timeout=20s -n default create secret generic wristband-signing-key \
  --from-file=key.pem="$KEYS/priv.pem" --dry-run=client -o yaml | kubectl --request-timeout=20s apply -f - >/dev/null || die "secret"
kubectl --request-timeout=20s apply -f "$CONF/authconfig-filtered.yaml" >/dev/null || die "authconfig"

say "7. authorino"
"$ENGINE" run -d --name authorino --network "$NET" -p 5001:5001 -p 8083:8083 -p 8081:8081 -p 50052:50051 \
  -e KUBECONFIG=/etc/authorino/kubeconfig -v "$CP/kubeconfig-authorino.yaml:/etc/authorino/kubeconfig:ro" \
  "$IMG_AUTHORINO" --log-level=debug --log-mode=development >/dev/null || die "authorino"
wait_ok http://127.0.0.1:8081/readyz "authorino /readyz"
for i in $(seq 1 30); do
  [ "$(kubectl --request-timeout=5s -n default get authconfig mcp-edge -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = "True" ] && { echo "   AuthConfig Ready"; break; }
  sleep 1
done

say "8. envoy (ext_proc -> ext_authz -> router)"
"$ENGINE" run -d --name envoy --network "$NET" -p 8000:8080 -p 9901:9901 \
  -v "$CONF/envoy.yaml:/etc/envoy/envoy.yaml:ro" --entrypoint envoy \
  "$IMG_ENVOY" -c /etc/envoy/envoy.yaml --log-level warning >/dev/null || die "envoy"
wait_ok http://127.0.0.1:9901/ready "envoy admin"

say "9. emit .stack-env"
cat > "$BASE/.stack-env" <<EOF
KUADRANT_CONTAINER_ENGINE=$ENGINE
KUADRANT_NET=$NET
KUADRANT_GATEWAY_URL=http://127.0.0.1:8000/mcp
KUADRANT_GATEWAY_HOST=mcp.local:8080
KUADRANT_ISSUER_PRIV=$IDP/issuer_priv.pem
KUADRANT_ISSUER_KID=$KID
KUADRANT_ISSUER=http://jwks:8089
KUADRANT_AUDIENCE=mcp-gateway
KUADRANT_KUBECONFIG=$CP/kubeconfig-host.yaml
KUADRANT_BACKEND_STATS=http://127.0.0.1:9001
EOF
echo "   wrote $BASE/.stack-env"
echo
echo "   stack up. Gateway MCP: http://127.0.0.1:8000/mcp (Host: mcp.local:8080)"
echo "   teardown: bash $BASE/teardown.sh"
