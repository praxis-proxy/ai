#!/usr/bin/env bash
set -euo pipefail

umask 077
cert_dir=/opt/keycloak/data/mcp-poc-tls
mkdir -p "$cert_dir" /public

if [[ ! -s "$cert_dir/https.p12" ]]; then
    keytool -genkeypair \
        -alias mcp-poc-localhost \
        -keyalg RSA \
        -keysize 2048 \
        -validity 365 \
        -storetype PKCS12 \
        -keystore "$cert_dir/https.p12" \
        -storepass "$DEV_KEYSTORE_PASSWORD" \
        -keypass "$DEV_KEYSTORE_PASSWORD" \
        -dname "CN=localhost" \
        -ext "SAN=dns:localhost,ip:127.0.0.1"
fi

keytool -exportcert \
    -rfc \
    -alias mcp-poc-localhost \
    -keystore "$cert_dir/https.p12" \
    -storepass "$DEV_KEYSTORE_PASSWORD" \
    -file /public/keycloak.crt.tmp
mv /public/keycloak.crt.tmp /public/keycloak.crt
chmod 0644 /public/keycloak.crt

exec /opt/keycloak/bin/kc.sh start-dev \
    --http-enabled=false \
    --https-port=8443 \
    --https-key-store-file="$cert_dir/https.p12" \
    --https-key-store-password="$DEV_KEYSTORE_PASSWORD" \
    --import-realm
