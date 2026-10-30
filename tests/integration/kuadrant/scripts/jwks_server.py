#!/usr/bin/env python3
"""Tiny static IdP: serves the JWKS (and an OIDC discovery doc) for Authorino.

GET /jwks                               -> jwks.json
GET /.well-known/openid-configuration   -> minimal OIDC discovery pointing at /jwks

Pure stdlib. Env: PORT (default 8089), JWKS_FILE (default /idp/jwks.json),
ISSUER (default http://jwks:8089).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8089"))
JWKS_FILE = os.environ.get("JWKS_FILE", "/idp/jwks.json")
ISSUER = os.environ.get("ISSUER", "http://jwks:8089")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/jwks"):
            with open(JWKS_FILE) as f:
                self._send(json.load(f))
            return
        if self.path.startswith("/.well-known/openid-configuration"):
            self._send({
                "issuer": ISSUER,
                "jwks_uri": ISSUER + "/jwks",
                "response_types_supported": ["id_token"],
                "subject_types_supported": ["public"],
                "id_token_signing_alg_values_supported": ["RS256"],
            })
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *args):
        print("[jwks]", self.path, file=sys.stderr, flush=True)


if __name__ == "__main__":
    print(f"[jwks] serving {JWKS_FILE} on 0.0.0.0:{PORT} issuer={ISSUER}", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
