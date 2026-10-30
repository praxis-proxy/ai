#!/usr/bin/env python3
"""Deterministic passing NeMo `/v1/checks` endpoint for the full-flow test config."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "3003"))
_LOCK = threading.Lock()
_REQUESTS = 0


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _LOCK:
            payload = json.dumps({"requests": _REQUESTS}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        global _REQUESTS
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path == "/reset":
            with _LOCK:
                _REQUESTS = 0
        elif self.path == "/v1/checks":
            with _LOCK:
                _REQUESTS += 1
        else:
            self.send_error(404)
            return
        payload = json.dumps({"status": "passed", "content": "safe"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
