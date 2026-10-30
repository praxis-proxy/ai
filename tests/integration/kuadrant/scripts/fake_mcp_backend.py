#!/usr/bin/env python3
"""Fake MCP streamable-HTTP backend for the Kuadrant spike (Part 2).

Serves MCP initialize / notifications/initialized / tools/list / tools/call, and
RECORDS a per-tool execution counter so "zero backend executions" is provable.

Endpoints:
  POST /mcp      MCP streamable-HTTP
  GET  /stats    -> {"get_forecast": N, "get_alerts": M, "total": T}  (tool-call executions)
  POST /reset    -> zero the counters

Only *tools/call* for a KNOWN tool increments the counter. Discovery (tools/list) and
the hairpin initialize do NOT. Pure stdlib.

Env: PORT (default 9001), MCP_RESPONSE_MODE (json|sse, default json)
"""
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "9001"))
MODE = os.environ.get("MCP_RESPONSE_MODE", "json").lower()
DEFAULT_PROTOCOL = "2025-06-18"

TOOLS = [
    {"name": "get_forecast", "description": "Get the weather forecast for a location",
     "inputSchema": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
    {"name": "get_alerts", "description": "Get active weather alerts for a location",
     "inputSchema": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
]
KNOWN = {t["name"] for t in TOOLS}

_lock = threading.Lock()
_counts = {"get_forecast": 0, "get_alerts": 0}


def log(*a):
    print("[backend]", *a, file=sys.stderr, flush=True)


def tool_result(name, args):
    loc = (args or {}).get("location", "unknown")
    if name == "get_forecast":
        return f"Forecast for {loc}: sunny, 21C (REAL backend execution)"
    if name == "get_alerts":
        return f"Alerts for {loc}: none (REAL backend execution)"
    return f"{name}: ok"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json_rpc(self, obj, session_id=None):
        body = json.dumps(obj).encode()
        if MODE == "sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.end_headers()
            self.wfile.write(b"event: message\ndata: " + body + b"\n\n")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.end_headers()
            self.wfile.write(body)

    def _plain(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _accepted(self):
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/stats"):
            with _lock:
                snap = dict(_counts)
            snap["total"] = sum(snap.values())
            self._plain(200, snap)
            return
        # No server-initiated SSE stream.
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/reset"):
            with _lock:
                for k in _counts:
                    _counts[k] = 0
            self._plain(200, {"reset": True})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            msg = json.loads(raw) if raw else {}
        except Exception as e:
            log("bad json:", e)
            self.send_response(400); self.send_header("Content-Length", "0"); self.end_headers()
            return
        if isinstance(msg, list):
            msg = msg[0] if msg else {}

        method = msg.get("method")
        msg_id = msg.get("id")
        log(f"POST {self.path} method={method} id={msg_id} session={self.headers.get('Mcp-Session-Id')}")

        if method == "initialize":
            client_pv = (msg.get("params") or {}).get("protocolVersion") or DEFAULT_PROTOCOL
            self._json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {
                "protocolVersion": client_pv,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "fake-weather", "version": "0.0.2"},
            }}, session_id=uuid.uuid4().hex)
            return

        if method and method.startswith("notifications/"):
            self._accepted()
            return

        if method == "tools/list":
            self._json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})
            return

        if method == "tools/call":
            params = msg.get("params") or {}
            tool = params.get("name")
            args = params.get("arguments") or {}
            if tool in KNOWN:
                with _lock:
                    _counts[tool] += 1
                    n = _counts[tool]
                log(f"EXECUTED tool={tool} count={n}")
                self._json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": tool_result(tool, args)}], "isError": False}})
            else:
                self._json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": f"unknown tool {tool}"}], "isError": True}})
            return

        if method == "ping":
            self._json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            return

        if msg_id is not None:
            self._json_rpc({"jsonrpc": "2.0", "id": msg_id,
                            "error": {"code": -32601, "message": f"Method not found: {method}"}})
        else:
            self._accepted()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    log(f"starting fake MCP backend (v2, exec-counter) on 0.0.0.0:{PORT} mode={MODE}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
