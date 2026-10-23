#!/usr/bin/env python3
"""Deterministic scripted OpenAI Responses model backend for the Part 3 Praxis test.

Praxis's IRR (openai_responses_proxy) POSTs a native Responses create request here.
Turn logic (stateless, keyed on request content):
  - If the input already carries a function_call_output (the MCP tool result came
    back), return a FINAL assistant message (loop ends).
  - Else (first turn): find a function tool in the request whose name contains
    CALL_TOOL_SUBSTR (default "forecast"). If found, return a function_call for it
    (the agentic loop dispatches it to the MCP gateway). If NOT found (tool was
    filtered out at discovery), return a final message (no dispatch).

Env: PORT (default 3001), CALL_TOOL_SUBSTR (default "forecast"). Pure stdlib.
Logs each request's tool names + decision to stderr.
"""
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "3001"))
SUBSTR = os.environ.get("CALL_TOOL_SUBSTR", "forecast")


def log(*a):
    print("[model]", *a, file=sys.stderr, flush=True)


def input_has_tool_output(body):
    inp = body.get("input")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                return True
    return False


def function_tools(body):
    out = []
    for t in body.get("tools") or []:
        if isinstance(t, dict) and t.get("type") == "function" and t.get("name"):
            out.append(t["name"])
    return out


def forced_tool(body):
    """If the user input is `force-call:<name>`, emit a function_call for <name> on
    turn 1 regardless of whether it was discovered. Used to prove Praxis won't
    dispatch a model-generated call to a tool the user never discovered."""
    inp = body.get("input")
    if isinstance(inp, str) and inp.startswith("force-call:"):
        return inp.split(":", 1)[1].strip()
    return None


def base_response(body, output):
    return {
        "id": "resp_" + uuid.uuid4().hex[:16],
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model", "scripted-model"),
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": body.get("tools") or [],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "metadata": {},
        "incomplete_details": None,
        "error": None,
    }


def final_message(body, text):
    return base_response(body, [{
        "type": "message",
        "id": "msg_" + uuid.uuid4().hex[:16],
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }])


def function_call(body, name):
    return base_response(body, [{
        "type": "function_call",
        "id": "fc_" + uuid.uuid4().hex[:16],
        "call_id": "call_" + uuid.uuid4().hex[:16],
        "name": name,
        "arguments": json.dumps({"location": "NYC"}),
        "caller": None,
        "namespace": None,
        "status": "completed",
    }])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception as e:
            log("bad json:", e); self.send_response(400); self.send_header("Content-Length","0"); self.end_headers(); return

        ftools = function_tools(body)
        if input_has_tool_output(body):
            log(f"turn2 (tool result present) -> final message; tools={ftools}")
            self._send(final_message(body, "The weather in NYC is sunny, 21C. (final answer)"))
            return

        forced = forced_tool(body)
        if forced:
            log(f"turn1 -> FORCED function_call {forced}; tools={ftools}")
            self._send(function_call(body, forced))
            return

        match = next((n for n in ftools if SUBSTR in n), None)
        if match:
            log(f"turn1 -> function_call {match}; tools={ftools}")
            self._send(function_call(body, match))
        else:
            log(f"turn1 -> NO matching tool (substr='{SUBSTR}'); final message; tools={ftools}")
            self._send(final_message(body, f"I don't have a '{SUBSTR}' tool available."))

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length","2"); self.end_headers(); self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    log(f"scripted Responses model on 0.0.0.0:{PORT} call-substr='{SUBSTR}'")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
