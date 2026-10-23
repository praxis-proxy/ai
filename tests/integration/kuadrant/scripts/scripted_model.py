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


def _msg_item(text):
    return {
        "type": "message",
        "id": "msg_" + uuid.uuid4().hex[:16],
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _fc_item(name):
    return {
        "type": "function_call",
        "id": "fc_" + uuid.uuid4().hex[:16],
        "call_id": "call_" + uuid.uuid4().hex[:16],
        "name": name,
        "arguments": json.dumps({"location": "NYC"}),
        "caller": None,
        "namespace": None,
        "status": "completed",
    }


def _sse_frame(ev, data):
    payload = {"type": ev}
    payload.update(data)
    return f"event: {ev}\ndata: {json.dumps(payload)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, body, item):
        rid = "resp_" + uuid.uuid4().hex[:16]
        frames = [
            _sse_frame("response.created", {
                "response": {"id": rid, "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0,
            }),
            _sse_frame("response.output_item.added", {
                "response_id": rid, "output_index": 0, "item": item, "sequence_number": 1,
            }),
        ]
        seq = 2
        if item["type"] == "message":
            frames.append(_sse_frame("response.output_text.delta", {
                "response_id": rid, "item_id": item["id"], "output_index": 0,
                "delta": item["content"][0]["text"], "sequence_number": seq,
            }))
            seq += 1
        resp = base_response(body, [item])
        resp["id"] = rid
        frames.append(_sse_frame("response.completed", {"response": resp, "sequence_number": seq}))
        frames.append(b"data: [DONE]\n\n")
        data = b"".join(frames)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
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
        stream = bool(body.get("stream"))
        if input_has_tool_output(body):
            log(f"turn2 (tool result present) -> final message; stream={stream}; tools={ftools}")
            item = _msg_item("The weather in NYC is sunny, 21C. (final answer)")
        else:
            name = forced_tool(body) or next((n for n in ftools if SUBSTR in n), None)
            if name:
                log(f"turn1 -> function_call {name}; stream={stream}; tools={ftools}")
                item = _fc_item(name)
            else:
                log(f"turn1 -> NO matching tool (substr='{SUBSTR}'); final message; stream={stream}; tools={ftools}")
                item = _msg_item(f"I don't have a '{SUBSTR}' tool available.")

        if stream:
            self._send_sse(body, item)
        else:
            self._send(base_response(body, [item]))

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length","2"); self.end_headers(); self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    log(f"scripted Responses model on 0.0.0.0:{PORT} call-substr='{SUBSTR}'")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
