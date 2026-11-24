"""Minimal MCP server (Streamable HTTP, JSON responses) for SDK tests.

Serves ``initialize``, ``tools/list``, and ``tools/call`` on ``/mcp`` and
records every JSON-RPC request so tests can assert what Praxis sent. Shared by
the stubbed and live-vLLM Anthropic MCP connector suites.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-03-26"
SESSION_ID = "stub-mcp-session"

WEATHER_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "delete_city",
        "description": "Delete a city from the weather service.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]


def weather_result(arguments: dict) -> str:
    """The text the stub returns for a ``get_weather`` call."""
    return f"The weather in {arguments.get('city', 'unknown')} is sunny and 22C."


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class McpStub:
    """A threaded MCP server exposing ``get_weather`` and ``delete_city``."""

    def __init__(self, tools: list[dict] | None = None):
        self.port = _free_port()
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        tools = WEATHER_TOOLS if tools is None else tools
        captured = self.requests
        captured_headers = self.headers

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, status: int, payload: dict | None, session: bool = False) -> None:
                body = json.dumps(payload).encode() if payload is not None else b""
                self.send_response(status)
                if payload is not None:
                    self.send_header("Content-Type", "application/json")
                if session:
                    self.send_header("Mcp-Session-Id", SESSION_ID)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def do_GET(self):  # noqa: N802 (http.server API)
                self._reply(405, None)

            def do_DELETE(self):  # noqa: N802 (http.server API)
                self._reply(200, None)

            def do_POST(self):  # noqa: N802 (http.server API)
                length = int(self.headers.get("Content-Length", "0"))
                message = json.loads(self.rfile.read(length) or b"{}")
                captured.append(message)
                captured_headers.append({key.lower(): value for key, value in self.headers.items()})
                method = message.get("method")
                if "id" not in message:
                    self._reply(202, None)
                    return
                if method == "initialize":
                    result = {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "praxis-sdk-stub", "version": "0.0.0"},
                    }
                    self._reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": result}, session=True)
                elif method == "tools/list":
                    self._reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": tools}})
                elif method == "tools/call":
                    params = message.get("params", {})
                    text = weather_result(params.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": text}], "isError": False}
                    self._reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": result})
                else:
                    error = {"code": -32601, "message": "method not found"}
                    self._reply(200, {"jsonrpc": "2.0", "id": message["id"], "error": error})

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self) -> "McpStub":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def calls(self, method: str) -> list[dict]:
        """Captured JSON-RPC requests for ``method``."""
        return [request for request in self.requests if request.get("method") == method]
