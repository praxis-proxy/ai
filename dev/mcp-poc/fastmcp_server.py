import uvicorn
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

mcp = FastMCP("Praxis MCP POC FastMCP server")
mcp_posts = 0


class CountMcpPosts:
    """Count local MCP POSTs so auth-denial tests can prove no upstream dial."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global mcp_posts
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/mcp":
            mcp_posts += 1
        await self.app(scope, receive, send)


@mcp.tool
def echo(message: str) -> str:
    """Return a deterministic marker containing the supplied message."""
    return f"fastmcp-echo: {message}"


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.custom_route("/__poc/metrics", methods=["GET"])
async def poc_metrics(_request: Request) -> JSONResponse:
    """Expose the loopback-only MCP POST counter used by the local POC check."""
    return JSONResponse({"mcp_posts": mcp_posts})


app = mcp.http_app(
    stateless_http=True,
    middleware=[Middleware(CountMcpPosts)],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
