import argparse
import asyncio
import json
from pathlib import Path

from fastmcp import Client

PROTOCOL_VERSION = "2026-07-28"


def parse_server(value: str) -> tuple[str, str]:
    name, separator, url = value.partition("=")
    if not separator or not name or not url:
        raise argparse.ArgumentTypeError("server must be NAME=URL")
    return name, url


async def discover_one(name: str, url: str) -> dict[str, object]:
    async with Client(url, mode="auto") as client:
        negotiated_version = client.protocol_version
        if negotiated_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"{name} negotiated {negotiated_version!r}, expected {PROTOCOL_VERSION!r}"
            )
        tools = await client.list_tools()

    definitions = [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in tools
    ]
    for tool in definitions:
        if not isinstance(tool.get("name"), str) or not isinstance(tool.get("inputSchema"), dict):
            raise RuntimeError(f"{name} returned a tool without name/inputSchema: {tool!r}")

    return {
        "name": name,
        "cluster": name,
        "path": "/mcp",
        "tool_prefix": f"{name}_",
        "tools": definitions,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull tools/list from stateless MCP servers and write mcp_poc catalog YAML/JSON."
    )
    parser.add_argument("--server", action="append", type=parse_server, required=True, metavar="NAME=URL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = [name for name, _ in args.server]
    if len(names) != len(set(names)):
        raise SystemExit("server names must be unique")

    catalog = {
        "protocol_version": PROTOCOL_VERSION,
        "servers": [await discover_one(name, url) for name, url in args.server],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {sum(len(server['tools']) for server in catalog['servers'])} tools to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
