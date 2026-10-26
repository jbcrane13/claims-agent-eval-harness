"""The agent's only way to reach business data: an MCP client session.

Two transports:
- "stdio" (default) launches the mock server as a subprocess, so every eval crosses a real
  MCP boundary, the same way a production agent would.
- "inproc" connects to the server object in-process. Faster; used by the unit tests.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from claims_eval.agent.types import ToolSpec

Transport = Literal["stdio", "inproc"]


class ToolBox:
    def __init__(self, client: Client):
        self._client = client
        self._specs: list[ToolSpec] | None = None

    async def specs(self) -> list[ToolSpec]:
        if self._specs is None:
            listing = await self._client.list_tools()
            self._specs = [
                ToolSpec(name=t.name, description=t.description or "", input_schema=t.input_schema)
                for t in listing.tools
            ]
        return self._specs

    async def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (text, is_error). Unknown tools and bad arguments come back as errors, not exceptions,
        so the model can see them and recover."""
        result = await self._client.call_tool(name, args)
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        return text, bool(result.is_error)


@asynccontextmanager
async def open_toolbox(transport: Transport = "stdio", server_log: Path | None = None) -> AsyncIterator[ToolBox]:
    if transport == "inproc":
        from claims_eval.mock_facets.server import mcp as server

        async with Client(server) as client:
            yield ToolBox(client)
        return

    params = StdioServerParameters(command=sys.executable, args=["-m", "claims_eval.mock_facets.server"])
    log_path = server_log or Path("results/server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as errlog:
        async with Client(stdio_client(params, errlog=errlog)) as client:
            yield ToolBox(client)
