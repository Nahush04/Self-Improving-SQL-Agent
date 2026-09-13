"""The memory MCP server: exposes MemoryStore as MCP tools.

Run over stdio for local use (the agent, Claude Desktop):

    python -m sqlagent.memory.server

Or over streamable HTTP, guarded by a static API key, for a remote demo:

    python -m sqlagent.memory.server --transport http --port 8000
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import MEMORY_DB_PATH
from .store import MemoryNotFound, MemoryStore


def build_server(store: MemoryStore) -> MCPServer:
    server = MCPServer(
        name="sql-agent-memory",
        instructions=(
            "Memory for a self-improving text-to-SQL agent. Holds lessons learned from "
            "past attempts and example question/SQL pairs, retrievable by similarity, "
            "plus a log of full attempt episodes."
        ),
    )

    @server.tool()
    def search_memory(
        query: str, db_id: str | None = None, kind: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Find memories (lessons or example queries) similar to a natural-language query."""
        return store.search_memory(query, db_id=db_id, kind=kind, top_k=top_k)

    @server.tool()
    def add_memory(
        text: str, kind: str, db_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add a new memory. kind is 'lesson' or 'example'."""
        memory_id = store.add_memory(text, kind, db_id=db_id, metadata=metadata)
        return store.get_memory(memory_id)

    @server.tool()
    def update_memory(
        memory_id: int, text: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Update a memory's text and/or metadata in place."""
        try:
            return store.update_memory(memory_id, text=text, metadata=metadata)
        except MemoryNotFound:
            raise ValueError(f"no memory with id {memory_id}")

    @server.tool()
    def get_memory(memory_id: int) -> dict[str, Any]:
        """Fetch one memory by id."""
        try:
            return store.get_memory(memory_id)
        except MemoryNotFound:
            raise ValueError(f"no memory with id {memory_id}")

    @server.tool()
    def list_memories(
        db_id: str | None = None, kind: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List memories, newest first, optionally filtered by database or kind."""
        return store.list_memories(db_id=db_id, kind=kind, limit=limit, offset=offset)

    @server.tool()
    def delete_memory(memory_id: int) -> bool:
        """Delete a memory by id. Returns whether a row was actually removed."""
        return store.delete_memory(memory_id)

    @server.tool()
    def record_episode(
        question: str,
        correct: bool,
        question_id: str | None = None,
        db_id: str | None = None,
        gold_sql: str | None = None,
        final_sql: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Record a full agent attempt at a question, right or wrong, for later analysis."""
        episode_id = store.record_episode(
            question,
            correct,
            question_id=question_id,
            db_id=db_id,
            gold_sql=gold_sql,
            final_sql=final_sql,
            attempts=attempts,
            reason=reason,
        )
        return {"id": episode_id}

    @server.tool()
    def get_stats() -> dict[str, Any]:
        """Summary counts: memories by kind/database, episode count and accuracy."""
        return store.get_stats()

    return server


def _build_http_app(server: MCPServer, api_key: str | None):
    app = server.streamable_http_app()
    if not api_key:
        return app

    from starlette.responses import PlainTextResponse
    from starlette.types import ASGIApp, Receive, Scope, Send

    class ApiKeyMiddleware:
        def __init__(self, inner: ASGIApp, key: str) -> None:
            self.inner = inner
            self.key = key

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self.inner(scope, receive, send)
                return
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(b"x-api-key", b"").decode()
            if supplied != self.key:
                response = PlainTextResponse("unauthorized", status_code=401)
                await response(scope, receive, send)
                return
            await self.inner(scope, receive, send)

    return ApiKeyMiddleware(app, api_key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db-path", default=str(MEMORY_DB_PATH))
    ap.add_argument(
        "--api-key",
        default=os.getenv("SQLAGENT_MCP_API_KEY"),
        help="required header value for HTTP transport (env SQLAGENT_MCP_API_KEY)",
    )
    args = ap.parse_args()

    from pathlib import Path

    store = MemoryStore(Path(args.db_path))
    server = build_server(store)

    if args.transport == "stdio":
        import asyncio

        asyncio.run(server.run_stdio_async())
    else:
        import uvicorn

        app = _build_http_app(server, args.api_key)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
