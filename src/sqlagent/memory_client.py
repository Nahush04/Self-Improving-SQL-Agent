"""The agent's MCP client for the memory server.

Kept separate from `sqlagent.memory` on purpose: the agent only ever talks to that
server the way any other MCP client would, over the protocol, never by importing its
storage code directly.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MEMORY_DB_PATH


def _result_value(result: Any) -> Any:
    structured = result.structured_content
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    return structured


class MemoryClient:
    """One persistent MCP session over stdio, reused across many questions."""

    def __init__(self, session: ClientSession, stack: AsyncExitStack) -> None:
        self._session = session
        self._stack = stack

    @classmethod
    async def connect(
        cls, db_path: Path | str = MEMORY_DB_PATH, python_exe: str = sys.executable
    ) -> "MemoryClient":
        params = StdioServerParameters(
            command=python_exe,
            args=["-m", "sqlagent.memory.server", "--db-path", str(db_path)],
        )
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        return cls(session, stack)

    async def close(self) -> None:
        await self._stack.aclose()

    async def __aenter__(self) -> "MemoryClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def search_memory(
        self, query: str, db_id: str | None = None, kind: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        result = await self._session.call_tool(
            "search_memory", {"query": query, "db_id": db_id, "kind": kind, "top_k": top_k}
        )
        return _result_value(result) or []

    async def add_memory(
        self, text: str, kind: str, db_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        result = await self._session.call_tool(
            "add_memory", {"text": text, "kind": kind, "db_id": db_id, "metadata": metadata}
        )
        return _result_value(result)

    async def update_memory(
        self, memory_id: int, text: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        result = await self._session.call_tool(
            "update_memory", {"memory_id": memory_id, "text": text, "metadata": metadata}
        )
        return _result_value(result)

    async def record_episode(
        self,
        question: str,
        correct: bool,
        question_id: str | None = None,
        db_id: str | None = None,
        gold_sql: str | None = None,
        final_sql: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        result = await self._session.call_tool(
            "record_episode",
            {
                "question": question,
                "correct": correct,
                "question_id": question_id,
                "db_id": db_id,
                "gold_sql": gold_sql,
                "final_sql": final_sql,
                "attempts": attempts,
                "reason": reason,
            },
        )
        return _result_value(result)

    async def get_stats(self) -> dict[str, Any]:
        result = await self._session.call_tool("get_stats", {})
        return _result_value(result)
