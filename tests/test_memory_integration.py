"""Integration test: drives the memory server through a real MCP client session
over stdio, in a subprocess, exercising the actual local embedding model.
"""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _tool_result_json(result):
    structured = result.structured_content
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    return structured


@pytest.mark.asyncio
async def test_memory_server_full_round_trip(tmp_path):
    db_path = tmp_path / "integration_memory.sqlite"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sqlagent.memory.server", "--db-path", str(db_path)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert names == {
                "search_memory",
                "add_memory",
                "update_memory",
                "get_memory",
                "list_memories",
                "delete_memory",
                "record_episode",
                "get_stats",
            }

            added = await session.call_tool(
                "add_memory",
                {"text": "revenue means net_amount, not gross_amount", "kind": "lesson", "db_id": "financial"},
            )
            memory = _tool_result_json(added)
            memory_id = memory["id"]
            assert memory["text"].startswith("revenue means")

            fetched = _tool_result_json(await session.call_tool("get_memory", {"memory_id": memory_id}))
            assert fetched["id"] == memory_id

            updated = _tool_result_json(
                await session.call_tool(
                    "update_memory", {"memory_id": memory_id, "metadata": {"confirmed": True}}
                )
            )
            assert updated["metadata"] == {"confirmed": True}

            found = _tool_result_json(
                await session.call_tool(
                    "search_memory", {"query": "what does revenue mean here", "top_k": 3}
                )
            )
            assert any(m["id"] == memory_id for m in found)

            listed = _tool_result_json(await session.call_tool("list_memories", {}))
            assert any(m["id"] == memory_id for m in listed)

            episode = _tool_result_json(
                await session.call_tool(
                    "record_episode",
                    {
                        "question": "what was total revenue in 1997?",
                        "correct": True,
                        "db_id": "financial",
                        "final_sql": "SELECT SUM(net_amount) FROM trans",
                    },
                )
            )
            assert "id" in episode

            stats = _tool_result_json(await session.call_tool("get_stats", {}))
            assert stats["total_memories"] == 1
            assert stats["total_episodes"] == 1
            assert stats["episode_accuracy"] == 1.0

            deleted = _tool_result_json(
                await session.call_tool("delete_memory", {"memory_id": memory_id})
            )
            assert deleted is True
