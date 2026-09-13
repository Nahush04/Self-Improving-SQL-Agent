"""Integration test: the agent-side MemoryClient talking to the real memory server
subprocess over stdio, using the real local embedding model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from sqlagent.memory_client import MemoryClient


@pytest.mark.asyncio
async def test_memory_client_round_trip(tmp_path):
    db_path = tmp_path / "client_integration.sqlite"

    async with await MemoryClient.connect(db_path=db_path) as client:
        added = await client.add_memory(
            "revenue means net_amount, not gross_amount", "lesson", db_id="financial"
        )
        assert added["kind"] == "lesson"

        found = await client.search_memory("what does revenue mean", top_k=3)
        assert any(m["text"].startswith("revenue means") for m in found)

        episode = await client.record_episode(
            "what was revenue in 1997?", correct=True, db_id="financial"
        )
        assert "id" in episode

        stats = await client.get_stats()
        assert stats["total_memories"] == 1
        assert stats["total_episodes"] == 1
