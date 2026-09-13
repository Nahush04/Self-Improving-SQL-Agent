"""Unit tests for the add/merge/drop lesson-integration logic, against a stub client
that mimics MemoryClient's async interface without spinning up a real MCP session.
"""

from __future__ import annotations

import pytest

from sqlagent.memory_update import integrate_lesson, store_example


class _StubClient:
    def __init__(self):
        self.added = []
        self.updated = []
        self._next_search_result = []

    def set_search_result(self, memories):
        self._next_search_result = memories

    async def search_memory(self, query, db_id=None, kind=None, top_k=5):
        return self._next_search_result

    async def add_memory(self, text, kind, db_id=None, metadata=None):
        entry = {"id": len(self.added) + 1, "text": text, "kind": kind, "db_id": db_id, "metadata": metadata or {}}
        self.added.append(entry)
        return entry

    async def update_memory(self, memory_id, text=None, metadata=None):
        self.updated.append((memory_id, text, metadata))
        return {"id": memory_id, "text": text, "metadata": metadata or {}}


@pytest.mark.asyncio
async def test_integrate_lesson_adds_when_no_existing_memory():
    client = _StubClient()
    client.set_search_result([])
    action = await integrate_lesson(client, "financial", "a new lesson")
    assert action == "added"
    assert client.added[0]["text"] == "a new lesson"


@pytest.mark.asyncio
async def test_integrate_lesson_drops_near_duplicate():
    client = _StubClient()
    client.set_search_result([{"id": 1, "text": "an existing lesson", "distance": 0.1}])
    action = await integrate_lesson(client, "financial", "an existing lesson restated")
    assert action == "dropped"
    assert client.added == []
    assert client.updated == []


@pytest.mark.asyncio
async def test_integrate_lesson_merges_related_lesson():
    client = _StubClient()
    client.set_search_result([{"id": 1, "text": "status column has codes", "distance": 0.65}])
    action = await integrate_lesson(client, "financial", "status uses numeric codes not words")
    assert action == "merged"
    assert client.updated[0][0] == 1
    assert "status column has codes" in client.updated[0][1]
    assert "status uses numeric codes not words" in client.updated[0][1]


@pytest.mark.asyncio
async def test_integrate_lesson_merge_is_a_no_op_if_already_contained():
    client = _StubClient()
    client.set_search_result(
        [{"id": 1, "text": "already mentions status uses numeric codes not words", "distance": 0.65}]
    )
    action = await integrate_lesson(client, "financial", "status uses numeric codes not words")
    assert action == "merged"
    assert client.updated == []


@pytest.mark.asyncio
async def test_integrate_lesson_adds_when_unrelated_to_nearest_match():
    client = _StubClient()
    client.set_search_result([{"id": 1, "text": "unrelated lesson", "distance": 1.3}])
    action = await integrate_lesson(client, "financial", "a totally different lesson")
    assert action == "added"
    assert client.added[0]["text"] == "a totally different lesson"


@pytest.mark.asyncio
async def test_store_example_sets_metadata():
    client = _StubClient()
    result = await store_example(client, "financial", "how many clients?", "SELECT COUNT(*) FROM client")
    assert result["kind"] == "example"
    assert result["metadata"]["question"] == "how many clients?"
    assert result["metadata"]["sql"] == "SELECT COUNT(*) FROM client"
