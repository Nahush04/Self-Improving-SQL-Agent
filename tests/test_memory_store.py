"""Unit tests for MemoryStore, one per tool. Uses a fast fake embedder — no real
model load — so these tests run instantly and only check the storage/retrieval logic.
"""

from __future__ import annotations

import pytest

from sqlagent.memory.embeddings import EMBED_DIM
from sqlagent.memory.store import MemoryNotFound, MemoryStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic char-frequency vectors: similar text -> similar vector."""
    vectors = []
    for text in texts:
        v = [0.0] * EMBED_DIM
        for ch in text.lower():
            v[ord(ch) % EMBED_DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vectors.append([x / norm for x in v])
    return vectors


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.sqlite", embed_fn=fake_embed)
    yield s
    s.close()


def test_add_and_get_memory(store):
    mid = store.add_memory("status column stores codes not text", "lesson", db_id="financial")
    mem = store.get_memory(mid)
    assert mem["text"] == "status column stores codes not text"
    assert mem["kind"] == "lesson"
    assert mem["db_id"] == "financial"
    assert mem["hit_count"] == 0


def test_get_memory_missing_raises(store):
    with pytest.raises(MemoryNotFound):
        store.get_memory(999)


def test_update_memory_text_and_metadata(store):
    mid = store.add_memory("original text", "lesson")
    updated = store.update_memory(mid, text="revised text", metadata={"note": "fixed"})
    assert updated["text"] == "revised text"
    assert updated["metadata"] == {"note": "fixed"}
    assert updated["updated_at"] >= updated["created_at"]


def test_update_memory_missing_raises(store):
    with pytest.raises(MemoryNotFound):
        store.update_memory(999, text="x")


def test_delete_memory(store):
    mid = store.add_memory("temporary lesson", "lesson")
    assert store.delete_memory(mid) is True
    assert store.delete_memory(mid) is False
    with pytest.raises(MemoryNotFound):
        store.get_memory(mid)


def test_list_memories_filters_and_orders(store):
    store.add_memory("lesson one", "lesson", db_id="financial")
    store.add_memory("example one", "example", db_id="financial")
    store.add_memory("lesson two", "lesson", db_id="formula_1")

    financial_only = store.list_memories(db_id="financial")
    assert {m["text"] for m in financial_only} == {"lesson one", "example one"}

    lessons_only = store.list_memories(kind="lesson")
    assert {m["text"] for m in lessons_only} == {"lesson one", "lesson two"}

    newest_first = store.list_memories()
    assert newest_first[0]["text"] == "lesson two"


def test_search_memory_ranks_by_similarity_and_bumps_hit_count(store):
    salary_id = store.add_memory("average salary means net salary not gross", "lesson")
    store.add_memory("crime counts are recorded per district per year", "lesson")

    results = store.search_memory("what does salary refer to", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == salary_id
    assert "distance" in results[0]

    assert store.get_memory(salary_id)["hit_count"] == 1


def test_search_memory_filters_by_db_id(store):
    store.add_memory("loan duration lesson", "lesson", db_id="financial")
    store.add_memory("loan duration lesson", "lesson", db_id="formula_1")

    results = store.search_memory("loan duration", db_id="formula_1", top_k=5)
    assert all(r["db_id"] == "formula_1" for r in results)


def test_record_episode_and_get_stats(store):
    store.add_memory("a lesson", "lesson")
    store.add_memory("an example", "example")
    store.record_episode(
        "how many clients?",
        correct=True,
        question_id="q1",
        db_id="financial",
        gold_sql="SELECT 1",
        final_sql="SELECT 1",
        attempts=[{"sql": "SELECT 1", "error": None}],
    )
    store.record_episode("how many loans?", correct=False, db_id="financial")

    stats = store.get_stats()
    assert stats["total_memories"] == 2
    assert stats["memories_by_kind"] == {"lesson": 1, "example": 1}
    assert stats["total_episodes"] == 2
    assert stats["correct_episodes"] == 1
    assert stats["episode_accuracy"] == 0.5


def test_add_memory_rejects_unknown_kind(store):
    with pytest.raises(ValueError):
        store.add_memory("x", "not-a-real-kind")
