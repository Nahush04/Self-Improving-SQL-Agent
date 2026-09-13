import pytest

from sqlagent.episode_retrieval import similar_episodes
from sqlagent.memory.store import MemoryStore


def fake_embed(texts):
    return [[0.0] * 384 for _ in texts]


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "episodes.sqlite", embed_fn=fake_embed)
    yield s
    s.close()


def test_list_episodes_filters_by_db_and_orders_newest_first(store):
    store.record_episode("q1", correct=True, db_id="financial")
    store.record_episode("q2", correct=False, db_id="formula_1")
    store.record_episode("q3", correct=True, db_id="financial")

    financial = store.list_episodes(db_id="financial")
    assert [e["question"] for e in financial] == ["q3", "q1"]


def test_similar_episodes_ranks_by_word_overlap(store):
    store.record_episode(
        "how many clients live in Prague?", correct=True, db_id="financial", final_sql="SELECT 1"
    )
    store.record_episode(
        "what is the average loan amount?", correct=False, db_id="financial", final_sql="SELECT 2"
    )

    results = similar_episodes(store, "financial", "how many clients live in Brno?", top_k=1)
    assert len(results) == 1
    assert results[0]["kind"] == "episode"
    assert "Prague" in results[0]["question"]


def test_similar_episodes_empty_when_no_overlap(store):
    store.record_episode("completely unrelated text", correct=True, db_id="financial")
    results = similar_episodes(store, "financial", "xyz nomatch words here", top_k=3)
    assert results == []
