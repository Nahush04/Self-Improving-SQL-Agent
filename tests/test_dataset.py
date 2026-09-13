from sqlagent import dataset as ds
from sqlagent.config import DATABASES, TEST_PER_DB


def test_split_is_disjoint_and_stable():
    train_ids = {q.question_id for q in ds.train_set()}
    test_ids = {q.question_id for q in ds.test_set()}
    assert train_ids.isdisjoint(test_ids)
    # second call returns the same objects (cached, deterministic)
    assert [q.question_id for q in ds.test_set()] == sorted(test_ids)


def test_each_database_contributes_its_test_quota():
    per_db: dict[str, int] = {}
    for q in ds.test_set():
        per_db[q.db_id] = per_db.get(q.db_id, 0) + 1
    assert set(per_db) == set(DATABASES)
    for db_id, n in per_db.items():
        assert abs(n - TEST_PER_DB) <= 2, (db_id, n)


def test_questions_have_gold_and_difficulty():
    for q in list(ds.train_set())[:20]:
        assert q.gold_sql.lower().startswith("select")
        assert q.difficulty in {"simple", "moderate", "challenging"}


def test_train_sample_is_stratified_subset_and_deterministic():
    sample = ds.train_sample(150)
    assert len(sample) <= 150
    train_ids = {q.question_id for q in ds.train_set()}
    assert {q.question_id for q in sample} <= train_ids
    assert set(q.db_id for q in sample) == set(DATABASES)
    assert ds.train_sample(150) == sample


def test_train_sample_returns_whole_pool_if_n_too_large():
    full = ds.train_set()
    assert ds.train_sample(len(full) + 100) == full
