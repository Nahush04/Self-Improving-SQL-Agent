"""Load the BIRD dev questions for our databases and split them train/test.

The split is deterministic and stratified by (database, difficulty), so the train and
test sets have the same difficulty mix and the test set is stable across runs.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache

from .config import BIRD_DEV_JSON, DATABASES, SPLIT_SEED, TEST_PER_DB


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def _load_all() -> list[Question]:
    raw = json.loads(BIRD_DEV_JSON.read_text(encoding="utf-8"))
    return [
        Question(
            question_id=r["question_id"],
            db_id=r["db_id"],
            question=r["question"].strip(),
            evidence=(r.get("evidence") or "").strip(),
            gold_sql=" ".join(r["SQL"].split()),
            difficulty=r.get("difficulty", "unknown"),
        )
        for r in raw
        if r["db_id"] in DATABASES
    ]


@lru_cache(maxsize=1)
def _split() -> tuple[tuple[Question, ...], tuple[Question, ...]]:
    questions = _load_all()
    rng = random.Random(SPLIT_SEED)

    strata: dict[tuple[str, str], list[Question]] = {}
    for q in questions:
        strata.setdefault((q.db_id, q.difficulty), []).append(q)

    test: list[Question] = []
    train: list[Question] = []
    # How many test items each stratum contributes, so each database gets TEST_PER_DB.
    for db_id in DATABASES:
        db_strata = {k: v for k, v in strata.items() if k[0] == db_id}
        db_total = sum(len(v) for v in db_strata.values())
        for key, items in db_strata.items():
            rng.shuffle(items)
            n_test = round(TEST_PER_DB * len(items) / db_total)
            test.extend(items[:n_test])
            train.extend(items[n_test:])

    test.sort(key=lambda q: (q.db_id, q.question_id))
    train.sort(key=lambda q: (q.db_id, q.question_id))
    return tuple(train), tuple(test)


def train_set() -> tuple[Question, ...]:
    return _split()[0]


def test_set() -> tuple[Question, ...]:
    return _split()[1]
