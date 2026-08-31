"""Execution-accuracy grading, the BIRD metric.

A prediction is correct when running it returns the same set of rows as the gold
query. Row order is ignored unless the question implies an ordering is part of the
answer (the gold query has ORDER BY without a LIMIT that already pins the rows).
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import QueryError, run_query


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    reason: str  # "match", "mismatch", "pred_error:<msg>", "gold_error:<msg>"
    pred_row_count: int | None = None
    gold_row_count: int | None = None


def _as_multiset(rows: list[tuple]) -> dict:
    bag: dict = {}
    for r in rows:
        key = tuple("" if v is None else v for v in r)
        bag[key] = bag.get(key, 0) + 1
    return bag


def grade(db_id: str, pred_sql: str, gold_sql: str, timeout_s: float | None = None) -> GradeResult:
    try:
        gold_rows = run_query(db_id, gold_sql, timeout_s=timeout_s)
    except QueryError as exc:
        # A gold query that will not run is a dataset problem, not an agent failure.
        return GradeResult(False, f"gold_error:{exc}")

    if not pred_sql.strip():
        return GradeResult(False, "pred_error:empty", None, len(gold_rows))

    try:
        pred_rows = run_query(db_id, pred_sql, timeout_s=timeout_s)
    except QueryError as exc:
        return GradeResult(False, f"pred_error:{exc}", None, len(gold_rows))

    correct = _as_multiset(pred_rows) == _as_multiset(gold_rows)
    return GradeResult(
        correct,
        "match" if correct else "mismatch",
        len(pred_rows),
        len(gold_rows),
    )
