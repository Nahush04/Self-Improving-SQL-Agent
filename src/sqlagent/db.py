"""Read-only access to a BIRD SQLite database, with a query time limit."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import BIRD_DB_DIR, settings


def db_path(db_id: str) -> Path:
    return BIRD_DB_DIR / db_id / f"{db_id}.sqlite"


class QueryError(RuntimeError):
    """A generated query failed to execute (bad SQL, timeout, missing column, ...)."""


def _connect(path: Path, timeout_s: float) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    # Interrupt any query that runs past the wall-clock budget. progress_handler fires
    # every N bytecode ops; checking the clock there lets us stop a runaway query.
    import time

    deadline = time.monotonic() + timeout_s

    def _guard() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_guard, 10_000)
    return conn


def run_query(
    db_id: str, sql: str, timeout_s: float | None = None, limit: int | None = None
) -> list[tuple[Any, ...]]:
    """Execute ``sql`` against ``db_id`` and return the rows as tuples.

    Raises QueryError on any SQLite failure or on timeout.
    """
    path = db_path(db_id)
    if not path.exists():
        raise QueryError(f"database not found: {db_id}")

    timeout_s = settings.query_timeout_s if timeout_s is None else timeout_s
    conn = _connect(path, timeout_s)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall() if limit is None else cur.fetchmany(limit)
        return [tuple(r) for r in rows]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryError(f"query timed out after {timeout_s}s") from exc
        raise QueryError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise QueryError(str(exc)) from exc
    finally:
        conn.close()
