"""Render a database's schema as text the model can read."""

from __future__ import annotations

import functools
import sqlite3

from .db import db_path

_SAMPLE_ROWS = 3


@functools.lru_cache(maxsize=None)
def schema_text(db_id: str) -> str:
    """CREATE statements for every table, each followed by a few sample rows.

    Sample rows matter for text-to-SQL: they show the model that a status column
    holds codes rather than words, how dates are formatted, and so on.
    """
    path = db_path(db_id)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            (name, ddl)
            for name, ddl in conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        blocks = []
        for name, ddl in tables:
            block = [f"{ddl.strip()};"]
            try:
                rows = conn.execute(
                    f'SELECT * FROM "{name}" LIMIT {_SAMPLE_ROWS}'
                ).fetchall()
                cols = [d[0] for d in conn.execute(f'SELECT * FROM "{name}" LIMIT 0').description]
            except sqlite3.Error:
                rows, cols = [], []
            if rows:
                block.append(f"-- sample rows ({name}): {cols}")
                for r in rows:
                    block.append(f"--   {tuple(r)}")
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)
    finally:
        conn.close()
