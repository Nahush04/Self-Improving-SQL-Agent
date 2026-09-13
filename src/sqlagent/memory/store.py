"""SQLite + sqlite-vec storage for memories (lessons, example queries) and episodes.

One file, no server process. Vector search lives in a vec0 virtual table keyed by the
same rowid as the `memories` table, so a join gets both the row and its distance.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import sqlite_vec

from .embeddings import EMBED_DIM, embed_texts

MEMORY_KINDS = ("lesson", "example")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    db_id TEXT,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT,
    db_id TEXT,
    question TEXT NOT NULL,
    gold_sql TEXT,
    final_sql TEXT,
    attempts TEXT NOT NULL DEFAULT '[]',
    correct INTEGER NOT NULL,
    reason TEXT,
    created_at REAL NOT NULL
);
"""


class MemoryNotFound(KeyError):
    """No memory row with the given id."""


@dataclass
class MemoryStore:
    db_path: Path
    embed_fn: Callable[[list[str]], list[list[float]]] = field(default=embed_texts)
    _conn: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False, default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # MCP tool calls run on worker threads, so this connection is shared across
        # threads; _lock below serializes all access to it.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING "
            f"vec0(embedding float[{EMBED_DIM}])"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- memories ---------------------------------------------------------

    def add_memory(
        self,
        text: str,
        kind: str,
        db_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of {MEMORY_KINDS}, got {kind!r}")
        [vec] = self.embed_fn([text])
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (kind, db_id, text, metadata, hit_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 0, ?, ?)",
                (kind, db_id, text, json.dumps(metadata or {}), now, now),
            )
            row_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (row_id, sqlite_vec.serialize_float32(vec)),
            )
            self._conn.commit()
        return row_id

    def get_memory(self, memory_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            raise MemoryNotFound(memory_id)
        return _row_to_memory(row)

    def update_memory(
        self,
        memory_id: int,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise MemoryNotFound(memory_id)

            new_text = text if text is not None else row["text"]
            new_metadata = metadata if metadata is not None else json.loads(row["metadata"])
            self._conn.execute(
                "UPDATE memories SET text = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (new_text, json.dumps(new_metadata), time.time(), memory_id),
            )
            if text is not None and text != row["text"]:
                [vec] = self.embed_fn([new_text])
                self._conn.execute(
                    "UPDATE memories_vec SET embedding = ? WHERE rowid = ?",
                    (sqlite_vec.serialize_float32(vec), memory_id),
                )
            self._conn.commit()
            return self.get_memory(memory_id)

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_memories(
        self,
        db_id: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if db_id is not None:
            clauses.append("db_id = ?")
            params.append(db_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories {where} ORDER BY id DESC LIMIT ? OFFSET ?", params
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def search_memory(
        self,
        query: str,
        db_id: str | None = None,
        kind: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        [qvec] = self.embed_fn([query])
        # Over-fetch from the vector index, then filter by db_id/kind and truncate,
        # since vec0 doesn't join against arbitrary predicates directly.
        candidate_n = max(top_k * 5, 20)
        with self._lock:
            matches = self._conn.execute(
                "SELECT rowid, distance FROM memories_vec WHERE embedding MATCH ? "
                "ORDER BY distance LIMIT ?",
                (sqlite_vec.serialize_float32(qvec), candidate_n),
            ).fetchall()

            results = []
            for m in matches:
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (m["rowid"],)
                ).fetchone()
                if row is None:
                    continue
                if db_id is not None and row["db_id"] not in (db_id, None):
                    continue
                if kind is not None and row["kind"] != kind:
                    continue
                mem = _row_to_memory(row)
                mem["distance"] = m["distance"]
                results.append(mem)
                if len(results) >= top_k:
                    break

            if results:
                self._conn.executemany(
                    "UPDATE memories SET hit_count = hit_count + 1 WHERE id = ?",
                    [(r["id"],) for r in results],
                )
                self._conn.commit()
        return results

    # -- episodes -----------------------------------------------------------

    def record_episode(
        self,
        question: str,
        correct: bool,
        question_id: str | None = None,
        db_id: str | None = None,
        gold_sql: str | None = None,
        final_sql: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
        reason: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes "
                "(question_id, db_id, question, gold_sql, final_sql, attempts, correct, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    db_id,
                    question,
                    gold_sql,
                    final_sql,
                    json.dumps(attempts or []),
                    int(correct),
                    reason,
                    time.time(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    # -- stats ----------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            mem_by_kind = {
                r["kind"]: r["n"]
                for r in self._conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind"
                ).fetchall()
            }
            mem_by_db = {
                (r["db_id"] or "*"): r["n"]
                for r in self._conn.execute(
                    "SELECT db_id, COUNT(*) AS n FROM memories GROUP BY db_id"
                ).fetchall()
            }
            ep_row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS correct FROM episodes"
            ).fetchone()
        total_memories = sum(mem_by_kind.values())
        total_episodes = ep_row["n"]
        correct_episodes = ep_row["correct"]
        return {
            "total_memories": total_memories,
            "memories_by_kind": mem_by_kind,
            "memories_by_db": mem_by_db,
            "total_episodes": total_episodes,
            "correct_episodes": correct_episodes,
            "episode_accuracy": round(correct_episodes / total_episodes, 4)
            if total_episodes
            else None,
        }


def _row_to_memory(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "db_id": row["db_id"],
        "text": row["text"],
        "metadata": json.loads(row["metadata"]),
        "hit_count": row["hit_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
