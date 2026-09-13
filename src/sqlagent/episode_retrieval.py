"""Simple keyword-overlap episode retrieval, used only by the 'episodes only' ablation
in the M4 experiment. Deliberately not embedding-based and not exposed as an MCP tool —
this is a contrast condition, testing raw uncurated attempt logs against the curated
lessons/examples the production retrieval path uses.
"""

from __future__ import annotations

import re
from typing import Any

from .memory.store import MemoryStore

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def similar_episodes(
    store: MemoryStore, db_id: str, question: str, top_k: int = 3
) -> list[dict[str, Any]]:
    qwords = _words(question)
    scored = []
    for ep in store.list_episodes(db_id=db_id):
        overlap = len(qwords & _words(ep["question"]))
        if overlap:
            scored.append((overlap, ep))
    scored.sort(key=lambda t: -t[0])
    return [{"kind": "episode", **ep} for _, ep in scored[:top_k]]
