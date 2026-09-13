"""Turn a reflection lesson into a memory-store write: add, merge, or drop.

A simple, fixed distance-threshold rule instead of a learned or LLM-judged merge step —
see LESSON_DUP_DISTANCE / LESSON_MERGE_DISTANCE in config.py. Thresholds were picked by
checking embedding distances between a paraphrased lesson pair (~0.64) and unrelated
lesson pairs (~1.2-1.4) with the real embedding model.
"""

from __future__ import annotations

from .config import LESSON_DUP_DISTANCE, LESSON_MERGE_DISTANCE
from .memory_client import MemoryClient


async def integrate_lesson(client: MemoryClient, db_id: str, lesson_text: str) -> str:
    """Add a new lesson to memory, deduping against the closest existing one.

    Returns which action was taken: "added", "merged", or "dropped".
    """
    matches = await client.search_memory(lesson_text, db_id=db_id, kind="lesson", top_k=1)
    if not matches:
        await client.add_memory(lesson_text, "lesson", db_id=db_id)
        return "added"

    nearest = matches[0]
    distance = nearest["distance"]

    if distance <= LESSON_DUP_DISTANCE:
        return "dropped"

    if distance <= LESSON_MERGE_DISTANCE:
        if lesson_text not in nearest["text"]:
            merged_text = f"{nearest['text']}; {lesson_text}"
            await client.update_memory(nearest["id"], text=merged_text)
        return "merged"

    await client.add_memory(lesson_text, "lesson", db_id=db_id)
    return "added"


async def store_example(client: MemoryClient, db_id: str, question: str, sql: str) -> dict:
    """Record a correctly-answered question as a retrievable example for later ones."""
    return await client.add_memory(
        question, "example", db_id=db_id, metadata={"question": question, "sql": sql}
    )
