"""Run the agent with memory retrieval wired in, over the memory MCP server (M2).

This proves the agent retrieves memories through the server and folds them into its
prompt — it does not yet write lessons back (that's the reflection step, M3), so a run
against an empty memory store behaves like the baseline, just with an extra round trip.

    python -m sqlagent.run_with_memory [--limit N] [--memory-db PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

from . import agent
from .config import MEMORY_DB_PATH, RESULTS_DIR, settings
from .dataset import test_set
from .grader import grade
from .llm import LLM, CostCapExceeded
from .memory_client import MemoryClient


async def _retrieve(client: MemoryClient, db_id: str, question: str) -> list[dict]:
    lessons = await client.search_memory(question, db_id=db_id, kind="lesson", top_k=3)
    examples = await client.search_memory(question, db_id=db_id, kind="example", top_k=3)
    return lessons + examples


async def run(limit: int | None, memory_db_path: str) -> dict:
    questions = test_set()
    if limit:
        questions = questions[:limit]

    llm = LLM(settings.agent_model)
    rows: list[dict] = []
    stopped_early = None

    async with await MemoryClient.connect(db_path=memory_db_path) as client:
        for i, q in enumerate(questions, 1):
            t0 = time.monotonic()
            memories = await _retrieve(client, q.db_id, q.question)
            try:
                res = agent.answer(
                    llm, q.db_id, q.question, q.evidence, settings.max_attempts, memories=memories
                )
            except CostCapExceeded as exc:
                stopped_early = str(exc)
                break
            g = grade(q.db_id, res.final_sql, q.gold_sql)
            await client.record_episode(
                q.question,
                g.correct,
                question_id=str(q.question_id),
                db_id=q.db_id,
                gold_sql=q.gold_sql,
                final_sql=res.final_sql,
                attempts=[{"sql": a.sql, "error": a.error} for a in res.attempts],
                reason=g.reason,
            )
            rows.append(
                {
                    "question_id": q.question_id,
                    "db_id": q.db_id,
                    "difficulty": q.difficulty,
                    "question": q.question,
                    "gold_sql": q.gold_sql,
                    "pred_sql": res.final_sql,
                    "memories_used": len(memories),
                    "correct": g.correct,
                    "reason": g.reason,
                    "wall_s": round(time.monotonic() - t0, 2),
                }
            )
            mark = "OK " if g.correct else "XX "
            print(
                f"[{i:3}/{len(questions)}] {mark} {q.db_id:10} mem={len(memories)}  "
                f"${llm.total.cost_usd:6.3f}  {q.question[:60]}"
            )

    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "with-memory-m2-demo",
        "agent_model": settings.agent_model,
        "stopped_early": stopped_early,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "cost_usd": round(llm.total.cost_usd, 4),
        "results": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--memory-db", default=str(MEMORY_DB_PATH))
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    report = asyncio.run(run(args.limit, args.memory_db))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"with_memory_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\naccuracy {report['accuracy']:.1%}   cost ${report['cost_usd']:.3f}")
    if report["stopped_early"]:
        print(f"STOPPED EARLY: {report['stopped_early']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
