"""Run the agent over the training stream with retrieval, reflection, and memory
updates wired together (M3) — this is how the memory store gets "warm" for M4.

The reflection step sees gold SQL here, during training only; never at test time.

    python -m sqlagent.run_train [--limit N] [--memory-db PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone

from . import agent
from .config import MEMORY_DB_PATH, RESULTS_DIR, settings
from .dataset import train_set
from .grader import grade
from .llm import LLM, CostCapExceeded
from .memory_client import MemoryClient
from .memory_update import integrate_lesson, store_example
from .reflection import ReflectionInput, reflect


async def _retrieve(client: MemoryClient, db_id: str, question: str) -> list[dict]:
    lessons = await client.search_memory(question, db_id=db_id, kind="lesson", top_k=3)
    examples = await client.search_memory(question, db_id=db_id, kind="example", top_k=3)
    return lessons + examples


async def run(limit: int | None, memory_db_path: str) -> dict:
    questions = train_set()
    if limit:
        questions = questions[:limit]

    agent_llm = LLM(settings.agent_model)
    reflection_llm = LLM(settings.reflection_model)
    rows: list[dict] = []
    lesson_actions: Counter[str] = Counter()
    stopped_early = None

    async with await MemoryClient.connect(db_path=memory_db_path) as client:
        for i, q in enumerate(questions, 1):
            t0 = time.monotonic()
            combined_cost = agent_llm.total.cost_usd + reflection_llm.total.cost_usd
            if combined_cost >= settings.cost_cap_usd:
                stopped_early = f"spend cap ${settings.cost_cap_usd:.2f} reached (${combined_cost:.4f} so far)"
                break

            memories = await _retrieve(client, q.db_id, q.question)
            try:
                res = agent.answer(
                    agent_llm, q.db_id, q.question, q.evidence, settings.max_attempts, memories=memories
                )
            except CostCapExceeded as exc:
                stopped_early = str(exc)
                break

            g = grade(q.db_id, res.final_sql, q.gold_sql)

            lessons: list[str] = []
            try:
                lessons = reflect(
                    reflection_llm,
                    ReflectionInput(
                        db_id=q.db_id,
                        question=q.question,
                        evidence=q.evidence,
                        attempts=res.attempts,
                        correct=g.correct,
                        gold_sql=q.gold_sql,
                    ),
                )
            except CostCapExceeded as exc:
                stopped_early = str(exc)

            actions = []
            for lesson in lessons:
                action = await integrate_lesson(client, q.db_id, lesson)
                lesson_actions[action] += 1
                actions.append({"lesson": lesson, "action": action})

            if g.correct:
                await store_example(client, q.db_id, q.question, res.final_sql)

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
                    "correct": g.correct,
                    "memories_used": len(memories),
                    "lessons": actions,
                    "wall_s": round(time.monotonic() - t0, 2),
                }
            )
            mark = "OK " if g.correct else "XX "
            print(
                f"[{i:3}/{len(questions)}] {mark} {q.db_id:10} mem={len(memories)} "
                f"lessons={len(actions)}  ${agent_llm.total.cost_usd + reflection_llm.total.cost_usd:6.3f}  "
                f"{q.question[:50]}"
            )
            if stopped_early:
                break

    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "train-with-reflection-m3",
        "agent_model": settings.agent_model,
        "reflection_model": settings.reflection_model,
        "stopped_early": stopped_early,
        "questions": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "lesson_actions": dict(lesson_actions),
        "cost_usd": round(agent_llm.total.cost_usd + reflection_llm.total.cost_usd, 4),
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
    out = RESULTS_DIR / f"train_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ntrain accuracy {report['accuracy']:.1%}   cost ${report['cost_usd']:.3f}")
    print(f"lesson actions: {report['lesson_actions']}")
    if report["stopped_early"]:
        print(f"STOPPED EARLY: {report['stopped_early']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
