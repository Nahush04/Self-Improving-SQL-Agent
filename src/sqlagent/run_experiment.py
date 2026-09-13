"""The M4 experiment: cold-vs-warm plus every ablation, in one run.

    python -m sqlagent.run_experiment [--train-limit N] [--test-limit N]

Phases, all against one fresh memory store:
1. Cold: memory-free baseline over the held-out test set (A0).
2. Train: the agent works through a stratified sample of the training stream, with
   retrieval, reflection, and memory updates (M2/M3) — this is what makes memory "warm".
3. Warm: the held-out test set again, now with full retrieval (lessons + examples) (A1).
4. Ablations, on the same warm memory, held-out set unchanged:
   - lessons only, examples only: retrieval limited to one memory kind.
   - memory on but retrieval off: the store is warm but the agent never queries it —
     a sanity check that just having memory isn't itself doing anything.
   - episodes only: raw, uncurated past attempts instead of curated lessons/examples,
     picked by simple keyword overlap (episode_retrieval.py) rather than embeddings.

Every phase shares one running dollar total against the cost cap; the whole experiment
stops and reports where it got to if the cap would be exceeded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from . import agent
from .config import RESULTS_DIR, TRAIN_SAMPLE_SIZE, settings
from .dataset import Question, test_set, train_sample
from .episode_retrieval import similar_episodes
from .grader import grade
from .llm import LLM, CostCapExceeded
from .memory.store import MemoryStore
from .memory_client import MemoryClient
from .memory_update import integrate_lesson, store_example
from .reflection import ReflectionInput, reflect
from .run_baseline import _summary


class Budget:
    """Tracks combined spend across every LLM instance in the experiment."""

    def __init__(self, agent_llm: LLM, reflection_llm: LLM) -> None:
        self.agent_llm = agent_llm
        self.reflection_llm = reflection_llm

    @property
    def spent(self) -> float:
        return self.agent_llm.total.cost_usd + self.reflection_llm.total.cost_usd

    def check(self) -> None:
        if self.spent >= settings.cost_cap_usd:
            raise CostCapExceeded(
                f"spend cap ${settings.cost_cap_usd:.2f} reached (${self.spent:.4f} so far)"
            )


RetrieveFn = Callable[[str, str], Awaitable[list[dict]]]


async def _run_test_pass(
    label: str, llm: LLM, budget: Budget, questions: tuple[Question, ...], retrieve: RetrieveFn
) -> dict:
    rows = []
    stopped_early = None
    for q in questions:
        try:
            budget.check()
            memories = await retrieve(q.db_id, q.question)
            res = agent.answer(llm, q.db_id, q.question, q.evidence, settings.max_attempts, memories=memories)
        except CostCapExceeded as exc:
            stopped_early = str(exc)
            break
        g = grade(q.db_id, res.final_sql, q.gold_sql)
        rows.append(
            {
                "question_id": q.question_id,
                "db_id": q.db_id,
                "difficulty": q.difficulty,
                "correct": g.correct,
                "reason": g.reason,
            }
        )
    report = _summary(rows)
    report["label"] = label
    report["stopped_early"] = stopped_early
    print(f"[{label}] accuracy {report['accuracy']:.1%} ({report['correct']}/{report['questions']})")
    return report


async def _train_pass(
    agent_llm: LLM, reflection_llm: LLM, budget: Budget, client: MemoryClient, questions: tuple[Question, ...]
) -> dict:
    from collections import Counter

    from .run_train import _retrieve

    lesson_actions: Counter[str] = Counter()
    correct_count = 0
    stopped_early = None

    for i, q in enumerate(questions, 1):
        try:
            budget.check()
            memories = await _retrieve(client, q.db_id, q.question)
            res = agent.answer(
                agent_llm, q.db_id, q.question, q.evidence, settings.max_attempts, memories=memories
            )
        except CostCapExceeded as exc:
            stopped_early = str(exc)
            break

        g = grade(q.db_id, res.final_sql, q.gold_sql)
        correct_count += int(g.correct)

        lessons: list[str] = []
        try:
            budget.check()
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

        for lesson in lessons:
            action = await integrate_lesson(client, q.db_id, lesson)
            lesson_actions[action] += 1

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
        if i % 25 == 0 or i == len(questions):
            print(f"[train] {i}/{len(questions)}  lessons so far: {dict(lesson_actions)}")
        if stopped_early:
            break

    return {
        "label": "train",
        "questions": len(questions),
        "accuracy": round(correct_count / len(questions), 4) if questions else 0.0,
        "lesson_actions": dict(lesson_actions),
        "stopped_early": stopped_early,
    }


async def run(train_limit: int | None, test_limit: int | None, memory_db_path: Path) -> dict:
    train_questions = train_sample(train_limit or TRAIN_SAMPLE_SIZE)
    test_questions = test_set()
    if test_limit:
        test_questions = test_questions[:test_limit]

    agent_llm = LLM(settings.agent_model)
    reflection_llm = LLM(settings.reflection_model)
    budget = Budget(agent_llm, reflection_llm)

    phases: dict[str, dict] = {}

    # 1. Cold: memory-free, so no memory client needed for this phase.
    async def _no_memory(db_id: str, question: str) -> list[dict]:
        return []

    phases["cold"] = await _run_test_pass("cold", agent_llm, budget, test_questions, _no_memory)

    # 2 & 3 & 4a-c: train, then warm + MCP-backed ablations, on one live memory server.
    async with await MemoryClient.connect(db_path=memory_db_path) as client:
        phases["train"] = await _train_pass(agent_llm, reflection_llm, budget, client, train_questions)

        async def _full(db_id: str, question: str) -> list[dict]:
            lessons = await client.search_memory(question, db_id=db_id, kind="lesson", top_k=3)
            examples = await client.search_memory(question, db_id=db_id, kind="example", top_k=3)
            return lessons + examples

        async def _lessons_only(db_id: str, question: str) -> list[dict]:
            return await client.search_memory(question, db_id=db_id, kind="lesson", top_k=5)

        async def _examples_only(db_id: str, question: str) -> list[dict]:
            return await client.search_memory(question, db_id=db_id, kind="example", top_k=5)

        phases["warm"] = await _run_test_pass("warm", agent_llm, budget, test_questions, _full)
        phases["ablation_lessons_only"] = await _run_test_pass(
            "ablation_lessons_only", agent_llm, budget, test_questions, _lessons_only
        )
        phases["ablation_examples_only"] = await _run_test_pass(
            "ablation_examples_only", agent_llm, budget, test_questions, _examples_only
        )
        phases["ablation_memory_no_retrieval"] = await _run_test_pass(
            "ablation_memory_no_retrieval", agent_llm, budget, test_questions, _no_memory
        )

    # 4d: episodes-only ablation reads the store directly — no MCP client needed for a
    # single-connection read, and the episodes tool is deliberately not part of the
    # public MCP surface (see episode_retrieval.py).
    store = MemoryStore(memory_db_path)
    try:
        async def _episodes_only(db_id: str, question: str) -> list[dict]:
            return similar_episodes(store, db_id, question, top_k=3)

        phases["ablation_episodes_only"] = await _run_test_pass(
            "ablation_episodes_only", agent_llm, budget, test_questions, _episodes_only
        )
    finally:
        store.close()

    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "m4-experiment",
        "agent_model": settings.agent_model,
        "reflection_model": settings.reflection_model,
        "train_questions": len(train_questions),
        "test_questions": len(test_questions),
        "total_cost_usd": round(budget.spent, 4),
        "phases": phases,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--memory-db", default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    memory_db_path = Path(args.memory_db) if args.memory_db else RESULTS_DIR.parent / "data" / f"experiment_{stamp}.sqlite"

    report = asyncio.run(run(args.train_limit, args.test_limit, memory_db_path))

    out = RESULTS_DIR / f"experiment_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    for label, phase in report["phases"].items():
        acc = phase.get("accuracy", 0.0)
        n = phase.get("questions", 0)
        flag = f"  STOPPED EARLY: {phase['stopped_early']}" if phase.get("stopped_early") else ""
        print(f"{label:28} accuracy {acc:.1%}  (n={n}){flag}")
    print(f"total cost ${report['total_cost_usd']:.3f}")
    print(f"written to {out}")
    print(f"memory db: {memory_db_path}")


if __name__ == "__main__":
    main()
