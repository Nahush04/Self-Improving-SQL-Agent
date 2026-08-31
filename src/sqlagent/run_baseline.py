"""Run the memory-free agent over the held-out test set and record the cold baseline.

    python -m sqlagent.run_baseline [--limit N] [--dry-run]

--dry-run prints the split and the first prompt without calling the API.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone

from . import agent
from .config import RESULTS_DIR, settings
from .dataset import test_set
from .grader import grade
from .llm import LLM, CostCapExceeded
from .schema import schema_text


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    by_diff: dict[str, list[int]] = {}
    by_db: dict[str, list[int]] = {}
    for r in rows:
        by_diff.setdefault(r["difficulty"], []).append(r["correct"])
        by_db.setdefault(r["db_id"], []).append(r["correct"])
    reasons = Counter(r["reason"].split(":")[0] for r in rows)
    return {
        "questions": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_difficulty": {
            k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in by_diff.items()
        },
        "by_database": {
            k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in by_db.items()
        },
        "outcome_reasons": dict(reasons),
    }


def run(limit: int | None = None) -> dict:
    questions = test_set()
    if limit:
        questions = questions[:limit]

    llm = LLM(settings.agent_model)
    rows: list[dict] = []
    stopped_early = None

    for i, q in enumerate(questions, 1):
        t0 = time.monotonic()
        try:
            res = agent.answer(llm, q.db_id, q.question, q.evidence, settings.max_attempts)
        except CostCapExceeded as exc:
            stopped_early = str(exc)
            break
        g = grade(q.db_id, res.final_sql, q.gold_sql)
        rows.append(
            {
                "question_id": q.question_id,
                "db_id": q.db_id,
                "difficulty": q.difficulty,
                "question": q.question,
                "gold_sql": q.gold_sql,
                "pred_sql": res.final_sql,
                "n_attempts": len(res.attempts),
                "attempt_errors": [a.error for a in res.attempts],
                "correct": g.correct,
                "reason": g.reason,
                "wall_s": round(time.monotonic() - t0, 2),
            }
        )
        mark = "OK " if g.correct else "XX "
        print(f"[{i:3}/{len(questions)}] {mark} {q.db_id:10} {q.difficulty:11} "
              f"${llm.total.cost_usd:6.3f}  {q.question[:70]}")

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "baseline-cold",
        "agent_model": settings.agent_model,
        "max_attempts": settings.max_attempts,
        "stopped_early": stopped_early,
        "usage": {
            "calls": llm.total.calls,
            "input_tokens": llm.total.input_tokens,
            "output_tokens": llm.total.output_tokens,
            "cache_write_tokens": llm.total.cache_write_tokens,
            "cache_read_tokens": llm.total.cache_read_tokens,
            "cost_usd": round(llm.total.cost_usd, 4),
            "avg_latency_s": round(llm.total.latency_s / max(llm.total.calls, 1), 2),
        },
        "summary": _summary(rows),
        "results": rows,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N test questions")
    ap.add_argument("--dry-run", action="store_true", help="show the split, no API calls")
    args = ap.parse_args()

    if args.dry_run:
        qs = test_set()
        by_db = Counter(q.db_id for q in qs)
        by_diff = Counter(q.difficulty for q in qs)
        print(f"test set: {len(qs)} questions   by db {dict(by_db)}   by difficulty {dict(by_diff)}")
        q = qs[0]
        print(f"\nfirst question ({q.db_id}, {q.difficulty}): {q.question}")
        print(f"gold: {q.gold_sql}")
        print(f"\nschema chars: {len(schema_text(q.db_id))}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    report = run(limit=args.limit)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"baseline_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    u = report["usage"]
    print("\n" + "=" * 60)
    print(f"cold baseline   accuracy {s['accuracy']:.1%}  ({s['correct']}/{s['questions']})")
    for k, v in sorted(s["by_difficulty"].items()):
        print(f"  {k:12} {v['accuracy']:.1%}  (n={v['n']})")
    for k, v in sorted(s["by_database"].items()):
        print(f"  {k:12} {v['accuracy']:.1%}  (n={v['n']})")
    print(f"cost ${u['cost_usd']:.3f}   calls {u['calls']}   avg latency {u['avg_latency_s']}s")
    if report["stopped_early"]:
        print(f"STOPPED EARLY: {report['stopped_early']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
