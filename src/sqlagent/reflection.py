"""The reflection step (M3): a stronger model looks at one finished attempt and writes
zero or more short lessons — e.g. "the `status` column stores codes, not text".

Training vs. test: `gold_sql` is passed only during the training stream. At test time the
reflection step never sees it, so held-out numbers stay clean — see the roadmap decision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .agent import Attempt
from .llm import LLM

_SYSTEM = (
    "You review one attempt by a text-to-SQL agent and decide whether anything worth "
    "remembering was learned — a fact about the schema, a naming convention, or a common "
    "mistake. Reply with a JSON array of short lesson strings, each a single generalizable "
    "sentence (e.g. \"the status column stores codes, not text\" or \"revenue means "
    "net_amount, not gross_amount\"). Only include a lesson if it would help on a *different* "
    "question later — never restate this specific question or its answer. Reply with an "
    "empty array [] if there is nothing worth remembering. Reply with the JSON array and "
    "nothing else."
)

_ARRAY_RE = re.compile(r"\[.*\]", re.S)


def _extract_lessons(text: str) -> list[str]:
    m = _ARRAY_RE.search(text)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _attempts_block(attempts: list[Attempt]) -> str:
    lines = []
    for i, a in enumerate(attempts, 1):
        status = f"error: {a.error}" if a.error else "ran, but result was wrong"
        lines.append(f"Attempt {i}: {a.sql}\n  {status}")
    return "\n".join(lines)


@dataclass
class ReflectionInput:
    db_id: str
    question: str
    evidence: str
    attempts: list[Attempt]
    correct: bool
    gold_sql: str | None = None  # training only — never set at test time


def reflect(llm: LLM, r: ReflectionInput) -> list[str]:
    if r.correct and len(r.attempts) == 1:
        # Got it right on the first try: nothing went wrong to learn from.
        return []

    parts = [
        f"Database: {r.db_id}",
        f"Question: {r.question}",
    ]
    if r.evidence:
        parts.append(f"Hint: {r.evidence}")
    parts.append(_attempts_block(r.attempts))
    parts.append(f"Final outcome: {'correct' if r.correct else 'incorrect'}")
    if r.gold_sql is not None:
        parts.append(f"Gold SQL (training only): {r.gold_sql}")
    prompt = "\n\n".join(parts)

    raw = llm.complete(
        [{"type": "text", "text": _SYSTEM}], prompt, max_tokens=512
    )
    return _extract_lessons(raw)
