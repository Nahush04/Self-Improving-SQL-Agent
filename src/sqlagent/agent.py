"""The baseline text-to-SQL agent: schema in, one SQL query out, one retry on error.

No memory. This is the cold baseline every later version is measured against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import settings
from .db import QueryError, run_query
from .llm import LLM
from .schema import schema_text

_SYSTEM = (
    "You are an expert SQLite analyst. You are given a database schema (with a few "
    "sample rows per table) and a question. Reply with exactly one SQLite query that "
    "answers the question and nothing else — no explanation, no markdown fences. "
    "Use the exact table and column names from the schema, quoting names that contain "
    "spaces or punctuation."
)

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    m = _FENCE.search(text)
    sql = (m.group(1) if m else text).strip()
    # Drop a leading label like "SQL:" and any trailing semicolon noise.
    sql = re.sub(r"^\s*sql\s*:\s*", "", sql, flags=re.I)
    return sql.strip().rstrip(";").strip()


@dataclass
class Attempt:
    sql: str
    error: str | None


@dataclass
class AgentResult:
    final_sql: str
    attempts: list[Attempt] = field(default_factory=list)


def _user_prompt(question: str, evidence: str) -> str:
    parts = [f"Question: {question}"]
    if evidence:
        parts.append(f"Hint: {evidence}")
    return "\n".join(parts)


def answer(llm: LLM, db_id: str, question: str, evidence: str, max_attempts: int = 2) -> AgentResult:
    # The schema block is marked for prompt caching. Running a whole database's
    # questions back to back means every question after the first reads it from cache.
    schema_block = {
        "type": "text",
        "text": f"Database `{db_id}` schema:\n\n{schema_text(db_id)}",
        "cache_control": {"type": "ephemeral"},
    }
    system_blocks = [{"type": "text", "text": _SYSTEM}, schema_block]

    result = AgentResult(final_sql="")
    prompt = _user_prompt(question, evidence)

    for i in range(max_attempts):
        raw = llm.complete(system_blocks, prompt, max_tokens=settings.agent_max_tokens)
        sql = extract_sql(raw)
        result.final_sql = sql

        try:
            run_query(db_id, sql, limit=1)
            result.attempts.append(Attempt(sql, None))
            return result
        except QueryError as exc:
            result.attempts.append(Attempt(sql, str(exc)))
            if i + 1 == max_attempts:
                return result
            prompt = (
                f"{_user_prompt(question, evidence)}\n\n"
                f"Your previous query failed:\n{sql}\n\n"
                f"SQLite error: {exc}\n\n"
                f"Return a corrected query."
            )
    return result
