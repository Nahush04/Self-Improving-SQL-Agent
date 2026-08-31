"""Central configuration: paths, model choices, pricing, and run limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

# BIRD dev set, unzipped in place. Not vendored — see README for the download step.
BIRD_ROOT = REPO_ROOT / "data" / "dev_20240627"
BIRD_DEV_JSON = BIRD_ROOT / "dev.json"
BIRD_DB_DIR = BIRD_ROOT / "dev_databases"

RESULTS_DIR = REPO_ROOT / "results"

# The two databases this project works on. financial is banking/transaction shaped;
# formula_1 is a larger relational schema with more hard questions. Together they give
# a schema-variety story without spreading across all eleven dev databases.
DATABASES = ("financial", "formula_1")

# Deterministic train/test split. The test set is never used for learning.
SPLIT_SEED = 20260830
TEST_PER_DB = 30

# Per-1M-token prices in USD. Keep in sync with the Anthropic pricing page.
PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
}


@dataclass(frozen=True)
class Settings:
    # Cheap model does the many SQL-writing attempts.
    agent_model: str = os.getenv("SQLAGENT_AGENT_MODEL", "claude-haiku-4-5")
    # Stronger model is reserved for the reflection step (added in a later milestone).
    reflection_model: str = os.getenv("SQLAGENT_REFLECTION_MODEL", "claude-sonnet-5")

    agent_max_tokens: int = 1024
    # One retry after a failed attempt: two model calls per question at most.
    max_attempts: int = 2

    # SQLite query wall-clock limit, seconds. Guards against a runaway generated query.
    query_timeout_s: float = 30.0

    # Hard spend ceiling for a single run. The runner aborts before exceeding it.
    cost_cap_usd: float = float(os.getenv("SQLAGENT_COST_CAP_USD", "10.0"))

    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))


settings = Settings()
