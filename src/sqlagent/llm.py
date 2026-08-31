"""Thin wrapper over the Anthropic client that tracks tokens, cost, and a spend cap."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from .config import PRICING, settings


class CostCapExceeded(RuntimeError):
    """The running total would pass the configured spend ceiling for this run."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_usd += other.cost_usd
        self.latency_s += other.latency_s


def _price(model: str, usage: anthropic.types.Usage) -> tuple[Usage, float]:
    p = PRICING.get(model)
    if p is None:  # unknown model — record tokens, leave cost at zero
        p = {"input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0}
    inp = usage.input_tokens
    out = usage.output_tokens
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = (
        inp * p["input"]
        + out * p["output"]
        + cw * p["cache_write"]
        + cr * p["cache_read"]
    ) / 1_000_000
    return Usage(1, inp, out, cw, cr, cost, 0.0), cost


@dataclass
class LLM:
    model: str
    total: Usage = field(default_factory=Usage)
    _client: anthropic.Anthropic | None = None

    def __post_init__(self) -> None:
        if not settings.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to a .env file in the repo root "
                "or export it before running."
            )
        self._client = anthropic.Anthropic(api_key=settings.api_key)

    def complete(
        self,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        if self.total.cost_usd >= settings.cost_cap_usd:
            raise CostCapExceeded(
                f"spend cap ${settings.cost_cap_usd:.2f} reached "
                f"(${self.total.cost_usd:.4f} so far)"
            )
        started = time.monotonic()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": user_text}],
        )
        elapsed = time.monotonic() - started
        call_usage, _ = _price(self.model, resp.usage)
        call_usage.latency_s = elapsed
        self.total.add(call_usage)

        return "".join(b.text for b in resp.content if b.type == "text")
