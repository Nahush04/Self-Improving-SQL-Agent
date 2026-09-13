"""Print a short cold-vs-warm-vs-ablations summary from an experiment JSON report (M5).

    python -m sqlagent.report [PATH]

With no path, prints the most recent results/experiment_*.json.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import RESULTS_DIR

_BAR_WIDTH = 30


def _bar(accuracy: float) -> str:
    filled = round(accuracy * _BAR_WIDTH)
    return "#" * filled + "-" * (_BAR_WIDTH - filled)


def format_report(data: dict) -> str:
    lines = []
    phases = data["phases"]

    lines.append(f"Agent: {data['agent_model']}   Reflection: {data['reflection_model']}")
    lines.append(
        f"Train questions: {data['train_questions']}   Test questions: {data['test_questions']}   "
        f"Total cost: ${data['total_cost_usd']:.3f}"
    )
    lines.append("")
    lines.append("Accuracy by phase:")
    for label, phase in phases.items():
        if "accuracy" not in phase:
            continue
        acc = phase["accuracy"]
        n = phase.get("questions", 0)
        lines.append(f"  {label:28} {_bar(acc)} {acc:6.1%}  (n={n})")

    if "cold" in phases and "warm" in phases:
        gap = phases["warm"]["accuracy"] - phases["cold"]["accuracy"]
        lines.append("")
        lines.append(f"Cold -> warm gap: {gap:+.1%}")

    train = phases.get("train")
    if train and train.get("memory_growth"):
        lines.append("")
        lines.append("Memory store growth during training:")
        for point in train["memory_growth"]:
            lines.append(f"  after {point['questions_seen']:4} questions: {point['total_memories']} memories stored")
        actions = train.get("lesson_actions", {})
        if actions:
            lines.append(f"  lesson actions: {actions}")

    return "\n".join(lines)


def _latest_experiment_json() -> str:
    candidates = sorted(RESULTS_DIR.glob("experiment_*.json"))
    if not candidates:
        raise SystemExit(f"no experiment_*.json files found in {RESULTS_DIR}")
    return str(candidates[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None)
    args = ap.parse_args()

    path = args.path or _latest_experiment_json()
    data = json.loads(open(path, encoding="utf-8").read())
    print(f"report: {path}\n")
    print(format_report(data))


if __name__ == "__main__":
    sys.exit(main() or 0)
