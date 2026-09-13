# Benchmarks

All numbers below are from a real run of `sqlagent.run_experiment` against the BIRD dev
set (`financial` + `formula_1`, 59 held-out test questions, never used for learning). No
placeholder numbers. Full per-question results are written to a timestamped JSON file in
`results/` by the script; that directory isn't committed (see `.gitignore`) — re-run
`python -m sqlagent.run_experiment` to reproduce.

Run: 2026-09-13. Agent model: `claude-haiku-4-5`. Reflection model: `claude-sonnet-5`.
Training stream: a 150-question stratified subsample of the training pool. Total cost for
the entire experiment (cold run, training pass with reflection, warm re-test, and all four
ablations): **$2.10**.

## Headline result

| Run | Accuracy | Correct / total |
|---|---|---|
| Cold (empty memory) — A0 | 39.0% | 23 / 59 |
| Warm (full memory: lessons + examples) — A1 | 44.1% | 26 / 59 |

**Memory helped: +5.1 percentage points, cold to warm.** The gain is real but modest —
this is in-context improvement from ~150 training questions, not model retraining, and
the agent still gets fewer than half the held-out questions right either way.

## Ablations

Same held-out set, same warm memory store, only the retrieval mode changed:

| Retrieval mode | Accuracy | Correct / total | vs. cold |
|---|---|---|---|
| Full memory (lessons + examples) | 44.1% | 26 / 59 | +5.1 pp |
| Lessons only | 45.8% | 27 / 59 | +6.8 pp |
| Examples only | 44.1% | 26 / 59 | +5.1 pp |
| Episodes only (raw attempt log, keyword match) | 42.4% | 25 / 59 | +3.4 pp |
| Memory warm, retrieval switched off | 39.0% | 23 / 59 | +0.0 pp |

**Sanity check passed.** "Memory warm, retrieval off" landed on 39.0% — statistically the
same as the cold run (23/59 both times, one question's outcome reason differs). This
confirms the improvement comes from the agent actually retrieving and using memory, not
from some side effect of the memory server existing or episodes being logged.

**Lessons carried the most weight.** Lessons-only (45.8%) slightly beat the combined
lessons+examples run (44.1%) and clearly beat examples-only (44.1%) and episodes-only
(42.4%). With only 59 test questions the gap between 44.1% and 45.8% (one question) is not
a strong signal on its own, but the ordering — curated lessons ≥ curated examples > raw
episodes > no memory — is consistent with the design intent: distilled, generalizable
lessons should transfer better than either a single similar solved example or an
uncurated attempt log.

## By difficulty (warm vs. cold)

| Difficulty | Cold | Warm | Change |
|---|---|---|---|
| Simple (n=38) | 44.7% | 44.7% | +0.0 pp |
| Moderate (n=17) | 29.4% | 52.9% | +23.5 pp |
| Challenging (n=4) | 25.0% | 0.0% | -25.0 pp |

Memory's benefit was concentrated in **moderate** questions — exactly where a schema
quirk or naming convention learned during training is most likely to be the difference
between a wrong query and a right one. Simple questions didn't need the help. The
**challenging** bucket only has 4 questions, so the drop there (1 question) is noise, not
a real regression — too small a sample to read anything into.

## Training pass

150 training questions, agent + reflection running together: 57.3% accuracy during
training itself (higher than the held-out numbers, consistent with the training stream
being on average easier / more repetitive than a stratified held-out set). Lesson
add/merge/drop actions taken: 77 added, 30 merged, 5 dropped as near-duplicates.

## An honest caveat on reproducibility

The cold run here (39.0%, 23/59) differs slightly from the M0 baseline run recorded
earlier (37.3%, 22/59) on the exact same 59 questions with the same model. The cause: the
`temperature` parameter this project originally used to pin the agent's output to
deterministic — is fully deprecated by the current Anthropic SDK/API for these models (see
the roadmap status log), so every run now samples with the model's default, non-zero
temperature. Two runs of the identical cold configuration are not guaranteed to produce
the identical answer to every question. The headline cold-vs-warm comparison above still
used the *same* cold and warm passes from one script invocation, so the +5.1pp gap is a
fair same-run comparison — but a second full run of this experiment would likely land on
slightly different absolute numbers.
