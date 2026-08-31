# self-improving-sql-agent

A text-to-SQL agent that answers natural-language questions about a database by writing
and running SQL, and gets better over time by reading from an external memory. The memory
is a separate [Model Context Protocol](https://modelcontextprotocol.io) server; the agent
is a client of it, and so is any other MCP client (Claude Desktop, etc.).

The point of the project is a measured result: accuracy on a held-out question set with
memory **cold** vs. memory **warm**, plus ablations, reported honestly.

## Status

| Milestone | What | State |
|---|---|---|
| M0 | Data, grader, memory-free baseline agent, cold-run harness | code done; cold run pending an API key |
| M1 | Memory MCP server | not started |
| M2 | Agent talks to the memory server | not started |
| M3 | Reflection + memory-update logic | not started |
| M4 | Full cold-vs-warm experiment + ablations | not started |
| M5 | Package: Docker, Claude Desktop demo, CI | not started |

## Task and data

Questions come from the [BIRD](https://bird-bench.github.io) dev set, restricted to two of
its databases: **financial** (banking / transactions) and **formula_1** (a larger
relational schema with more hard questions). Each item has a question, a gold SQL query,
and a short hint.

The split is deterministic and stratified by (database, difficulty): 30 held-out test
questions per database, the rest form the practice pool. The test set is never used for
learning.

### Getting the data

The BIRD dev set (~340 MB) is not committed. Download and unzip it into `data/`:

```
curl -L -o data/dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
cd data && unzip dev.zip && cd dev_20240627 && unzip dev_databases.zip
```

You should end up with `data/dev_20240627/dev.json` and
`data/dev_20240627/dev_databases/<db>/<db>.sqlite`.

## Grading

Execution accuracy, the BIRD metric: a prediction is correct when running it returns the
same multiset of rows as the gold query. Row order is ignored; row multiplicity is not;
`NULL` and `''` compare equal. A gold query that fails to run is reported separately from
an agent mistake.

## The baseline agent (M0)

Schema (with a few sample rows per table) plus the question and hint go in; one SQLite
query comes out. If it fails to execute, the agent sees the error and gets one retry.
No memory. This is the cold number every later version is measured against.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"          # Windows; use .venv/bin/pip elsewhere
```

Put your key in a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Optional overrides (defaults shown):

```
SQLAGENT_AGENT_MODEL=claude-haiku-4-5
SQLAGENT_REFLECTION_MODEL=claude-sonnet-5
SQLAGENT_COST_CAP_USD=10.0
```

## Running

```
python -m sqlagent.run_baseline --dry-run      # show the split, no API calls
python -m sqlagent.run_baseline --limit 5      # smoke test: 5 questions
python -m sqlagent.run_baseline                # full cold baseline, ~59 questions
pytest -q                                      # offline tests (grader, split, schema)
```

Each run writes a timestamped JSON report to `results/` with per-question rows and an
accuracy / cost / latency summary. The runner stops before it passes the spend cap.

## Cost

The agent runs on a cheap model with prompt caching on the schema block; a full cold run
over the test set is a few cents. The stronger model is only used later, for reflection.
