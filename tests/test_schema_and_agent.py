from sqlagent.agent import _memory_block, _user_prompt, answer, extract_sql
from sqlagent.schema import schema_text


def test_schema_lists_tables_and_sample_rows(tiny_db):
    text = schema_text(tiny_db)
    assert "CREATE TABLE person" in text
    assert "sample rows (person)" in text


def test_extract_sql_from_fenced_block():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_extract_sql_from_label_and_trailing_semicolon():
    assert extract_sql("SQL: SELECT a FROM t;") == "SELECT a FROM t"


def test_extract_sql_plain():
    assert extract_sql("SELECT count(*) FROM x") == "SELECT count(*) FROM x"


def test_memory_block_empty_when_no_memories():
    assert _memory_block(None) == ""
    assert _memory_block([]) == ""


def test_memory_block_renders_lessons_and_examples():
    memories = [
        {"kind": "lesson", "text": "status column stores codes not text", "metadata": {}},
        {
            "kind": "example",
            "text": "SELECT COUNT(*) FROM t",
            "metadata": {"question": "how many rows?", "sql": "SELECT COUNT(*) FROM t"},
        },
    ]
    block = _memory_block(memories)
    assert "status column stores codes not text" in block
    assert "Q: how many rows?" in block
    assert "SQL: SELECT COUNT(*) FROM t" in block


def test_memory_block_renders_episodes():
    memories = [
        {
            "kind": "episode",
            "question": "how many clients are there?",
            "final_sql": "SELECT COUNT(*) FROM client",
            "correct": False,
        }
    ]
    block = _memory_block(memories)
    assert "Past attempts on similar questions" in block
    assert "Q: how many clients are there?" in block
    assert "Tried SQL: SELECT COUNT(*) FROM client" in block
    assert "Outcome: incorrect" in block


def test_user_prompt_includes_memory_block():
    memories = [{"kind": "lesson", "text": "a lesson", "metadata": {}}]
    prompt = _user_prompt("a question", "", memories)
    assert "a question" in prompt
    assert "a lesson" in prompt


class _FakeLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen_prompts = []

    def complete(self, system_blocks, prompt, max_tokens):
        self.seen_prompts.append(prompt)
        return self._replies.pop(0)


def test_answer_folds_memories_into_prompt(tiny_db):
    llm = _FakeLLM(["SELECT COUNT(*) FROM person"])
    memories = [{"kind": "lesson", "text": "age is stored in years", "metadata": {}}]

    result = answer(llm, tiny_db, "how many people?", "", max_attempts=2, memories=memories)

    assert result.final_sql == "SELECT COUNT(*) FROM person"
    assert "age is stored in years" in llm.seen_prompts[0]


def test_answer_without_memories_omits_memory_block(tiny_db):
    llm = _FakeLLM(["SELECT COUNT(*) FROM person"])

    answer(llm, tiny_db, "how many people?", "", max_attempts=2, memories=None)

    assert "Lessons from past attempts" not in llm.seen_prompts[0]
