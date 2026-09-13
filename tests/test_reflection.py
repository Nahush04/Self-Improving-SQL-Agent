from sqlagent.agent import Attempt
from sqlagent.reflection import ReflectionInput, _extract_lessons, reflect


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, system_blocks, prompt, max_tokens):
        self.calls += 1
        self.last_prompt = prompt
        return self.reply


def test_extract_lessons_from_clean_json():
    assert _extract_lessons('["lesson one", "lesson two"]') == ["lesson one", "lesson two"]


def test_extract_lessons_from_noisy_text_with_array_inside():
    text = 'Here is my answer:\n["the status column stores codes"]\nThanks.'
    assert _extract_lessons(text) == ["the status column stores codes"]


def test_extract_lessons_empty_array():
    assert _extract_lessons("[]") == []


def test_extract_lessons_no_array_returns_empty():
    assert _extract_lessons("no lessons here") == []


def test_extract_lessons_drops_blank_entries():
    assert _extract_lessons('["a real lesson", "  ", ""]') == ["a real lesson"]


def test_reflect_skips_llm_call_on_first_try_correct():
    llm = _FakeLLM("[]")
    r = ReflectionInput(
        db_id="financial",
        question="q",
        evidence="",
        attempts=[Attempt("SELECT 1", None)],
        correct=True,
    )
    assert reflect(llm, r) == []
    assert llm.calls == 0


def test_reflect_calls_llm_when_incorrect():
    llm = _FakeLLM('["revenue means net_amount, not gross_amount"]')
    r = ReflectionInput(
        db_id="financial",
        question="what was total revenue?",
        evidence="",
        attempts=[Attempt("SELECT SUM(gross_amount) FROM trans", None)],
        correct=False,
    )
    lessons = reflect(llm, r)
    assert lessons == ["revenue means net_amount, not gross_amount"]
    assert llm.calls == 1


def test_reflect_calls_llm_after_a_retry_even_if_eventually_correct():
    llm = _FakeLLM('["the status column stores codes, not text"]')
    r = ReflectionInput(
        db_id="financial",
        question="q",
        evidence="",
        attempts=[Attempt("bad sql", "no such column: status_text"), Attempt("good sql", None)],
        correct=True,
    )
    assert reflect(llm, r) == ["the status column stores codes, not text"]
    assert llm.calls == 1


def test_reflect_includes_gold_sql_only_when_provided():
    llm = _FakeLLM("[]")
    r = ReflectionInput(
        db_id="financial",
        question="q",
        evidence="",
        attempts=[Attempt("bad", "error"), Attempt("bad2", "error2")],
        correct=False,
        gold_sql="SELECT 1",
    )
    reflect(llm, r)
    assert "SELECT 1" in llm.last_prompt

    llm2 = _FakeLLM("[]")
    r2 = ReflectionInput(
        db_id="financial",
        question="q",
        evidence="",
        attempts=[Attempt("bad", "error"), Attempt("bad2", "error2")],
        correct=False,
    )
    reflect(llm2, r2)
    assert "Gold SQL" not in llm2.last_prompt
