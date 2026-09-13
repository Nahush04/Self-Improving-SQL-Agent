from sqlagent.report import format_report


def _sample_report():
    return {
        "agent_model": "claude-haiku-4-5",
        "reflection_model": "claude-sonnet-5",
        "train_questions": 150,
        "test_questions": 59,
        "total_cost_usd": 2.0987,
        "phases": {
            "cold": {"questions": 59, "correct": 23, "accuracy": 0.3898},
            "train": {
                "questions": 150,
                "accuracy": 0.5733,
                "lesson_actions": {"added": 77, "merged": 30, "dropped": 5},
                "memory_growth": [
                    {"questions_seen": 25, "total_memories": 15},
                    {"questions_seen": 150, "total_memories": 107},
                ],
            },
            "warm": {"questions": 59, "correct": 26, "accuracy": 0.4407},
        },
    }


def test_format_report_includes_headline_numbers():
    text = format_report(_sample_report())
    assert "claude-haiku-4-5" in text
    assert "cold" in text
    assert "warm" in text
    assert "39.0%" in text
    assert "44.1%" in text


def test_format_report_includes_cold_to_warm_gap():
    text = format_report(_sample_report())
    assert "Cold -> warm gap: +5.1%" in text


def test_format_report_includes_memory_growth():
    text = format_report(_sample_report())
    assert "after   25 questions: 15 memories stored" in text
    assert "after  150 questions: 107 memories stored" in text
    assert "'added': 77" in text
