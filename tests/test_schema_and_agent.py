from sqlagent.agent import extract_sql
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
