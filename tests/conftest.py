import sqlite3

import pytest


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    """A small SQLite database wired in as db_id 'tiny' for grader/db tests."""
    path = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, city TEXT, age INTEGER);
        INSERT INTO person VALUES (1,'Ana','Delhi',30),(2,'Ben','Delhi',25),
                                  (3,'Cy','Pune',40),(4,'Di',NULL,25);
        """
    )
    conn.commit()
    conn.close()

    from sqlagent import db, schema

    monkeypatch.setattr(db, "db_path", lambda db_id: path)
    monkeypatch.setattr(schema, "db_path", lambda db_id: path)
    schema.schema_text.cache_clear()
    yield "tiny"
    schema.schema_text.cache_clear()
