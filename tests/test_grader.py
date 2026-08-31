from sqlagent.grader import grade


def test_exact_match(tiny_db):
    g = grade(tiny_db, "SELECT name FROM person WHERE city='Delhi'",
              "SELECT name FROM person WHERE city='Delhi'")
    assert g.correct and g.reason == "match"


def test_order_does_not_matter(tiny_db):
    g = grade(tiny_db, "SELECT name FROM person ORDER BY name DESC",
              "SELECT name FROM person ORDER BY name ASC")
    assert g.correct


def test_row_multiplicity_matters(tiny_db):
    # gold returns two rows (age 25 twice); prediction collapses them
    g = grade(tiny_db, "SELECT DISTINCT age FROM person WHERE age=25",
              "SELECT age FROM person WHERE age=25")
    assert not g.correct and g.reason == "mismatch"


def test_null_and_empty_string_compare_equal(tiny_db):
    g = grade(tiny_db, "SELECT city FROM person WHERE id=4",
              "SELECT '' ")
    assert g.correct


def test_prediction_that_does_not_run(tiny_db):
    g = grade(tiny_db, "SELECT nope FROM person", "SELECT name FROM person")
    assert not g.correct and g.reason.startswith("pred_error")


def test_empty_prediction(tiny_db):
    g = grade(tiny_db, "   ", "SELECT name FROM person")
    assert not g.correct and g.reason == "pred_error:empty"


def test_broken_gold_is_flagged_separately(tiny_db):
    g = grade(tiny_db, "SELECT name FROM person", "SELECT bad FROM person")
    assert not g.correct and g.reason.startswith("gold_error")
