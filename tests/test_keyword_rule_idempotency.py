"""
Regression test for KeywordRule duplicate accumulation.

RuleManager.add_rule used to INSERT a new keyword_rule row unconditionally, so:
  * calling add_rule twice for the same (keyword, category) created duplicates;
  * HybridPredictor seeds its default rules on every __init__, so each
    instantiation re-inserted the same defaults, growing the table without bound.

add_rule is now idempotent on (keyword, category, is_regex): an existing rule is
not duplicated (and is reactivated if it had been deactivated). These tests lock
that in using an in-memory SQLite database.
"""
import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("flask_sqlalchemy")

from flask import Flask
from models import db, KeywordRule
from utils.rule_manager import RuleManager


def _make_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    return app


def test_add_rule_is_idempotent():
    app = _make_app()
    with app.app_context():
        db.create_all()
        rm = RuleManager()
        assert rm.add_rule("uber", "Travel") is True
        assert rm.add_rule("uber", "Travel") is True  # duplicate attempt
        rows = KeywordRule.query.filter_by(keyword="uber", category="Travel").all()
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"


def test_same_keyword_different_category_is_allowed():
    app = _make_app()
    with app.app_context():
        db.create_all()
        rm = RuleManager()
        rm.add_rule("gas", "Utilities")
        rm.add_rule("gas", "Travel")  # legitimately different rule
        assert KeywordRule.query.filter_by(keyword="gas").count() == 2


def test_add_rule_reactivates_deactivated_rule():
    app = _make_app()
    with app.app_context():
        db.create_all()
        rm = RuleManager()
        rm.add_rule("hotel", "Travel")
        rule = KeywordRule.query.filter_by(keyword="hotel", category="Travel").one()
        rule.is_active = False
        db.session.commit()

        assert rm.add_rule("hotel", "Travel") is True
        # still a single row, now reactivated
        rows = KeywordRule.query.filter_by(keyword="hotel", category="Travel").all()
        assert len(rows) == 1
        assert rows[0].is_active is True


def test_repeated_hybrid_predictor_init_does_not_multiply_rules():
    pytest.importorskip("flask_login")
    pytest.importorskip("Levenshtein")
    app = _make_app()
    with app.app_context():
        db.create_all()
        from utils.hybrid_predictor import HybridPredictor
        HybridPredictor()
        first = KeywordRule.query.count()
        assert first > 0
        HybridPredictor()  # second init must not re-insert the defaults
        assert KeywordRule.query.count() == first


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
