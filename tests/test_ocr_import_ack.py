"""Decision 2026-09-30 (delegated by the Chairman: user friendliness,
simplicity, effectiveness): a statement whose balances were read and do not
add up is NOT blocked from import, but importing it takes one deliberate tick
— "I've checked the flagged rows — import anyway". The review screen makes the
box required; the confirm route refuses without it, saving nothing."""
import pytest

pytest.importorskip("flask_sqlalchemy")
pytest.importorskip("flask_login")

from models import db, Transaction

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ocr_amount_balance_concat import (  # noqa: E402
    _bad_parse_result, _render, _rows, STATEMENT_TEXT)
from test_ocr_confirm_filter import _login, _make_app, _seed_user  # noqa: E402

ROWS = {"date": ["2026-03-01"], "description": ["TEST ROW"], "amount": ["10.00"],
        "include": ["0"], "has_include_filter": "1", "filename": "statement.pdf"}


def _post(extra):
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)
    resp = client.post("/ocr/statement/confirm", data={**ROWS, **extra})
    with app.app_context():
        count = Transaction.query.count()
    return resp, count, client


def test_an_unticked_mismatch_imports_nothing_and_says_why():
    resp, count, client = _post({"needs_ack": "1"})
    assert resp.status_code in (301, 302)
    assert count == 0
    with client.session_transaction() as sess:
        flashes = " ".join(m for _, m in sess.get("_flashes", []))
    assert "Nothing was imported" in flashes and "flagged rows" in flashes


def test_a_ticked_mismatch_imports():
    _, count, _ = _post({"needs_ack": "1", "ack_unreconciled": "1"})
    assert count == 1


def test_a_statement_that_needs_no_tick_imports_as_before():
    _, count, _ = _post({})
    assert count == 1


def test_review_asks_for_the_tick_only_when_the_statement_does_not_add_up(monkeypatch):
    from ocr import pdf_text_extraction
    from ocr.statement_extractor import extract_bank_statement
    from ocr.statement_integrity import self_audit
    bad = _bad_parse_result()
    body = _render(self_audit(bad), _rows(bad))
    assert 'name="ack_unreconciled"' in body and "required" in body
    assert "I've checked the flagged rows" in body or "I&#39;ve checked the flagged rows" in body
    monkeypatch.setattr(pdf_text_extraction, "extract_text", lambda b: STATEMENT_TEXT)
    ok = extract_bank_statement(b"%PDF-fake")
    body = _render(ok.report_card, [dict(r, duplicate=False) for r in ok.rows])
    assert 'name="ack_unreconciled"' not in body
