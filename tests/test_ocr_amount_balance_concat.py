"""Regression: a bank-statement row's amount must never be glued onto the digits
beside it, and a parse that does not reconcile must never be shown as a success.

Live, 2026-09-30: a TEST statement parsed 6 rows, two of them at ~R10 000 000
(figures read as one number), the statement was R20 million out — and the
review screen still said "Extraction successful" with "100%" confidence.
"""
import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import pdf_text_extraction  # noqa: E402
from ocr.bank_profiles import GENERIC, parse_transaction_lines  # noqa: E402
from ocr.statement_extractor import extract_bank_statement  # noqa: E402
from ocr.statement_integrity import (  # noqa: E402
    ExtractionResult,
    StatementHeader,
    StatementLine,
    normalize_amount,
    self_audit,
)

# Six rows. Three narrations end in digits right before the amount (the old
# token rule read "INV 1000 1 500.00-" as -R10 001 500.00), and one row has the
# amount and balance columns abutting with no space at all.
STATEMENT_TEXT = """
Test Bank Statement
Opening Balance 10 000.00
01/03/2026 SALARY ACME 1 500.00 11 500.00
02/03/2026 EFT PAYMENT INV 1000 1 500.00- 10 000.00
03/03/2026 CARD 4521 250.00- 9 750.00
04/03/2026 DEPOSIT 2 000.0011 750.00
05/03/2026 BANK FEE 45.00- 11 705.00
06/03/2026 TRANSFER REF 2002 3 000.00- 8 705.00
Closing Balance 8 705.00
"""

EXPECTED = [
    ("SALARY ACME", Decimal("1500.00"), Decimal("11500.00")),
    ("EFT PAYMENT INV 1000", Decimal("-1500.00"), Decimal("10000.00")),
    ("CARD 4521", Decimal("-250.00"), Decimal("9750.00")),
    ("DEPOSIT", Decimal("2000.00"), Decimal("11750.00")),
    ("BANK FEE", Decimal("-45.00"), Decimal("11705.00")),
    ("TRANSFER REF 2002", Decimal("-3000.00"), Decimal("8705.00")),
]


def test_amount_and_balance_are_read_as_two_figures():
    lines = parse_transaction_lines(STATEMENT_TEXT, GENERIC)
    got = [(ln.description, ln.amount, ln.balance) for ln in lines]
    assert got == EXPECTED


def test_abutting_statement_reconciles_end_to_end(monkeypatch):
    monkeypatch.setattr(pdf_text_extraction, "extract_text", lambda b: STATEMENT_TEXT)
    outcome = extract_bank_statement(b"%PDF-fake")
    assert outcome.ok and outcome.method == "digital_pdf"
    assert [r["amount"] for r in outcome.rows] == [float(a) for _, a, _ in EXPECTED]
    card = outcome.report_card
    assert card.reconciled, card.errors
    assert card.variance == Decimal("0.00")
    assert card.suspect_lines == []


def test_two_figures_run_together_are_refused_not_summed():
    # Old behaviour: "1 500 8 650.25" -> R15 008 650.25.
    with pytest.raises(ValueError):
        normalize_amount("1 500 8 650.25")
    with pytest.raises(ValueError):
        normalize_amount("1 500.00 8 650.25")
    # Proper thousands grouping is still accepted.
    assert normalize_amount("12 345 678.90") == Decimal("12345678.90")
    assert normalize_amount("R 1,234.56") == Decimal("1234.56")


def _bad_parse_result():
    """The live defect's shape: six rows, two ~R10m, R20m out."""
    rows = [
        ("2026-03-01", "SALARY ACME", "1500.00"),
        ("2026-03-02", "EFT PAYMENT", "10008650.25"),
        ("2026-03-03", "CARD CHECKERS", "-250.00"),
        ("2026-03-04", "DEPOSIT", "2000.00"),
        ("2026-03-05", "TRANSFER", "10050990.00"),
        ("2026-03-06", "BANK FEE", "-45.00"),
    ]
    return ExtractionResult(
        header=StatementHeader(opening_balance=Decimal("10000.00"),
                               closing_balance=Decimal("13205.25")),
        lines=[StatementLine(date=d, description=t, amount=a) for d, t, a in rows],
    )


def test_non_reconciling_parse_names_the_implausible_rows():
    card = self_audit(_bad_parse_result())
    assert not card.reconciled
    assert card.variance == Decimal("-20059640.00")
    named = [(s["date"], s["description"], s["amount"]) for s in card.suspect_lines]
    assert named == [
        ("2026-03-02", "EFT PAYMENT", "10008650.25"),
        ("2026-03-05", "TRANSFER", "10050990.00"),
    ]


def test_a_large_but_reconciled_payment_is_not_flagged():
    result = ExtractionResult(
        header=StatementHeader(opening_balance=Decimal("0.00"),
                               closing_balance=Decimal("4999800.00")),
        lines=[
            StatementLine(date="2026-03-01", description="PROPERTY SALE", amount="5000000.00"),
            StatementLine(date="2026-03-02", description="FEE", amount="-100.00"),
            StatementLine(date="2026-03-03", description="FEE", amount="-100.00"),
        ],
    )
    card = self_audit(result)
    assert card.reconciled
    assert card.suspect_lines == []


def _render(report_card, rows):
    pytest.importorskip("flask_sqlalchemy")
    pytest.importorskip("flask_login")
    from flask import Flask, render_template
    from flask_wtf import CSRFProtect
    from ocr import ocr as ocr_bp

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(__name__, template_folder=os.path.join(root, "templates"))
    app.config.update(SECRET_KEY="test", WTF_CSRF_ENABLED=False)
    CSRFProtect(app)  # csrf_token() template global
    from flask_login import LoginManager
    lm = LoginManager()
    lm.init_app(app)
    lm.user_loader(lambda uid: None)
    app.register_blueprint(ocr_bp)
    with app.test_request_context("/ocr/statement"):
        return render_template(
            "ocr/review.html", rows=rows, accounts=[], account_id="",
            filename="test.pdf", statement_header=None,
            report_card=report_card, extraction_method="digital_pdf",
        )


def _rows(result):
    return [{"date": ln.date, "description": ln.description,
             "amount": float(ln.amount), "confidence": 1.0, "duplicate": False}
            for ln in result.lines]


def test_non_reconciling_review_never_claims_success_or_100_percent():
    result = _bad_parse_result()
    body = _render(self_audit(result), _rows(result))
    assert "Extraction successful" not in body
    assert "100%" not in body
    assert "does not fully reconcile" in body
    assert "Variance: R -20059640.00" in body
    # The offending rows are named with date, description and amount.
    assert "2026-03-02 · EFT PAYMENT · R 10008650.25" in body
    assert "2026-03-05 · TRANSFER · R 10050990.00" in body
    assert body.count("Check amount") == 2
    # Confirm & import is still offered (blocking is the Chairman's decision).
    assert "Confirm &amp; import" in body


def test_reconciled_review_still_shows_success(monkeypatch):
    monkeypatch.setattr(pdf_text_extraction, "extract_text", lambda b: STATEMENT_TEXT)
    outcome = extract_bank_statement(b"%PDF-fake")
    rows = [dict(r, duplicate=False) for r in outcome.rows]
    body = _render(outcome.report_card, rows)
    assert "Extraction successful" in body
    assert "Statement balances." in body
    assert "100%" in body
    assert "Check amount" not in body
