"""Work-order #103: the statement reader dropped the money-out sign.

A PDF whose text keeps each figure but not the column it stood in (separate
Debit / Credit columns, or a Dr / Cr marker) was read as a column of plain
positive amounts. The review screen and the "does not add up" check were right;
they were handed the wrong signs. Ticking "import anyway" then saved every
payment as money received, and the trial balance showed expenses as income.

Two causes in the reader (ingress, not the frozen analysis engine):

1. ``ocr.bank_profiles._AMOUNT_RE`` did not capture a trailing ``Dr`` / ``Cr``,
   although ``normalize_amount`` already understands one, so the marker was
   thrown away before it could be read - on amounts, on running balances and on
   the opening / closing balances.
2. With no marker at all, nothing used the one thing a statement always proves:
   its own running balance. A row whose balance falls by exactly its amount is
   money out.

The inference changes a row only when the figures prove it to the cent. A
statement that already prints a minus, and a row the figures cannot explain,
read exactly as before. TEST data only.
"""
import os
import sys
from decimal import Decimal
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import statement_extractor as se  # noqa: E402
from ocr.bank_profiles import _AMOUNT_RE  # noqa: E402
from ocr.statement_integrity import (  # noqa: E402
    ExtractionResult, StatementHeader, StatementLine,
)

D = Decimal

# Separate Debit / Credit columns: the text keeps the figure, not the column.
DEBIT_CREDIT = """TEST Bank Statement
Statement Period : 01/03/2026 to 31/03/2026
Date Description Debit Credit Balance
Opening Balance 10,000.00
01/03/2026 Salary ACME 20,000.00 30,000.00
02/03/2026 Rent Landlord 8,500.00 21,500.00
04/03/2026 Checkers POS 1,200.50 20,299.50
09/03/2026 Refund Takealot 350.00 20,649.50
15/03/2026 Insurance Debit Order 2,000.00 18,649.50
20/03/2026 Bank Fee 145.00 18,504.50
Closing Balance 18,504.50
"""

# A Dr / Cr marker, with the account dipping into overdraft and back.
DR_CR = """TEST Bank Statement
Statement Period : 01/03/2026 to 31/03/2026
Date Description Amount Balance
Opening Balance 1,000.00Cr
05/03/2026 ATM Withdrawal 2,000.00Dr 1,000.00Dr
10/03/2026 Salary ACME 5,000.00Cr 4,000.00Cr
12/03/2026 Rent Landlord 4,500.00Dr 500.00Dr
Closing Balance 500.00Dr
"""

# Already signed: a trailing minus on every payment (the case that always worked).
ALREADY_SIGNED = """TEST Bank Statement
Statement Period : 01/03/2026 to 31/03/2026
Date Description Amount Balance
Opening Balance 10,000.00
01/03/2026 Salary ACME 20,000.00 30,000.00
02/03/2026 Rent Landlord 8,500.00- 21,500.00
04/03/2026 Checkers POS 1,200.50- 20,299.50
Closing Balance 20,299.50
"""


def _extract(text):
    with mock.patch("ocr.pdf_text_extraction.extract_text", return_value=text):
        return se.extract_bank_statement(b"%PDF-1.4 dummy")


def _amounts(extraction):
    return [D(str(r["amount"])) for r in extraction.rows]


# ---- the reported statement ------------------------------------------------------

def test_debit_credit_statement_reads_payments_as_money_out():
    got = _extract(DEBIT_CREDIT)
    assert got.error is None, got.error
    assert _amounts(got) == [D("20000.00"), D("-8500.00"), D("-1200.50"),
                             D("350.00"), D("-2000.00"), D("-145.00")]


def test_debit_credit_statement_now_balances_and_asks_for_no_tick_box():
    rc = _extract(DEBIT_CREDIT).report_card
    assert rc.reconciled is True
    assert rc.variance == D("0.00")
    assert rc.running_balance_breaks == []
    assert not [e for e in rc.errors if e.get("code") == "MATH_GATE_FAILED"]


def test_dr_cr_statement_reads_signs_and_the_overdrawn_balances():
    got = _extract(DR_CR)
    assert got.error is None, got.error
    assert _amounts(got) == [D("-2000.00"), D("5000.00"), D("-4500.00")]
    assert got.header.opening_balance == D("1000.00")
    assert got.header.closing_balance == D("-500.00"), "an overdrawn 'Dr' balance is negative"
    assert got.report_card.reconciled is True


# ---- the marker, on its own ------------------------------------------------------

def test_the_amount_pattern_captures_a_trailing_dr_or_cr():
    assert _AMOUNT_RE.findall("2,000.00Dr 1,000.00Dr") == ["2,000.00Dr", "1,000.00Dr"]
    assert _AMOUNT_RE.findall("5,000.00 CR 4,000.00 cr") == ["5,000.00 CR", "4,000.00 cr"]


def test_a_word_that_merely_starts_with_dr_or_cr_is_not_a_marker():
    assert _AMOUNT_RE.findall("1,500.00 Credit Note") == ["1,500.00"]
    assert _AMOUNT_RE.findall("150.00 Driver Fee") == ["150.00"]


# ---- safety: nothing that already worked changes ---------------------------------

def test_a_statement_that_prints_a_minus_reads_exactly_as_before():
    got = _extract(ALREADY_SIGNED)
    assert _amounts(got) == [D("20000.00"), D("-8500.00"), D("-1200.50")]
    assert got.report_card.reconciled is True


def _result(rows, opening, closing=None):
    return ExtractionResult(
        header=StatementHeader(opening_balance=opening, closing_balance=closing),
        lines=[StatementLine(date="2026-03-0%d" % (i + 1), description=f"row {i}",
                             amount=a, balance=b) for i, (a, b) in enumerate(rows)],
    )


def test_a_row_the_figures_cannot_explain_is_left_as_read_and_still_flagged():
    # 500.00 against a balance that moved by 123.45: neither +500 nor -500.
    res = _result([(D("500.00"), D("1123.45"))], D("1000.00"))
    out = se._infer_money_out_signs(res)
    assert [ln.amount for ln in out.lines] == [D("500.00")]


def test_a_row_with_no_balance_is_never_guessed():
    res = _result([(D("500.00"), None), (D("200.00"), D("300.00"))], D("1000.00"))
    out = se._infer_money_out_signs(res)
    # The chain breaks at the unbalanced row, exactly where self_audit's does:
    # row 2 has no known previous balance, so it is left alone.
    assert [ln.amount for ln in out.lines] == [D("500.00"), D("200.00")]


def test_the_first_row_needs_a_known_opening_balance():
    res = _result([(D("500.00"), D("500.00"))], None)
    assert [ln.amount for ln in se._infer_money_out_signs(res).lines] == [D("500.00")]


def test_an_already_negative_row_is_never_flipped_back():
    res = _result([(D("-500.00"), D("500.00"))], D("1000.00"))
    assert [ln.amount for ln in se._infer_money_out_signs(res).lines] == [D("-500.00")]


def test_inference_does_not_mutate_its_input():
    res = _result([(D("500.00"), D("500.00"))], D("1000.00"))
    out = se._infer_money_out_signs(res)
    assert out.lines[0].amount == D("-500.00")
    assert res.lines[0].amount == D("500.00")
