"""FNB statements read no transactions (2026-09-30, Festie's FNB brief).

Two defects in the Tier-1 reader (ocr/bank_profiles.py), both verified on the
code before fixing:
1. FNB prints transaction dates without a year ("01 Feb") and the year only in
   the header; the reader accepted only dates WITH a year, so no line matched.
2. The header filter dropped any short line containing "credit" or "debit", so
   real transactions such as "Magtape Credit Medihelp" or "Debit Order" were
   thrown away as column headings.
The fix gives a year-less date the statement's year (anchored on the latest
dated line, so an old account-opening date cannot pull it back) and treats the
column-heading words as a header only when the line does not start with a date.
TEST data only.
"""
import os
import sys
from decimal import Decimal
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.bank_profiles import FNB, parse_transaction_lines  # noqa: E402
from ocr import pdf_text_extraction  # noqa: E402

FNB_TEXT = """FNB First National Bank
TEST Account Statement
Statement Period : 01 January 2022 to 31 January 2022
Date Description Amount Balance
Opening Balance 7,150.25
01 Jan Magtape Credit Medihelp Smh0363383 1,500.00 8,650.25
05 Jan POS Purchase Checkers 150.00- 8,500.25
31 Jan Debit Order Insurance 200.00- 8,300.25
Closing Balance 8,300.25
"""


def test_year_less_fnb_lines_are_read_with_the_statement_year():
    lines = parse_transaction_lines(FNB_TEXT, FNB)
    assert [l.date for l in lines] == ['2022-01-01', '2022-01-05', '2022-01-31']
    assert [l.amount for l in lines] == [Decimal('1500.00'), Decimal('-150.00'), Decimal('-200.00')]


def test_lines_containing_credit_or_debit_are_kept():
    descs = [l.description for l in parse_transaction_lines(FNB_TEXT, FNB)]
    assert 'Magtape Credit Medihelp Smh0363383' in descs
    assert 'Debit Order Insurance' in descs


def test_headings_and_balance_lines_are_still_not_transactions():
    descs = ' '.join(l.description for l in parse_transaction_lines(FNB_TEXT, FNB)).lower()
    assert 'balance' not in descs
    assert 'description' not in descs
    assert len(parse_transaction_lines(FNB_TEXT, FNB)) == 3


def test_a_header_row_of_column_words_is_skipped():
    text = "Date Description Debit Credit Balance 0.00 0.00\n" + FNB_TEXT
    assert len(parse_transaction_lines(text, FNB)) == 3


def test_december_line_on_a_january_statement_is_the_year_before():
    text = "Statement date 15 January 2023\n28 Dec Salary 100.00 200.00\n10 Jan Fee 5.00- 195.00\n"
    assert [l.date for l in parse_transaction_lines(text, FNB)] == ['2022-12-28', '2023-01-10']


def test_an_old_opening_date_does_not_pull_the_year_back():
    text = "Account opened 03 March 2015\nStatement date 31 March 2022\n15 Mar Fee 5.00- 95.00\n"
    assert [l.date for l in parse_transaction_lines(text, FNB)] == ['2022-03-15']


def test_dated_lines_are_unchanged():
    text = "FNB\n15/03/2026 CARD PURCHASE 150.00- 850.00\n01 April 2022 Statement header line\n"
    lines = parse_transaction_lines(text, FNB)
    assert [(l.date, l.amount) for l in lines] == [('2026-03-15', Decimal('-150.00'))]


def test_the_whole_fnb_pdf_reconciles():
    with mock.patch.object(pdf_text_extraction, 'extract_text', return_value=FNB_TEXT):
        result = pdf_text_extraction.extract_pdf_statement(b'%PDF TEST')
    assert len(result.lines) == 3
    assert result.header.opening_balance == Decimal('7150.25')
    assert result.header.closing_balance == Decimal('8300.25')
    assert result.header.opening_balance + sum(l.amount for l in result.lines) == result.header.closing_balance
