"""The standalone trial balance is double entry (Festus re-open, 2026-09-28).

Analee stores one bank-signed row per bank line (money in +). The trial balance
used to sum that straight onto the category: income read as a debit, expenses
as a credit, the bank account was missing and the totals did not balance, so
THE ACCOUNTANTS imported the profit with its sign inverted.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from models import Account, CompanySettings, Transaction, UploadedFile, User, db
from reports.trial_balance_service import (
    SUSPENSE_LINK, UNRECORDED_BANK_LINK, load_trial_balance)

AS_AT = datetime(2025, 6, 30)   # Feb year-end: FY runs 1 Mar 2025 – 28 Feb 2026
_seq = iter(range(1, 10_000))


def _client() -> int:
    n = next(_seq)
    user = User(username=f'de{n}', email=f'de{n}@example.com',
                subscription_status='active')
    user.set_password('password')
    db.session.add(user)
    db.session.commit()
    db.session.add(CompanySettings(user_id=user.id, company_name='TEST',
                                   financial_year_end=2))
    db.session.commit()
    return user.id


def _account(uid, link, name, category):
    acc = Account(link=link, name=name, category=category, sub_category=category,
                  user_id=uid)
    db.session.add(acc)
    db.session.flush()
    return acc


def _statement(uid, bank=None):
    f = UploadedFile(filename='s.csv', user_id=uid,
                     bank_account_id=bank.id if bank else None)
    db.session.add(f)
    db.session.flush()
    return f


def _line(uid, statement, account, amount, when=datetime(2025, 5, 1)):
    db.session.add(Transaction(date=when, description='line', amount=amount,
                               user_id=uid, account_id=account.id if account else None,
                               file_id=statement.id if statement else None))


def _rows(ctx):
    return {r.link: r.amount for r in ctx.rows}


def test_sales_in_and_rent_out_give_the_right_profit_and_balance(app):
    """The QA reproduction: R15 000 sales in, R5 000 rent out."""
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank Cheque Account 1', 'Assets')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        rent = _account(uid, 'e.400.000', 'Lease Rental - Premises', 'Expenses')
        s = _statement(uid, bank)
        _line(uid, s, sales, 15000.0)
        _line(uid, s, rent, -5000.0)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=AS_AT)

    rows = _rows(ctx)
    assert rows['ca.810.001'] == Decimal('10000.00')    # bank: debit
    assert rows['i.100.000'] == Decimal('-15000.00')    # income: credit
    assert rows['e.400.000'] == Decimal('5000.00')      # expense: debit
    assert sum(rows.values()) == 0
    assert ctx.total_debits == ctx.total_credits == Decimal('15000.00')
    profit = -(rows['i.100.000'] + rows['e.400.000'])
    assert profit == Decimal('10000.00')


def test_an_uncategorised_line_waits_in_suspense(app):
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        s = _statement(uid, bank)
        _line(uid, s, bank, -250.0)        # still on the bank: not categorised
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=AS_AT)
    assert _rows(ctx) == {'ca.810.001': Decimal('-250.00'),
                          SUSPENSE_LINK: Decimal('250.00')}


def test_an_older_statement_infers_its_bank_from_a_stamped_line(app):
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.002', 'Bank Cheque Account 2', 'Assets')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        s = _statement(uid)                 # no bank recorded (legacy)
        _line(uid, s, bank, -20.0)          # one line still stamped with it
        _line(uid, s, sales, 100.0)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=AS_AT)
    rows = _rows(ctx)
    assert rows['ca.810.002'] == Decimal('80.00')
    assert rows['i.100.000'] == Decimal('-100.00')
    assert UNRECORDED_BANK_LINK not in rows


def test_an_unknowable_bank_is_named_not_guessed(app):
    """Two banks in use and a legacy statement with every line categorised:
    the bank side goes to one clearly named row, never onto either bank."""
    with app.app_context():
        uid = _client()
        b1 = _account(uid, 'ca.810.001', 'Bank 1', 'Assets')
        b2 = _account(uid, 'ca.810.002', 'Bank 2', 'Assets')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        _statement(uid, b1)
        _statement(uid, b2)
        legacy = _statement(uid)
        _line(uid, legacy, sales, 300.0)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=AS_AT)
    rows = _rows(ctx)
    assert rows[UNRECORDED_BANK_LINK] == Decimal('300.00')
    assert 'ca.810.001' not in rows and 'ca.810.002' not in rows
    assert sum(rows.values()) == 0


def test_the_only_bank_in_use_is_used_for_a_legacy_statement(app):
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        _statement(uid, bank)
        legacy = _statement(uid)
        _line(uid, legacy, sales, 300.0)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=AS_AT)
    assert _rows(ctx) == {'ca.810.001': Decimal('300.00'),
                          'i.100.000': Decimal('-300.00')}


def test_earlier_years_trading_rolls_into_retained_earnings_and_balances(app):
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        retained = _account(uid, 'q.200.000', 'Retained Earnings', 'Equity')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        s = _statement(uid, bank)
        _line(uid, s, sales, 400.0, when=datetime(2024, 9, 1))   # prior year
        _line(uid, s, sales, 100.0)                              # this year
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=AS_AT)
    assert retained.link == 'q.200.000'
    assert _rows(ctx) == {'ca.810.001': Decimal('500.00'),
                          'i.100.000': Decimal('-100.00'),
                          'q.200.000': Decimal('-400.00')}


def test_a_statement_naming_another_clients_bank_is_not_trusted(app):
    with app.app_context():
        mine = _client()
        theirs = _client()
        their_bank = _account(theirs, 'ca.810.009', 'Their bank', 'Assets')
        sales = _account(mine, 'i.100.000', 'Sales', 'Income')
        s = _statement(mine, their_bank)
        _line(mine, s, sales, 50.0)
        db.session.commit()
        ctx = load_trial_balance(mine, as_at=AS_AT)
    rows = _rows(ctx)
    assert 'ca.810.009' not in rows
    assert rows[UNRECORDED_BANK_LINK] == Decimal('50.00')


def test_the_csv_import_records_the_statements_bank(app):
    from bank_statements.services import BankStatementService
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        db.session.commit()
        record = BankStatementService().create_uploaded_file_record(
            'fnb.csv', uid, bank_account_id=bank.id)
        assert db.session.get(UploadedFile, record.id).bank_account_id == bank.id
