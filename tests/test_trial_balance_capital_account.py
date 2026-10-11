"""Capital Account (q.100.000) must not absorb prior-year P&L roll-forward.

When Retained Earnings (q.200.000) is missing from the chart, roll-forward must
use a synthetic q.200.000 row — not Capital, which should reflect only postings
to that account.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from models import Account, CompanySettings, Transaction, UploadedFile, User, db
from reports.trial_balance_service import load_trial_balance

FY2026 = datetime(2026, 6, 30)   # Feb year-end: FY2026 runs 1 Mar 2026 – 28 Feb 2027
IN_FY2025 = datetime(2025, 9, 1)
IN_FY2026 = datetime(2026, 9, 1)
_seq = iter(range(1, 10_000))


def _client() -> int:
    n = next(_seq)
    user = User(username=f'cap{n}', email=f'cap{n}@example.com',
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


def _statement(uid, bank):
    f = UploadedFile(filename='s.csv', user_id=uid, bank_account_id=bank.id)
    db.session.add(f)
    db.session.flush()
    return f


def _line(uid, statement, account, amount, when):
    db.session.add(Transaction(date=when, description='line', amount=amount,
                               user_id=uid, account_id=account.id,
                               file_id=statement.id))


def _rows(ctx):
    return {r.link: r.amount for r in ctx.rows}


def test_capital_stays_zero_when_only_retained_is_missing_from_chart(app):
    """Regression: prior-year expenses rolled into q.100.000 when q.200 absent."""
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        capital = _account(uid, 'q.100.000', 'Capital Account', 'Equity')
        rent = _account(uid, 'e.400.000', 'Lease Rental - Premises', 'Expenses')
        utilities = _account(uid, 'e.410.000', 'Electricity', 'Expenses')
        s = _statement(uid, bank)
        _line(uid, s, rent, -20_000.0, IN_FY2025)
        _line(uid, s, utilities, -5_301.24, IN_FY2025)
        _line(uid, s, rent, -8_101.11, IN_FY2026)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=FY2026)

    rows = _rows(ctx)
    assert capital.link not in rows, (
        'Capital must not appear when it has no postings and no opening balance')
    assert rows['e.400.000'] == Decimal('8101.11')
    assert 'e.410.000' not in rows
    # Prior-year expense movement rolled to retained, not capital.
    prior_expense_debits = Decimal('20000.00') + Decimal('5301.24')
    assert rows['q.200.000'] == prior_expense_debits
    assert sum(rows.values()) == 0


def test_actual_capital_postings_still_show_on_capital(app):
    with app.app_context():
        uid = _client()
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        capital = _account(uid, 'q.100.000', 'Share Capital - Ordinary', 'Equity')
        retained = _account(uid, 'q.200.000', 'Retained Earnings', 'Equity')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        s = _statement(uid, bank)
        _line(uid, s, capital, 50_000.0, IN_FY2025)   # contribution in
        _line(uid, s, sales, 1_000.0, IN_FY2025)      # prior-year trading
        _line(uid, s, sales, 200.0, IN_FY2026)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=FY2026)

    rows = _rows(ctx)
    assert rows['q.100.000'] == Decimal('-50000.00')
    assert rows['q.200.000'] == Decimal('-1000.00')
    assert rows['i.100.000'] == Decimal('-200.00')
