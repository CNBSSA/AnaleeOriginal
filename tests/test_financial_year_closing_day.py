"""Financial-year closing day: timestamps on the last day must count in-year.

``get_financial_year`` uses an exclusive ``end_date`` (start of the day after
the year closes). Every consumer must filter with ``date < end_date``, not
``<=`` midnight on the last calendar day.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from models import Account, CompanySettings, Transaction, User, db, transaction_in_financial_period
from reports.trial_balance_service import BANK_LINK_PREFIXES, load_trial_balance

PASSWORD = 'Sup3rSecret!'


def _user(fy_end: int = 12) -> int:
    user = User(username=f'fyclose{fy_end}', email=f'fyclose{fy_end}@example.com',
                subscription_status='active')
    user.set_password(PASSWORD)
    db.session.add(user)
    db.session.commit()
    db.session.add(CompanySettings(
        user_id=user.id, company_name='Close Day Co',
        registration_number='2020/123456/07', financial_year_end=fy_end))
    db.session.commit()
    return user.id


def _bank(user_id: int) -> Account:
    acc = Account(name='Bank', link='ca.810.001', category='Assets',
                  user_id=user_id, is_active=True)
    db.session.add(acc)
    db.session.flush()
    return acc


def _line(user_id: int, bank: Account, when: datetime, amount: float):
    signed = amount if bank.link.startswith(BANK_LINK_PREFIXES) else -amount
    db.session.add(Transaction(
        date=when, description='closing-day probe', amount=signed,
        user_id=user_id, account_id=bank.id))


class TestGetFinancialYearExclusiveEnd:
    def test_december_year_end_is_start_of_next_calendar_day(self, app):
        with app.app_context():
            uid = _user(fy_end=12)
            settings = CompanySettings.query.filter_by(user_id=uid).first()
            fy = settings.get_financial_year(date=datetime(2025, 6, 15))
        assert fy['end_date'] == datetime(2026, 1, 1)
        assert CompanySettings.financial_year_period_end_inclusive(
            fy['end_date']).date() == datetime(2025, 12, 31).date()


class TestClosingDayTransactionsInTrialBalance:
    @pytest.mark.parametrize('when', [
        datetime(2025, 12, 31, 8, 30),
        datetime(2025, 12, 31, 14, 30),
        datetime(2025, 12, 31, 23, 59, 59),
    ])
    def test_afternoon_and_late_closing_day_lines_are_in_year(self, app, when):
        with app.app_context():
            uid = _user(fy_end=12)
            bank = _bank(uid)
            _line(uid, bank, when, 250.0)
            db.session.commit()
            ctx = load_trial_balance(uid, as_at=datetime(2025, 12, 15))
            in_year = Transaction.query.filter(
                Transaction.user_id == uid,
                transaction_in_financial_period(
                    Transaction.date, ctx.start_date, ctx.end_date),
            ).count()
        assert in_year == 1

    def test_midnight_first_day_of_next_year_is_excluded(self, app):
        with app.app_context():
            uid = _user(fy_end=12)
            bank = _bank(uid)
            _line(uid, bank, datetime(2025, 12, 31, 12, 0), 100.0)
            _line(uid, bank, datetime(2026, 1, 1, 0, 0), 200.0)
            db.session.commit()
            ctx = load_trial_balance(uid, as_at=datetime(2025, 12, 15))
            rows = Transaction.query.filter(
                Transaction.user_id == uid,
                transaction_in_financial_period(
                    Transaction.date, ctx.start_date, ctx.end_date),
            ).all()
        assert len(rows) == 1
        assert rows[0].amount == 100.0


class TestClosingDayOnTrialBalance:
    def test_fy_mode_includes_late_closing_day(self, canary_app):
        app = canary_app
        with app.app_context():
            uid = _user(fy_end=12)
            sales = Account(link='i.100.000', name='Sales', category='Income',
                            user_id=uid, is_active=True)
            db.session.add(sales)
            db.session.flush()
            db.session.add(Transaction(
                date=datetime(2025, 12, 31, 16, 45), description='late sale',
                amount=500.0, user_id=uid, account_id=sales.id))
            db.session.commit()
            user = User.query.get(uid)
            email = user.email

        client = app.test_client()
        client.post('/auth/login', data={'email': email, 'password': PASSWORD})
        html = client.get(
            '/trial-balance?financial_year=2025',
        ).get_data(as_text=True)
        assert '500.00' in html or '500.0' in html
