"""QA register #13 and #12 — what a trial balance for a financial year means.

**#13.** Every account was cumulative, income and expenses included, so a client
with two years of data produced a 2025 trial balance containing 2024's trading as
well — and THE ACCOUNTANTS, which compiles the AFS from this payload, counted
those prior years again. Balance-sheet accounts must carry forward; income and
expense accounts must show the year's movement.

**#12.** Accounts were selected by filtering the JOINED transactions table on
``date >= start AND date <= end``, which turns an outer join into an effective
inner one. An account was therefore listed only if it had activity INSIDE the
year, so a balance-sheet account carrying a prior-year balance and no current-year
movement vanished from the trial balance while its balance was not zero.

The safety property that makes #13 a correction rather than a gamble: for a client
whose data lies inside ONE financial year both rules give the same number. That is
asserted here, not assumed.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from models import Account, CompanySettings, Transaction, User, db
from reports.trial_balance_service import (
    BANK_LINK_PREFIXES, SUSPENSE_LINK, load_trial_balance)

# February year-end: FY2025 runs 1 Mar 2025 → 28 Feb 2026.
FY2025 = datetime(2025, 6, 30)      # a date inside FY2025
IN_FY2025 = datetime(2025, 9, 1)
IN_FY2024 = datetime(2024, 9, 1)    # the prior year
IN_FY2026 = datetime(2026, 9, 1)    # the next year

_seq = iter(range(1, 10_000))


def _client(fy_end: int = 2) -> int:
    n = next(_seq)
    user = User(username=f'sem{n}', email=f'sem{n}@example.com',
                subscription_status='active')
    user.set_password('password')
    db.session.add(user)
    db.session.commit()
    db.session.add(CompanySettings(
        user_id=user.id, company_name='ACME Pty Ltd',
        registration_number='2020/123456/07', financial_year_end=fy_end))
    db.session.commit()
    return user.id


def _account(user_id: int, link: str, name: str, category: str) -> Account:
    acc = Account(link=link, name=name, category=category,
                  sub_category=category, user_id=user_id)
    db.session.add(acc)
    db.session.flush()
    return acc


def _txn(user_id: int, account: Account, when: datetime, amount: float):
    """Post ``amount`` to ``account`` the way the ledger reads it (debit +).

    A real Analee row is a BANK line — money in positive — and the trial
    balance posts its category side as the negative of that (2026-09-28
    double-entry fix). So the bank line that produces a ledger amount of
    ``amount`` on the category carries ``-amount``. Expected figures in these
    tests are therefore unchanged from the single-legged era.

    A line still sitting on a bank account is that bank's own movement, and a
    bank line's sign already IS the bank's ledger sign, so it is not negated.
    """
    bank_signed = amount if account.link.startswith(BANK_LINK_PREFIXES) else -amount
    db.session.add(Transaction(date=when, description='line', amount=bank_signed,
                               user_id=user_id, account_id=account.id))


def _amount_for(ctx, link: str):
    for row in ctx.rows:
        if row.link == link:
            return row.amount
    return None


def _links(ctx) -> set[str]:
    return {a.link for a in ctx.accounts}


class TestIncomeAndExpensesCarryTheYearsMovementOnly:
    def test_prior_year_income_is_excluded(self, app):
        with app.app_context():
            uid = _client()
            sales = _account(uid, 'i.100.000', 'Sales', 'Income')
            _txn(uid, sales, IN_FY2024, -400.0)   # last year
            _txn(uid, sales, IN_FY2025, -100.0)   # this year
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert _amount_for(ctx, 'i.100.000') == Decimal('-100.00'), (
            'the 2025 trial balance reported 2024 income as well — the AFS would '
            'count those receipts twice')
        # Double entry (2026-09-28): last year's income is not lost, it is in
        # opening equity, and the trial balance balances.
        assert _amount_for(ctx, 'q.200.000') == Decimal('-400.00')
        assert ctx.total_credits == Decimal('500.00') == ctx.total_debits

    def test_prior_year_expenses_are_excluded(self, app):
        with app.app_context():
            uid = _client()
            bank_charges = _account(uid, 'e.321.000', 'Bank Charges', 'Expenses')
            _txn(uid, bank_charges, IN_FY2024, -90.0)
            _txn(uid, bank_charges, IN_FY2025, -30.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert _amount_for(ctx, 'e.321.000') == Decimal('-30.00')

    def test_next_year_income_is_also_excluded(self, app):
        """The 2026-09-07 fix already bounded the top end; it must stay bounded."""
        with app.app_context():
            uid = _client()
            sales = _account(uid, 'i.100.000', 'Sales', 'Income')
            _txn(uid, sales, IN_FY2025, -100.0)
            _txn(uid, sales, IN_FY2026, -700.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert _amount_for(ctx, 'i.100.000') == Decimal('-100.00')

    def test_it_agrees_with_the_income_statement_rule(self, app):
        """The income statement scopes P&L with `Transaction.date.between(...)`.
        The trial balance must produce the same figure — these two reports
        disagreeing is the defect, not the rounding."""
        with app.app_context():
            uid = _client()
            sales = _account(uid, 'i.100.000', 'Sales', 'Income')
            for when, amount in ((IN_FY2024, -11.0), (IN_FY2025, -22.0),
                                 (datetime(2025, 12, 1), -33.0), (IN_FY2026, -44.0)):
                _txn(uid, sales, when, amount)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)
            from sqlalchemy import func
            income_statement_figure = db.session.query(
                func.sum(Transaction.amount)).filter(
                Transaction.account_id == sales.id,
                Transaction.date.between(ctx.start_date, ctx.end_date)).scalar()

        # The income statement sums bank-signed lines (money in +); the trial
        # balance states the same figure as a credit.
        assert _amount_for(ctx, 'i.100.000') == -Decimal(
            str(round(income_statement_figure, 2)))


class TestBalanceSheetAccountsStillCarryForward:
    def test_an_asset_keeps_its_prior_year_balance(self, app):
        with app.app_context():
            uid = _client()
            bank = _account(uid, 'ca.810.001', 'Bank Cheque Account 1', 'Assets')
            _txn(uid, bank, IN_FY2024, 400.0)
            _txn(uid, bank, IN_FY2025, 100.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert _amount_for(ctx, 'ca.810.001') == Decimal('500.00'), (
            'a bank balance must be everything that ever happened to it')

    @pytest.mark.parametrize('category', ['Assets', 'Liabilities', 'Equity'])
    def test_every_balance_sheet_category_carries_forward(self, app, category):
        with app.app_context():
            uid = _client()
            acc = _account(uid, 'x.001.000', f'{category} account', category)
            _txn(uid, acc, IN_FY2024, 60.0)
            _txn(uid, acc, IN_FY2025, 40.0)
            db.session.commit()
            ctx = load_trial_balance(uid, as_at=FY2025)
        assert _amount_for(ctx, 'x.001.000') == Decimal('100.00')

    def test_an_unknown_category_keeps_the_old_cumulative_behaviour(self, app):
        """Fail safe: an account we cannot classify must not silently lose its
        prior years."""
        with app.app_context():
            uid = _client()
            odd = _account(uid, 'z.999.000', 'Legacy account', '')
            _txn(uid, odd, IN_FY2024, 60.0)
            _txn(uid, odd, IN_FY2025, 40.0)
            db.session.commit()
            ctx = load_trial_balance(uid, as_at=FY2025)
        assert _amount_for(ctx, 'z.999.000') == Decimal('100.00')


class TestASingleYearClientIsUnaffected:
    """The #13 safety property: for a client whose data lies inside one year,
    the movement rule and the cumulative rule agree — nothing rolls into
    Retained Earnings. Written 2026-09-28 against a real statement: its bank
    account recorded on the upload, lines bank-signed, posted double entry."""

    def test_income_expense_and_asset_are_all_unchanged(self, app):
        from models import UploadedFile
        with app.app_context():
            uid = _client()
            bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            sales = _account(uid, 'i.100.000', 'Sales', 'Income')
            fees = _account(uid, 'e.321.000', 'Bank Charges', 'Expenses')
            statement = UploadedFile(filename='fnb.csv', user_id=uid,
                                     bank_account_id=bank.id)
            db.session.add(statement)
            db.session.flush()
            for when, amount, account in ((IN_FY2025, 200.0, sales),
                                          (datetime(2025, 11, 5), -50.0, fees)):
                db.session.add(Transaction(date=when, description='line',
                                           amount=amount, user_id=uid,
                                           account_id=account.id,
                                           file_id=statement.id))
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert _amount_for(ctx, 'ca.810.001') == Decimal('150.00')
        assert _amount_for(ctx, 'i.100.000') == Decimal('-200.00')
        assert _amount_for(ctx, 'e.321.000') == Decimal('50.00')
        assert _amount_for(ctx, 'q.200.000') is None
        assert ctx.total_debits == Decimal('200.00')
        assert ctx.total_credits == Decimal('200.00')


class TestACarriedBalanceIsNeverDropped:
    """#12 — the row set."""

    def test_a_dormant_asset_with_a_prior_year_balance_is_listed(self, app):
        with app.app_context():
            uid = _client()
            dormant = _account(uid, 'na.030.000', 'Motor Vehicles - Cost', 'Assets')
            active = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            _txn(uid, dormant, IN_FY2024, 120_000.0)   # bought last year, idle since
            _txn(uid, active, IN_FY2025, 10.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert 'na.030.000' in _links(ctx), (
            'an asset carrying R120 000 vanished from the trial balance because it '
            'had no movement this year')
        assert _amount_for(ctx, 'na.030.000') == Decimal('120000.00')

    def test_a_dormant_pnl_account_is_correctly_absent(self, app):
        """The twin, and the reason the two findings belong together: under the
        new semantics last year's income has no 2025 movement, so it has nothing
        to report and must NOT appear."""
        with app.app_context():
            uid = _client()
            old_sales = _account(uid, 'i.100.000', 'Sales', 'Income')
            active = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            _txn(uid, old_sales, IN_FY2024, -5_000.0)
            _txn(uid, active, IN_FY2025, 10.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert 'i.100.000' not in _links(ctx)
        assert _amount_for(ctx, 'i.100.000') is None

    def test_an_in_period_account_netting_to_zero_is_still_listed(self, app):
        """Iron Rule: that row was displayed as 0.00 before and still must be."""
        with app.app_context():
            uid = _client()
            bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            _txn(uid, bank, IN_FY2025, 50.0)
            _txn(uid, bank, datetime(2025, 10, 1), -50.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert 'ca.810.001' in _links(ctx)
        assert _amount_for(ctx, 'ca.810.001') is None   # zero rows are not exported

    def test_an_account_with_no_transactions_at_all_is_still_excluded(self, app):
        """Unchanged: the page must not render the whole ~1000-line chart."""
        with app.app_context():
            uid = _client()
            _account(uid, 'e.999.000', 'Never used', 'Expenses')
            used = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            _txn(uid, used, IN_FY2025, 10.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert 'e.999.000' not in _links(ctx)

    def test_an_account_listed_once_however_many_transactions_it_has(self, app):
        """Selection is by id through a distinct sub-select, so the row set cannot
        depend on transaction count — the previous join made that a live risk."""
        with app.app_context():
            uid = _client()
            bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            for day in range(1, 13):
                _txn(uid, bank, datetime(2025, 9, day), 5.0)
            db.session.commit()

            ctx = load_trial_balance(uid, as_at=FY2025)

        assert [a.link for a in ctx.accounts].count('ca.810.001') == 1
        assert _amount_for(ctx, 'ca.810.001') == Decimal('60.00')

    def test_one_clients_accounts_never_appear_in_anothers(self, app):
        """The selection query was rewritten, so the tenant filter is re-proven."""
        with app.app_context():
            mine = _client()
            theirs = _client()
            my_bank = _account(mine, 'ca.810.001', 'Bank', 'Assets')
            their_bank = _account(theirs, 'ca.810.002', 'Their Bank', 'Assets')
            _txn(mine, my_bank, IN_FY2025, 10.0)
            _txn(theirs, their_bank, IN_FY2025, 999.0)
            db.session.commit()

            ctx = load_trial_balance(mine, as_at=FY2025)

        # The bank side of lines with no recorded statement is its own row.
        # Its bank line has no category yet, so it waits in Suspense.
        assert _links(ctx) - {SUSPENSE_LINK} == {'ca.810.001'}
        assert _amount_for(ctx, 'ca.810.002') is None
