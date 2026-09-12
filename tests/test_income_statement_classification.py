"""QA register #15 — the income statement reported profit equal to revenue.

``reports/routes.py`` classified accounts with strings that do not exist in the
chart's category vocabulary (``services/chart_seed_data.py`` gives every account
one of Assets / Liabilities / Equity / Income / Expenses):

    if account.category in ['Income', 'Revenue']:        # 'Income' matched
    elif account.category in ['Expense', 'Cost of Sales']:   # NOTHING matched

'Expense' is singular and 'Cost of Sales' is a *sub_category*, so no account ever
reached the expense branch: ``total_expenses`` was always 0 and the template's
``total_income - total_expenses`` reported **profit equal to revenue**.

The statement of financial position carried the identical defect on BOTH sides
(``'Asset'``, ``'Current Asset'``, ``'Fixed Asset'``, ``'Liability'`` — none of
them real categories), so assets and liabilities were always empty too.

These tests drive the real routes through the real app, so they assert what the
page actually renders. Sign convention: Analee transactions are single-legged and
carry the bank's own sign — a receipt is positive, a payment negative
(``bank_statements/services.py`` stores ``float(row['Amount'])`` unchanged).
"""
import re

import pytest

EMAIL = 'is15@example.com'
PASSWORD = 'TestPass123!'


def _page(client, url, heading):
    """Fetch a report page, proving it actually rendered.

    Vacuity guard. Several assertions below are of the form "this row is NOT on
    the page", which a 404 or an error redirect satisfies trivially — an early
    draft of this file passed four tests against a 404 because the URL was
    wrong. Nothing is asserted about a page that did not render.
    """
    resp = client.get(url)
    assert resp.status_code == 200, f'{url} did not render ({resp.status_code})'
    body = resp.get_data(as_text=True)
    assert heading in body, f'{url} rendered, but not as the expected report'
    return body


def _money(text, label):
    """The figure rendered in the row/total whose label is ``label``."""
    # Rows are '<td>Name</td><td class="text-end">123.45</td>' and totals are
    # the same shape with <th>. Tolerate either.
    pattern = (r'<t[dh][^>]*>\s*' + re.escape(label) +
               r'\s*</t[dh]>\s*<t[dh][^>]*>\s*(-?[\d,]+\.\d{2})\s*</t[dh]>')
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1).replace(',', ''))


@pytest.fixture
def books(canary_app):
    """A logged-in client with one year of income and expenses posted."""
    app = canary_app
    client = app.test_client()
    from models import db, User, Account, Transaction, CompanySettings, Entity
    from datetime import datetime

    client.post('/auth/register', data={
        'username': 'is15', 'email': EMAIL,
        'password': PASSWORD, 'confirm_password': PASSWORD})
    resp = client.post('/auth/login', data={'email': EMAIL, 'password': PASSWORD})
    assert '/login' not in resp.headers.get('Location', ''), 'login bounced'

    with app.app_context():
        user = User.query.filter_by(email=EMAIL).first()
        entity = Entity.query.order_by(Entity.name).first()
        settings = CompanySettings.query.filter_by(user_id=user.id).first()
        if settings is None:
            settings = CompanySettings(user_id=user.id)
            db.session.add(settings)
        settings.company_name = 'Fifteen Co'
        settings.registration_number = 'R15'
        settings.financial_year_end = 12          # calendar year
        if entity is not None and hasattr(settings, 'entity_id'):
            settings.entity_id = entity.id
        db.session.commit()

        uid = user.id

        def account(link, name, category, sub_category):
            acc = Account(link=link, name=name, category=category,
                          sub_category=sub_category, user_id=uid)
            db.session.add(acc)
            db.session.flush()
            return acc

        sales = account('i.100.000', 'Sales', 'Income', 'Sales')
        purchases = account('cos.002.000', 'Purchases', 'Expenses', 'Cost of Sales')
        bank_charges = account('e.321.000', 'Bank Charges', 'Expenses', 'Expenses')
        bank = account('ca.810.001', 'Bank Cheque Account 1', 'Assets', 'Current Asset')
        payables = account('cl.500.000', 'Trade Payables', 'Liabilities', 'Current Liability')
        # An untouched account, to prove unused chart lines are not listed.
        account('e.330.000', 'Cleaning', 'Expenses', 'Expenses')

        def txn(acc, when, amount):
            db.session.add(Transaction(date=when, description='line',
                                       amount=amount, user_id=uid,
                                       account_id=acc.id))

        # Receipts positive, payments negative.
        txn(sales, datetime(2025, 3, 10), 100000.0)
        txn(sales, datetime(2025, 6, 10), 50000.0)
        txn(purchases, datetime(2025, 4, 10), -40000.0)
        txn(bank_charges, datetime(2025, 5, 10), -1500.0)
        txn(bank, datetime(2025, 3, 10), 25000.0)
        txn(payables, datetime(2025, 4, 10), -9000.0)
        # Outside the year — must not reach the income statement.
        txn(sales, datetime(2024, 6, 10), 777000.0)
        db.session.commit()

    return app, client


# --------------------------------------------------------------------------
# The defect itself
# --------------------------------------------------------------------------

def test_expenses_appear_and_profit_is_not_revenue(books):
    app, client = books
    body = _page(client, '/income-statement?financial_year=2025', 'Total Income')

    assert _money(body, 'Purchases') == 40000.00, (
        'a Cost of Sales account never reached the expense side — its category '
        'is "Expenses"; "Cost of Sales" is the sub_category')
    assert _money(body, 'Bank Charges') == 1500.00

    # Revenue 150 000, expenses 41 500, so profit is 108 500 — NOT 150 000.
    assert _money(body, 'Total Income') == 150000.00
    assert _money(body, 'Total Expenses') == 41500.00, \
        'total expenses was 0.00, so the page reported profit equal to revenue'


def test_the_displayed_rows_sum_to_the_displayed_total(books):
    """The page's own arithmetic must hold.

    The old code displayed ``abs(balance)`` but added only conventionally-signed
    accounts, so a row could appear on the page and be absent from its total.
    """
    app, client = books
    body = _page(client, '/income-statement?financial_year=2025', 'Total Income')
    assert (_money(body, 'Purchases') + _money(body, 'Bank Charges')
            == _money(body, 'Total Expenses'))
    assert _money(body, 'Sales') == _money(body, 'Total Income')


def test_a_prior_year_receipt_is_not_in_this_years_statement(books):
    app, client = books
    body = _page(client, '/income-statement?financial_year=2025', 'Total Income')
    assert '777000.00' not in body.replace(',', ''), \
        "last year's receipt was reported as this year's income"


def test_an_account_that_never_traded_is_not_listed(books):
    app, client = books
    body = _page(client, '/income-statement?financial_year=2025', 'Total Income')
    assert _money(body, 'Cleaning') is None, \
        'every unused line of a ~1 000-account chart was listed, burying the figures'


def test_balance_sheet_accounts_stay_off_the_income_statement(books):
    app, client = books
    body = _page(client, '/income-statement?financial_year=2025', 'Total Income')
    assert _money(body, 'Bank Cheque Account 1') is None
    assert _money(body, 'Trade Payables') is None


# --------------------------------------------------------------------------
# The same defect on the statement of financial position
# --------------------------------------------------------------------------

def test_financial_position_reports_both_sides(books):
    app, client = books
    body = _page(client, '/financial-position?financial_year=2025', 'Total Assets')

    assert _money(body, 'Bank Cheque Account 1') == 25000.00, \
        'assets were always empty — "Assets" is the category, not "Current Asset"'
    assert _money(body, 'Trade Payables') == 9000.00, \
        'liabilities were always empty'
    assert _money(body, 'Total Assets') == 25000.00
    assert _money(body, 'Total Liabilities') == 9000.00


def test_financial_position_excludes_income_statement_accounts(books):
    app, client = books
    body = _page(client, '/financial-position?financial_year=2025', 'Total Assets')
    assert _money(body, 'Sales') is None
    assert _money(body, 'Purchases') is None
