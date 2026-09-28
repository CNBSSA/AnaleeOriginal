"""Dashboard: Transaction Count and the Financial Year selector (QA 2026-09-28 #4).

Live QA: "Transaction Count blank and year selector empty". dashboard.html asks
for ``transaction_count``, ``financial_years`` and ``current_year``; the route
passed none of them, so Jinja rendered nothing. The Income/Expenses cards were
labelled "For current financial year" but summed the current calendar MONTH.
"""
import re
from datetime import datetime

PASSWORD = 'Sup3rSecret!'


def _setup(app, fy_end=2):
    from models import db, User, Account, CompanySettings, Transaction
    with app.app_context():
        user = User(username='dash', email='dash@example.com',
                    subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(user_id=user.id, company_name='Co',
                                       financial_year_end=fy_end))
        bank = Account(name='Bank', link='ca.810.001', category='Assets',
                       user_id=user.id, is_active=True)
        db.session.add(bank)
        db.session.commit()
        rows = [
            # FY 2024/2025 (Mar 2024 - Feb 2025)
            (datetime(2024, 6, 1), 'Old deposit', 100.0),
            (datetime(2025, 1, 15), 'Old fee', -10.0),
            # FY 2025/2026 (Mar 2025 - Feb 2026)
            (datetime(2025, 4, 3), 'Salary', 15000.0),
            (datetime(2025, 5, 3), 'Rent', -5000.0),
            (datetime(2026, 2, 28), 'Fee', -50.0),
        ]
        for when, desc, amount in rows:
            db.session.add(Transaction(date=when, description=desc, amount=amount,
                                       user_id=user.id, account_id=bank.id))
        db.session.commit()
    client = app.test_client()
    client.post('/auth/login', data={'email': 'dash@example.com', 'password': PASSWORD})
    return client


def _count_card(html):
    match = re.search(r'Transaction Count</h5>\s*<h2>\s*([^<]*)</h2>', html)
    assert match, 'Transaction Count card not found'
    return match.group(1).strip()


def _options(html):
    select = re.search(r'<select[^>]*id="financialYear"[^>]*>(.*?)</select>', html, re.S)
    assert select, 'financial year selector not found'
    return re.findall(r'<option value="(\d+)"\s*(selected)?\s*>\s*([^<]*?)\s*</option>',
                      select.group(1))


def test_year_selector_lists_the_users_financial_years(canary_app):
    client = _setup(canary_app)
    html = client.get('/dashboard?financial_year=2025').get_data(as_text=True)
    options = _options(html)
    values = [v for v, _sel, _label in options]
    assert '2024' in values and '2025' in values
    selected = [v for v, sel, _label in options if sel]
    assert selected == ['2025']
    labels = {v: label for v, _sel, label in options}
    assert labels['2025'] == 'FY 2025/2026'


def test_transaction_count_and_totals_are_for_the_selected_year(canary_app):
    client = _setup(canary_app)
    html = client.get('/dashboard?financial_year=2025').get_data(as_text=True)
    assert _count_card(html) == '3'
    assert '15000.00' in html          # income in FY 2025/2026
    assert '5050.00' in html           # expenses in FY 2025/2026

    html = client.get('/dashboard?financial_year=2024').get_data(as_text=True)
    assert _count_card(html) == '2'


def test_unknown_year_falls_back_and_never_renders_blank(canary_app):
    client = _setup(canary_app)
    html = client.get('/dashboard?financial_year=1999').get_data(as_text=True)
    assert _count_card(html) != ''
    assert any(sel for _v, sel, _l in _options(html))


def test_december_year_end_uses_calendar_years(canary_app):
    client = _setup(canary_app, fy_end=12)
    html = client.get('/dashboard?financial_year=2025').get_data(as_text=True)
    labels = {v: label for v, _sel, label in _options(html)}
    assert labels['2025'] == 'FY 2025'
    # 2025-04-03, 2025-05-03 are in calendar 2025; 2025-01-15 as well.
    assert _count_card(html) == '3'
