"""Unit 1.6 — default to the latest financial year that has data when the current FY is empty."""
import re
from datetime import datetime

from chat.routes import get_financial_context
from models import Account, CompanySettings, Transaction, User, db

PASSWORD = 'Sup3rSecret!'

# Live "today" during cloud runs is 2026-10; Feb year-end => current FY start 2026.
_NOW_YEAR = datetime.now().year


def _seed_historical_only(app, n_rows=5):
    """Transactions only in FY starting 2021 (June 2021); none in the current FY."""
    with app.app_context():
        user = User(username='fyctx', email='fyctx@example.com', subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(
            CompanySettings(user_id=user.id, company_name='Co', financial_year_end=2))
        bank = Account(
            name='Bank', link='ca.810.001', category='Assets',
            user_id=user.id, is_active=True)
        db.session.add(bank)
        db.session.commit()
        for i in range(n_rows):
            db.session.add(Transaction(
                date=datetime(2021, 6, 1 + i),
                description=f'Historical {i}',
                amount=1000.0,
                user_id=user.id,
                account_id=bank.id,
            ))
        db.session.commit()
        uid = user.id
    client = app.test_client()
    client.post('/auth/login', data={'email': 'fyctx@example.com', 'password': PASSWORD})
    return client, uid


def _count_card(html):
    match = re.search(r'Transaction Count</h5>\s*<h2>\s*([^<]*)</h2>', html)
    assert match, 'Transaction Count card not found'
    return match.group(1).strip()


def _selected_fy(html):
    select = re.search(r'<select[^>]*id="financialYear"[^>]*>(.*?)</select>', html, re.S)
    assert select, 'financial year selector not found'
    for value, selected, _label in re.findall(
            r'<option value="(\d+)"\s*(selected)?\s*>\s*([^<]*?)\s*</option>',
            select.group(1)):
        if selected:
            return int(value)
    return None


def test_dashboard_defaults_to_latest_fy_with_data_when_current_is_empty(canary_app):
    client, _uid = _seed_historical_only(canary_app, n_rows=7)
    html = client.get('/dashboard').get_data(as_text=True)
    assert _selected_fy(html) == 2021
    assert _count_card(html) == '7'


def test_chat_context_uses_same_fy_as_dashboard(canary_app):
    _client, uid = _seed_historical_only(canary_app, n_rows=12)
    with canary_app.app_context():
        ctx = get_financial_context(uid)
    assert ctx['total_transactions'] == 12
    assert ctx['income'] == 12000.0
    assert ctx['period_start'].startswith('2021-03-01') or ctx['period_start'].startswith('2021-03-02')


def test_financial_insights_loads_transactions_from_default_fy(canary_app, monkeypatch):
    client, uid = _seed_historical_only(canary_app, n_rows=4)
    captured = {}

    def _capture_render(_template, **kwargs):
        captured.update(kwargs)
        return 'ok'

    monkeypatch.setattr('routes.render_template', _capture_render)
    resp = client.get('/financial-insights')
    assert resp.status_code == 200
    assert len(captured['transactions']) == 4
    assert captured['transactions'][0].description.startswith('Historical')
    assert captured['start_date'].year == 2021


def test_explicit_empty_current_fy_still_honoured(canary_app):
    """Choosing the empty current FY explicitly must not be overridden."""
    client, _uid = _seed_historical_only(canary_app, n_rows=3)
    # Current FY start year for Feb year-end in Oct 2026 is 2026.
    html = client.get(f'/dashboard?financial_year={_NOW_YEAR}').get_data(as_text=True)
    assert _selected_fy(html) == _NOW_YEAR
    assert _count_card(html) == '0'
