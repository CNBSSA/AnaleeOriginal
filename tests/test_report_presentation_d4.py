"""QA D4 — trial balance default financial year matches dashboard workspace FY."""
import re
from datetime import datetime

from models import Account, CompanySettings, Transaction, User, db

PASSWORD = 'Sup3rSecret!'


def _seed_fy_2025_only(app):
    """Feb year-end; transactions in FY 2025/26 only (not the empty current FY)."""
    with app.app_context():
        user = User(username='rpd4', email='rpd4@example.com', subscription_status='active')
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
        for i in range(3):
            db.session.add(Transaction(
                date=datetime(2025, 6, 1 + i),
                description=f'FY25 line {i}',
                amount=500.0,
                user_id=user.id,
                account_id=bank.id,
            ))
        db.session.commit()
    client = app.test_client()
    client.post('/auth/login', data={'email': 'rpd4@example.com', 'password': PASSWORD})
    return client


def _selected_tb_fy(html):
    select = re.search(
        r'<select[^>]*id="financial_year"[^>]*>(.*?)</select>', html, re.S)
    assert select, 'trial balance financial year selector not found'
    for value, selected, _label in re.findall(
            r'<option value="(\d+)"\s*(selected)?\s*>\s*([^<]*?)\s*</option>',
            select.group(1)):
        if selected:
            return int(value)
    return None


def test_trial_balance_defaults_to_workspace_fy_when_current_is_empty(canary_app):
    client = _seed_fy_2025_only(canary_app)
    html = client.get('/trial-balance').get_data(as_text=True)
    assert html.count('Trial Balance') >= 1
    assert _selected_tb_fy(html) == 2025


def test_trial_balance_explicit_current_fy_still_honoured(canary_app):
    client = _seed_fy_2025_only(canary_app)
    current_fy = datetime.now().year if datetime.now().month > 2 else datetime.now().year - 1
    html = client.get(f'/trial-balance?financial_year={current_fy}').get_data(as_text=True)
    assert _selected_tb_fy(html) == current_fy
