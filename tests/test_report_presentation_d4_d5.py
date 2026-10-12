"""QA D4/D5 — trial balance FY default and financial position equity section."""
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


def _money(text, label):
    pattern = (r'<t[dh][^>]*>\s*' + re.escape(label) +
               r'\s*</t[dh]>\s*<t[dh][^>]*>\s*(-?[\d,]+\.\d{2})\s*</t[dh]>')
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1).replace(',', ''))


def test_financial_position_lists_equity_and_balances_equation(canary_app):
    client = canary_app.test_client()
    from models import Entity

    client.post('/auth/register', data={
        'username': 'rpd5', 'email': 'rpd5@example.com',
        'password': PASSWORD, 'confirm_password': PASSWORD})
    client.post('/auth/login', data={'email': 'rpd5@example.com', 'password': PASSWORD})

    with canary_app.app_context():
        user = User.query.filter_by(email='rpd5@example.com').first()
        entity = Entity.query.order_by(Entity.name).first()
        settings = CompanySettings.query.filter_by(user_id=user.id).first()
        if settings is None:
            settings = CompanySettings(user_id=user.id)
            db.session.add(settings)
        settings.company_name = 'D5 Co'
        settings.financial_year_end = 12
        if entity is not None and hasattr(settings, 'entity_id'):
            settings.entity_id = entity.id
        db.session.commit()
        uid = user.id

        def account(link, name, category):
            acc = Account(link=link, name=name, category=category,
                          sub_category=category, user_id=uid)
            db.session.add(acc)
            db.session.flush()
            return acc

        bank = account('ca.810.001', 'Bank', 'Assets')
        payables = account('cl.500.000', 'Trade Payables', 'Liabilities')
        capital = account('q.100.000', 'Share Capital', 'Equity')
        db.session.add(Transaction(
            date=datetime(2025, 4, 1), description='cash',
            amount=25000.0, user_id=uid, account_id=bank.id))
        db.session.add(Transaction(
            date=datetime(2025, 4, 2), description='supplier',
            amount=-9000.0, user_id=uid, account_id=payables.id))
        db.session.add(Transaction(
            date=datetime(2025, 4, 3), description='opening',
            amount=-16000.0, user_id=uid, account_id=capital.id))
        db.session.commit()

    body = client.get('/financial-position?financial_year=2025').get_data(as_text=True)
    assert 'Equity' in body
    assert _money(body, 'Share Capital') == 16000.00
    assert _money(body, 'Total Assets') == 25000.00
    assert _money(body, 'Total Liabilities') == 9000.00
    assert _money(body, 'Total Equity') == 16000.00
    assert _money(body, 'Total Liabilities plus Equity') == 25000.00
