"""Work #11035 — optional CompanySettings fields must not render as the literal None."""
from __future__ import annotations

import re

import pytest

PASSWORD = 'Sup3rSecret!'


def _login_client_with_null_optionals(app, email='cs-null@test.com', username='csnulluser'):
    """Mirror tests/test_wo_chairman_stage3._login_client; optionals left SQL NULL."""
    from models import db, User, CompanySettings

    with app.app_context():
        user = User(username=username, email=email, subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(
            user_id=user.id,
            company_name='Provisioned Co',
            financial_year_end=2,
        ))
        db.session.commit()
        uid = user.id
        cs = CompanySettings.query.filter_by(user_id=uid).first()
        assert cs.registration_number is None
        assert cs.tax_number is None
        assert cs.vat_number is None
        assert cs.address is None

    client = app.test_client()
    client.post('/auth/login', data={'email': email, 'password': PASSWORD})
    return client, uid


def _input_value(html: str, field_id: str) -> str | None:
    match = re.search(
        rf'<input[^>]+id="{field_id}"[^>]*value="([^"]*)"',
        html,
        re.I,
    )
    return match.group(1) if match else None


def _textarea_body(html: str, field_id: str) -> str | None:
    match = re.search(
        rf'<textarea[^>]+id="{field_id}"[^>]*>(.*?)</textarea>',
        html,
        re.I | re.S,
    )
    return match.group(1) if match else None


def test_company_settings_get_null_optionals_render_empty(canary_app):
    client, _uid = _login_client_with_null_optionals(canary_app)
    resp = client.get('/company-settings')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    for field_id in ('registration_number', 'tax_number', 'vat_number'):
        val = _input_value(html, field_id)
        assert val == '', f'#{field_id} should be empty, got value="{val}"'
        assert 'value="None"' not in html.split(f'id="{field_id}"')[1].split('>')[0]

    address = _textarea_body(html, 'address')
    assert address == '', f'#address textarea should be empty, got {address!r}'
    assert 'None' not in (address or '')
