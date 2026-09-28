"""Forms that POST must carry the CSRF token (QA 2026-09-28 #2, #3).

Live QA: "Edit Account" and "Generate AI Insights" both answered 400 "The CSRF
token is missing". CSRFProtect is on app-wide; both templates posted a plain
<form> with no ``csrf_token`` field. These tests boot the real app WITH CSRF
enforcement on, submit each form exactly as the browser would (only the fields
the rendered form contains), and require the request to get through.
"""
import re

import pytest

PASSWORD = 'Sup3rSecret!'


def _form_fields(html, action_fragment=None):
    """Return the name->value pairs of the hidden inputs in the first POST form
    (optionally the one whose action contains ``action_fragment``)."""
    for match in re.finditer(r'<form\b([^>]*)>(.*?)</form>', html, re.S | re.I):
        attrs, body = match.group(1), match.group(2)
        if not re.search(r'method=["\']?post', attrs, re.I):
            continue
        if action_fragment and action_fragment not in attrs:
            continue
        return dict(re.findall(
            r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', body))
    raise AssertionError('form not found in page')


@pytest.fixture
def csrf_client(canary_app):
    from models import db, User, Account, CompanySettings

    app = canary_app
    with app.app_context():
        user = User(username='csrf', email='csrf@example.com',
                    subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(user_id=user.id, company_name='Co',
                                       financial_year_end=2))
        account = Account(name='Office Rent', link='ce.100.001', category='Expenses',
                          user_id=user.id, is_active=True)
        db.session.add(account)
        db.session.commit()
        account_id = account.id

    client = app.test_client()
    resp = client.post('/auth/login', data={'email': 'csrf@example.com',
                                            'password': PASSWORD})
    assert '/login' not in resp.headers.get('Location', '')
    # Enforce CSRF from here on, as production does.
    app.config['WTF_CSRF_ENABLED'] = True
    return app, client, account_id


def test_edit_account_form_submits_with_csrf_enforced(csrf_client):
    app, client, account_id = csrf_client
    page = client.get(f'/account/{account_id}/edit').get_data(as_text=True)
    fields = _form_fields(page)
    assert 'csrf_token' in fields, 'Edit Account form carries no CSRF token'
    fields.update({'link': 'ce.100.001', 'name': 'Rent', 'category': 'Expenses',
                   'sub_category': ''})
    resp = client.post(f'/account/{account_id}/edit', data=fields)
    assert resp.status_code != 400, resp.get_data(as_text=True)[:300]
    from models import Account
    with app.app_context():
        assert Account.query.get(account_id).name == 'Rent'


def test_post_without_token_is_still_refused(csrf_client):
    """The fix adds the token; it does not exempt the routes."""
    _app, client, account_id = csrf_client
    assert client.post(f'/account/{account_id}/edit',
                       data={'link': 'x', 'name': 'x', 'category': 'x'}).status_code == 400
