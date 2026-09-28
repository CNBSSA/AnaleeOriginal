"""Generate AI Insights must carry the CSRF token (QA 2026-09-28 #3).

Live QA: the button answered 400 "The CSRF token is missing" -- the form in
financial_insights.html posted with no csrf_token field while CSRFProtect is on.
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


def test_generate_insights_form_submits_with_csrf_enforced(csrf_client):
    app, client, _ = csrf_client
    page = client.get('/financial-insights').get_data(as_text=True)
    fields = _form_fields(page, 'generate-insights')
    assert 'csrf_token' in fields, 'Generate AI Insights form carries no CSRF token'
    resp = client.post('/generate-insights', data=fields)
    assert resp.status_code != 400, resp.get_data(as_text=True)[:300]
    assert resp.status_code in (301, 302)
    assert '/financial-insights' in resp.headers['Location']


def test_generate_insights_without_token_is_still_refused(csrf_client):
    """The fix adds the token; it does not exempt the route."""
    _app, client, _ = csrf_client
    assert client.post('/generate-insights', data={}).status_code == 400
