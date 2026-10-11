"""Unit 4.2 — Dead UI and polish (TEST data only)."""
import os
import re
from datetime import datetime

import pytest

pytest.importorskip('flask_sqlalchemy')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSWORD = 'TestPassw0rd!long'


def _read(rel):
    with open(os.path.join(HERE, rel), encoding='utf-8') as fh:
        return fh.read()


def test_nav_shows_login_or_logout_not_both():
    base = _read('templates/base.html')
    assert 'href="/auth/login">Login</a>' in base
    assert 'href="{{ url_for(\'auth.logout\') }}">Logout</a>' in base
    assert '{% if current_user.is_authenticated %}' in base
    assert '{% else %}' in base
    assert 'include "admin/layout.html"' not in base


def test_tutorial_does_not_auto_open_on_first_visit():
    tutorial = _read('static/js/analyze/tutorial.js')
    assert 'showTutorial()' not in tutorial.split('initializeTutorial')[1].split('setupEventListeners')[0]
    assert 'export default new TutorialManager()' not in tutorial
    main = _read('static/js/analyze/main.js')
    assert 'tutorial.js' not in main


def test_batch_progress_wording_is_plain():
    assert 'processed in this run' in _read('templates/analyze.html')
    assert 'processed in this run' in _read('static/js/analyze/batchProcessor.js')
    assert 'reviewed' not in _read('templates/analyze.html').split('batchProgressText')[1][:80]


def test_user_facing_amounts_use_rand_not_dollar_in_key_templates():
    for rel in (
        'templates/expense_forecast.html',
        'templates/output.html',
        'templates/goals_dashboard.html',
        'templates/predictions/trends.html',
    ):
        text = _read(rel)
        assert '${{' not in text, rel
        assert 'R{{' in text, rel


def test_chart_forms_use_dropdowns():
    settings = _read('templates/settings.html')
    assert '<select' in settings and 'name="category"' in settings
    assert 'name="category"' in settings and 'type="text"' not in settings.split('name="category"')[0][-200:]
    edit = _read('templates/edit_account.html')
    assert '<select' in edit and 'chart_categories' in edit


def test_trial_balance_explains_nonzero_suspense():
    tb = _read('templates/reports/trial_balance.html')
    assert 'tbSuspenseHelp' in tb
    assert 'Analyze Data' in tb
    assert 'suspense_balance' in tb


def test_settings_route_passes_chart_choices(canary_app):
    from models import CompanySettings, User, db

    with canary_app.app_context():
        user = User(username='ui42', email='ui42@test.example', subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(
            user_id=user.id, company_name='TEST UI Co', financial_year_end=2))
        db.session.commit()

    client = canary_app.test_client()
    client.post('/auth/login', data={'email': 'ui42@test.example', 'password': PASSWORD})
    r = client.get('/settings')
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'name="category"' in body and '<select' in body
    assert 'Assets' in body and 'Expenses' in body


def test_login_page_nav_for_guest_has_login_not_logout(canary_app):
    r = canary_app.test_client().get('/auth/login')
    body = r.get_data(as_text=True)
    assert 'nav-link' in body and '>Login</a>' in body
    assert '>Logout</a>' not in body


def test_authenticated_nav_has_logout_not_login(canary_app):
    from models import User, db

    with canary_app.app_context():
        user = User(username='nav42', email='nav42@test.example', subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()

    client = canary_app.test_client()
    client.post('/auth/login', data={'email': 'nav42@test.example', 'password': PASSWORD})
    r = client.get('/settings', follow_redirects=True)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert '>Logout</a>' in body
    assert re.search(r'nav-link[^>]*>Login</a>', body) is None
