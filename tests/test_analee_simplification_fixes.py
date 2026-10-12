"""Analee simplification list, 2026-09-30 (Festus: "go on the Analee list").

Enhancements to existing screens only — no feature added or removed, no menu
changed. Source: autonomusFV docs/analee/ANALEE_SIMPLIFICATION_INVENTORY_2026-09-30.md.
These lock the behaviour changes; wording-only items are covered where a user
would otherwise be misled. TEST data only.
"""
import os
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(HERE, rel), encoding='utf-8') as fh:
        return fh.read()


# ---- site-wide ---------------------------------------------------------------

def test_the_stylesheet_that_hides_the_tutorial_is_loaded():
    base = _read('templates/base.html')
    assert "filename='styles.css'" in base
    assert '.tutorial-overlay' in _read('static/styles.css')


def test_error_messages_get_the_red_style():
    base = _read('templates/base.html')
    assert "'error': 'danger'" in base


def test_messages_are_shown_once():
    for rel in ('templates/icountant.html', 'templates/ocr/review.html',
                'templates/ocr/statement_upload.html'):
        assert 'get_flashed_messages' not in _read(rel), rel


def test_analyze_page_speaks_rand_and_plain_words():
    page = _read('templates/analyze.html')
    assert '${{' not in page
    for jargon in ('> ASF', '> ESF', 'ERF:', '(ERF only)', '"AI Suggest"'):
        assert jargon not in page, jargon
    assert 'Suggest account' in page and 'Suggest explanation' in page
    assert 'Auto-process next 10' in page  # the frozen work loop's button is unchanged


# ---- a person's edit is recorded as theirs, and never over the client's -----

def _seed(app, source=None, explanation=None):
    from models import Account, Transaction, UploadedFile, User, db
    with app.app_context():
        user = User(username='simp', email='simp@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()
        acc = Account(link='ex.100.000', name='TEST Rent', category='Expenses', user_id=user.id)
        up = UploadedFile(filename='TEST.pdf', user_id=user.id, upload_date=datetime.utcnow())
        db.session.add_all([acc, up])
        db.session.commit()
        txn = Transaction(date=datetime(2026, 2, 4), description='TEST line', amount=-100.0,
                          user_id=user.id, file_id=up.id, explanation=explanation,
                          explanation_source=source)
        db.session.add(txn)
        db.session.commit()
        return user.id, acc.id, txn.id


def _call(app, user_id, path, view, *args, json=None):
    from flask_login import login_user
    from models import User, db
    with app.test_request_context(path, method='POST', json=json):
        login_user(db.session.get(User, user_id))
        rv = view(*args)
    return (rv[0].get_json(), rv[1]) if isinstance(rv, tuple) else (rv.get_json(), rv.status_code)


def test_an_edit_is_badged_as_yours_not_analees(canary_app):
    from models import Transaction, db
    from routes import save_transaction
    uid, acc, tid = _seed(canary_app, source='ai', explanation='Machine wrote this')
    _body, status = _call(canary_app, uid, f'/analyze/save-transaction/{tid}', save_transaction,
                          tid, json={'account_id': acc, 'explanation': 'TEST rent for Feb'})
    assert status == 200
    with canary_app.app_context():
        t = db.session.get(Transaction, tid)
        assert (t.explanation, t.explanation_source) == ('TEST rent for Feb', 'accountant')


def test_the_clients_explanation_is_kept_and_the_account_still_saves(canary_app):
    from models import Transaction, db
    from routes import save_transaction, update_explanation
    uid, acc, tid = _seed(canary_app, source='client', explanation='Client said: rent')
    body, status = _call(canary_app, uid, f'/analyze/save-transaction/{tid}', save_transaction,
                         tid, json={'account_id': acc, 'explanation': 'Overwrite'})
    assert status == 200 and 'kept' in body['message']
    body, status = _call(canary_app, uid, '/update_explanation', update_explanation,
                         json={'transaction_id': tid, 'explanation': 'Overwrite',
                               'description': 'TEST line'})
    assert status == 409 and 'kept' in body['error']
    with canary_app.app_context():
        t = db.session.get(Transaction, tid)
        assert t.explanation == 'Client said: rent' and t.account_id == acc


# ---- forms say why -----------------------------------------------------------

def test_a_taken_username_is_named_on_the_form(canary_app):
    from forms.auth import RegistrationForm
    from models import User, db
    with canary_app.app_context():
        u = User(username='taken-name', email='taken@x.test')
        u.set_password('pw12345!')
        db.session.add(u)
        db.session.commit()
    with canary_app.test_request_context('/auth/register', method='POST', data={
            'username': 'taken-name', 'email': 'new@x.test',
            'password': 'Passw0rd!long', 'confirm_password': 'Passw0rd!long'}):
        form = RegistrationForm()
        form.validate()
        assert 'That username is taken' in ' '.join(form.errors.get('username', []))


def test_database_errors_are_not_shown_to_the_user():
    from bank_statements.services import BankStatementService
    svc = BankStatementService.__new__(BankStatementService)
    msg = svc.get_friendly_error_message('db_error', 'psycopg2.errors.UniqueViolation: key')
    assert 'psycopg2' not in msg
    assert 'Missing column' in svc.get_friendly_error_message('processing_error', 'Missing column Date')


def test_a_pdf_import_lands_on_the_statement_it_created():
    from test_ocr_invalid_dates import _confirm, _login, _make_app, _seed_user
    from models import UploadedFile
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)
    r = _confirm(client, dates=['2026-03-15'], descriptions=['TEST'], amounts=['10.00'])
    with app.app_context():
        fid = UploadedFile.query.filter_by(user_id=uid).one().id
    assert r.status_code == 302 and r.headers['Location'].endswith(f'/analyze/{fid}')
