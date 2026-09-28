"""Admin approval before a standalone trial balance goes to THE ACCOUNTANTS.

Festus, 2026-09-28: "Approve before a standalone trial balance goes to THE
ACCOUNTANTS." Every transmission path refuses until an Analee administrator
has approved these exact figures; the Excel download is untouched.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from models import Account, CompanySettings, Transaction, TrialBalanceApproval, UploadedFile, User, db
from reports import tb_approval
from reports.tb_share_tokens import create_share_token
from reports.trial_balance_service import load_trial_balance


def _client(app, *, admin=False, n=0):
    with app.app_context():
        user = User(username=f'ap{n}{"a" if admin else ""}', email=f'ap{n}{"a" if admin else ""}@x.com',
                    subscription_status='active', is_admin=admin)
        user.set_password('pw12345678')
        db.session.add(user)
        db.session.commit()
        uid = user.id
        if not admin:
            db.session.add(CompanySettings(user_id=uid, company_name=f'TEST {n}',
                                           financial_year_end=2))
            bank = Account(link='ca.810.001', name='Bank', category='Assets',
                           sub_category='Current Asset', user_id=uid)
            sales = Account(link='i.100.000', name='Sales', category='Income',
                            sub_category='Income', user_id=uid)
            db.session.add_all([bank, sales])
            db.session.flush()
            f = UploadedFile(filename='s.csv', user_id=uid, bank_account_id=bank.id)
            db.session.add(f)
            db.session.flush()
            db.session.add(Transaction(date=datetime.now(), description='Sale', amount=100.0,
                                       user_id=uid, account_id=sales.id, file_id=f.id))
            db.session.commit()
    client = app.test_client()
    client.post('/auth/login', data={'email': f'ap{n}{"a" if admin else ""}@x.com',
                                     'password': 'pw12345678'})
    return client, uid


def _approve(app, uid, admin_id):
    with app.app_context():
        ctx = load_trial_balance(uid)
        record = tb_approval.request_approval(uid, ctx, requested_by=uid)
        tb_approval.decide(record, admin_id=admin_id, approve=True)


def test_every_transmission_path_refuses_an_unapproved_trial_balance(canary_app):
    client, uid = _client(canary_app, n=1)
    r = client.get('/api/trial-balance')
    assert r.status_code == 403 and r.get_json()['approval_required']
    r = client.get('/api/trial-balance/share')
    assert r.status_code == 403
    token = create_share_token(uid, secret_key=canary_app.config['SECRET_KEY'])
    assert client.get(f'/api/trial-balance/shared/{token}').status_code == 403
    # The Excel download is a manual export and is not gated.
    assert client.get('/trial-balance/export').status_code == 200


def test_the_page_offers_the_request_and_the_admin_approves_it(canary_app):
    client, uid = _client(canary_app, n=2)
    page = client.get('/trial-balance')
    assert b'Request approval to send' in page.data

    r = client.post('/trial-balance/request-approval', follow_redirects=True)
    assert b'Approval requested' in r.data
    with canary_app.app_context():
        record = TrialBalanceApproval.query.filter_by(user_id=uid).one()
        assert record.status == 'requested'
        assert record.total_debits == 100.0 == record.total_credits

    admin, admin_id = _client(canary_app, admin=True, n=2)
    listing = admin.get('/admin/tb-approvals')
    assert b'TEST 2' in listing.data
    r = admin.post(f'/admin/tb-approvals/{record.id}/decide',
                   data={'decision': 'approve', 'note': 'Looks right'}, follow_redirects=True)
    assert b'approved' in r.data

    assert client.get('/api/trial-balance').status_code == 200
    assert client.get('/api/trial-balance/share').status_code == 200
    token = create_share_token(uid, secret_key=canary_app.config['SECRET_KEY'])
    assert client.get(f'/api/trial-balance/shared/{token}').status_code == 200
    assert b'Approved for sending' in client.get('/trial-balance').data


def test_a_non_admin_cannot_approve(canary_app):
    client, uid = _client(canary_app, n=3)
    client.post('/trial-balance/request-approval')
    with canary_app.app_context():
        record_id = TrialBalanceApproval.query.filter_by(user_id=uid).one().id
    r = client.post(f'/admin/tb-approvals/{record_id}/decide', data={'decision': 'approve'})
    assert r.status_code in (302, 403)
    with canary_app.app_context():
        assert TrialBalanceApproval.query.get(record_id).status == 'requested'
    assert client.get('/api/trial-balance').status_code == 403


def test_an_approval_dies_when_the_figures_change(canary_app):
    client, uid = _client(canary_app, n=4)
    _, admin_id = _client(canary_app, admin=True, n=4)
    _approve(canary_app, uid, admin_id)
    assert client.get('/api/trial-balance').status_code == 200

    with canary_app.app_context():
        sales = Account.query.filter_by(user_id=uid, link='i.100.000').one()
        f = UploadedFile.query.filter_by(user_id=uid).one()
        db.session.add(Transaction(date=datetime.now(), description='Another sale',
                                   amount=50.0, user_id=uid, account_id=sales.id, file_id=f.id))
        db.session.commit()

    r = client.get('/api/trial-balance')
    assert r.status_code == 403
    assert 'changed since it was approved' in r.get_json()['error']
    assert b'Changed since it was approved' in client.get('/trial-balance').data


def test_a_declined_request_stays_refused_and_says_why(canary_app):
    client, uid = _client(canary_app, n=5)
    _, admin_id = _client(canary_app, admin=True, n=5)
    with canary_app.app_context():
        ctx = load_trial_balance(uid)
        record = tb_approval.request_approval(uid, ctx, requested_by=uid)
        tb_approval.decide(record, admin_id=admin_id, approve=False, note='Rent is missing')
    assert client.get('/api/trial-balance').status_code == 403
    page = client.get('/trial-balance').data
    assert b'Declined' in page and b'Rent is missing' in page


def test_the_gate_can_be_switched_off_by_env(canary_app, monkeypatch):
    client, uid = _client(canary_app, n=6)
    monkeypatch.setenv('ANALEE_TB_APPROVAL_REQUIRED', '0')
    assert client.get('/api/trial-balance').status_code == 200
