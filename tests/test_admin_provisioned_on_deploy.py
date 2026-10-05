"""A missing Analee administrator is provisioned on deploy (work-orders #66).

The trial-balance send gate (reports/tb_approval.py) waits for an administrator,
and production had NO administrator row at all: create_admin.py was never run
(no shell), and the pre-deploy guard only ever resets an existing legacy row —
so ADMIN_PASSWORD + ADMIN_PASSWORD_FORCE_RESET reported "no row … nothing to
do" and the gate could never be passed.

The pre-deploy step now creates the administrator named by ADMIN_EMAIL from
ADMIN_PASSWORD when that row does not exist. It never promotes an existing
ordinary account, never resets an existing administrator without the one-shot
force flag, refuses a known-default password, and never prints a password.
"""
from datetime import datetime

import pytest

from models import db, User
from security.default_admin_guard import (
    ADMIN_CREATED, ADMIN_EXISTS, ADMIN_FORCED, ADMIN_NOT_ADMIN, ADMIN_REFUSED,
    ADMIN_SKIPPED, ensure_admin_account)

EMAIL = 'test-admin@example.com'
PASSWORD = 'TEST-Admin-Pass-2026!'


def test_a_missing_admin_is_created_and_can_sign_in(app):
    result = ensure_admin_account(db.session, email=EMAIL, password=PASSWORD)
    assert result['status'] == ADMIN_CREATED
    row = User.query.filter_by(email=EMAIL).one()
    assert row.is_admin is True
    assert row.subscription_status == 'active'
    assert row.check_password(PASSWORD)


def test_email_is_matched_case_insensitively(app):
    ensure_admin_account(db.session, email='  Test-Admin@Example.com ', password=PASSWORD)
    assert User.query.filter_by(email=EMAIL).count() == 1
    assert ensure_admin_account(db.session, email=EMAIL,
                                password=PASSWORD)['status'] == ADMIN_EXISTS


def test_nothing_happens_without_both_variables(app):
    assert ensure_admin_account(db.session, email='', password=PASSWORD)['status'] == ADMIN_SKIPPED
    assert ensure_admin_account(db.session, email=EMAIL, password='')['status'] == ADMIN_SKIPPED
    assert User.query.count() == 0


def test_an_existing_admin_is_left_alone_without_the_force_flag(app):
    u = User(username='a', email=EMAIL, is_admin=True,
             created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    u.set_password('Their-Own-Password-1')
    db.session.add(u)
    db.session.commit()
    assert ensure_admin_account(db.session, email=EMAIL,
                                password=PASSWORD)['status'] == ADMIN_EXISTS
    assert User.query.get(u.id).check_password('Their-Own-Password-1')
    assert ensure_admin_account(db.session, email=EMAIL, password=PASSWORD,
                                force=True)['status'] == ADMIN_FORCED
    assert User.query.get(u.id).check_password(PASSWORD)


def test_an_ordinary_account_is_never_promoted_or_reset(app):
    u = User(username='c', email=EMAIL, is_admin=False,
             created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    u.set_password('Customer-Password-1')
    db.session.add(u)
    db.session.commit()
    for force in (False, True):
        assert ensure_admin_account(db.session, email=EMAIL, password=PASSWORD,
                                    force=force)['status'] == ADMIN_NOT_ADMIN
    row = User.query.get(u.id)
    assert row.is_admin is False
    assert row.check_password('Customer-Password-1')


def test_a_known_default_password_is_refused(app):
    assert ensure_admin_account(db.session, email=EMAIL,
                                password='admin123')['status'] == ADMIN_REFUSED
    assert User.query.count() == 0


def test_the_pre_deploy_step_creates_the_admin(app, monkeypatch, capsys):
    import app as app_module
    import neutralise_default_admin
    monkeypatch.setattr(app_module, 'create_app', lambda: app)
    monkeypatch.setenv('ADMIN_EMAIL', EMAIL)
    monkeypatch.setenv('ADMIN_PASSWORD', PASSWORD)
    monkeypatch.delenv('ADMIN_PASSWORD_FORCE_RESET', raising=False)
    assert neutralise_default_admin.main() == 0
    out = capsys.readouterr()
    assert f'admin provisioning: created administrator {EMAIL}' in out.out
    assert PASSWORD not in out.out + out.err
    assert User.query.filter_by(email=EMAIL, is_admin=True).count() == 1
