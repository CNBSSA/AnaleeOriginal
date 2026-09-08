"""The pre-deploy guard rotates the legacy admin password ONLY if it is the
known default, and never touches any other row (QA register #1)."""
from datetime import datetime

from models import db, User
from security.default_admin_guard import (
    LEGACY_ADMIN_EMAIL, LEGACY_ADMIN_PASSWORD, STATUS_ABSENT, STATUS_FORCED,
    STATUS_NOT_DEFAULT, STATUS_ROTATED, neutralise_default_admin)


def _admin(password, email=LEGACY_ADMIN_EMAIL):
    u = User(username='Admin', email=email, is_admin=True,
             created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    return u


def test_no_row_is_a_no_op(app):
    assert neutralise_default_admin(db.session)['status'] == STATUS_ABSENT


def test_row_with_a_real_password_is_never_touched(app):
    u = _admin('Correct-Horse-Battery-9')
    before = u.password_hash
    assert neutralise_default_admin(db.session)['status'] == STATUS_NOT_DEFAULT
    assert User.query.get(u.id).password_hash == before
    assert User.query.get(u.id).check_password('Correct-Horse-Battery-9')


def test_default_password_is_rotated_to_a_random_value(app):
    u = _admin(LEGACY_ADMIN_PASSWORD)
    result = neutralise_default_admin(db.session)
    assert result['status'] == STATUS_ROTATED and result['used_env_password'] is False
    row = User.query.get(u.id)
    assert not row.check_password(LEGACY_ADMIN_PASSWORD)
    assert row.is_admin is True  # the account itself is kept, only the secret changes


def test_default_password_is_rotated_to_the_operator_choice_when_given(app):
    u = _admin(LEGACY_ADMIN_PASSWORD)
    result = neutralise_default_admin(db.session, replacement_password='Chosen-By-Festus-1')
    assert result['status'] == STATUS_ROTATED and result['used_env_password'] is True
    row = User.query.get(u.id)
    assert row.check_password('Chosen-By-Festus-1')
    assert not row.check_password(LEGACY_ADMIN_PASSWORD)


def test_second_run_is_a_no_op(app):
    _admin(LEGACY_ADMIN_PASSWORD)
    assert neutralise_default_admin(db.session)['status'] == STATUS_ROTATED
    assert neutralise_default_admin(db.session)['status'] == STATUS_NOT_DEFAULT


def test_other_users_on_the_default_password_are_not_in_scope(app):
    """The guard is for the one legacy row; it must not sweep customers."""
    other = _admin(LEGACY_ADMIN_PASSWORD, email='someone@example.com')
    assert neutralise_default_admin(db.session)['status'] == STATUS_ABSENT
    assert User.query.get(other.id).check_password(LEGACY_ADMIN_PASSWORD)


def test_forced_reset_applies_the_operator_password_regardless_of_state(app):
    """Recovery after a random rotation: ADMIN_PASSWORD + the one-shot flag."""
    u = _admin('Some-Random-Rotation-Value')
    result = neutralise_default_admin(db.session, replacement_password='Recovered-1',
                                      force=True)
    assert result['status'] == STATUS_FORCED
    assert User.query.get(u.id).check_password('Recovered-1')


def test_force_without_a_password_never_randomises_a_working_account(app):
    u = _admin('Correct-Horse-Battery-9')
    assert neutralise_default_admin(db.session, force=True)['status'] == STATUS_NOT_DEFAULT
    assert User.query.get(u.id).check_password('Correct-Horse-Battery-9')
