"""QA register #14 — the guard must never report a rotation it did not perform.

``neutralise_default_admin`` computed
``new_password = replacement_password or secrets.token_urlsafe(32)`` and never
checked that the replacement DIFFERED from the credential it existed to retire. So
``ADMIN_PASSWORD=admin123`` made it detect the default was live, write the same
value back, commit, log "has been rotated" and return ``STATUS_ROTATED``.

That is worse than doing nothing: a false success ends the investigation. It was
raised in PR #122's security review and merged regardless, and the credential is
``festusa@cnbs.co.za`` — Festus's own address.
"""
from __future__ import annotations

import pytest

from models import User, db
from security.default_admin_guard import (
    LEGACY_ADMIN_EMAIL, LEGACY_ADMIN_PASSWORD, STATUS_NOT_DEFAULT, STATUS_REFUSED,
    STATUS_ROTATED, neutralise_default_admin)


@pytest.fixture
def legacy_admin(app):
    """The row as it exists in a database that predates the guard."""
    with app.app_context():
        user = User(username='festusa', email=LEGACY_ADMIN_EMAIL,
                    subscription_status='active')
        user.set_password(LEGACY_ADMIN_PASSWORD)
        db.session.add(user)
        db.session.commit()
        yield user.id


class TestAKnownDefaultIsRefusedAsAReplacement:
    def test_rotating_to_the_same_default_is_refused(self, app, legacy_admin):
        with app.app_context():
            result = neutralise_default_admin(
                db.session, replacement_password=LEGACY_ADMIN_PASSWORD)
        assert result['status'] == STATUS_REFUSED
        assert result['used_env_password'] is False

    def test_the_refusal_does_not_claim_a_rotation(self, app, legacy_admin):
        """The specific lie: STATUS_ROTATED over a live credential."""
        with app.app_context():
            result = neutralise_default_admin(
                db.session, replacement_password=LEGACY_ADMIN_PASSWORD)
        assert result['status'] != STATUS_ROTATED

    def test_the_credential_is_left_exactly_as_it_was(self, app, legacy_admin):
        """Refusing must not half-apply something. The operator's next step depends
        on the state being unchanged and knowable."""
        with app.app_context():
            neutralise_default_admin(db.session,
                                     replacement_password=LEGACY_ADMIN_PASSWORD)
            user = User.query.filter_by(email=LEGACY_ADMIN_EMAIL).first()
            assert user.check_password(LEGACY_ADMIN_PASSWORD) is True

    def test_the_log_says_the_credential_is_still_live(self, app, legacy_admin, caplog):
        """Festus reads Railway logs, not a shell. The log must not be reassuring."""
        with app.app_context(), caplog.at_level('ERROR'):
            neutralise_default_admin(db.session,
                                     replacement_password=LEGACY_ADMIN_PASSWORD)
        assert 'STILL LIVE' in caplog.text
        assert 'REFUSING' in caplog.text

    @pytest.mark.parametrize('weak', ['admin', 'password', 'changeme', 'ADMIN123',
                                      ' admin123 ', '123456'])
    def test_other_obvious_near_misses_are_refused_too(self, app, legacy_admin, weak):
        """Case and surrounding whitespace must not get a known value past the check."""
        with app.app_context():
            result = neutralise_default_admin(db.session, replacement_password=weak)
        assert result['status'] == STATUS_REFUSED, f'{weak!r} was accepted'

    def test_a_forced_reset_to_a_known_default_is_also_refused(self, app, legacy_admin):
        """force=True is the operator's recovery path, not a way around the check —
        otherwise the documented recovery step reintroduces the defect."""
        with app.app_context():
            result = neutralise_default_admin(
                db.session, replacement_password=LEGACY_ADMIN_PASSWORD, force=True)
            assert result['status'] == STATUS_REFUSED
            user = User.query.filter_by(email=LEGACY_ADMIN_EMAIL).first()
            assert user.check_password(LEGACY_ADMIN_PASSWORD) is True


class TestLegitimateRotationStillWorks:
    def test_a_real_replacement_rotates_and_retires_the_default(self, app, legacy_admin):
        """The twin. The check must refuse the known value without refusing the job."""
        with app.app_context():
            result = neutralise_default_admin(
                db.session, replacement_password='Chosen-By-Festus-1')
            assert result['status'] == STATUS_ROTATED
            assert result['used_env_password'] is True
            user = User.query.filter_by(email=LEGACY_ADMIN_EMAIL).first()
            assert user.check_password('Chosen-By-Festus-1') is True
            assert user.check_password(LEGACY_ADMIN_PASSWORD) is False

    def test_no_replacement_still_rotates_to_a_random_value(self, app, legacy_admin):
        """Unchanged behaviour: with nothing supplied the default is retired anyway."""
        with app.app_context():
            result = neutralise_default_admin(db.session)
            assert result['status'] == STATUS_ROTATED
            assert result['used_env_password'] is False
            user = User.query.filter_by(email=LEGACY_ADMIN_EMAIL).first()
            assert user.check_password(LEGACY_ADMIN_PASSWORD) is False

    def test_a_second_run_reports_not_default(self, app, legacy_admin):
        with app.app_context():
            assert neutralise_default_admin(db.session)['status'] == STATUS_ROTATED
            assert neutralise_default_admin(db.session)['status'] == STATUS_NOT_DEFAULT

    def test_a_weak_but_unknown_password_is_allowed(self, app, legacy_admin):
        """Scope: this guard makes rotation HONEST; it is not a password-strength
        policy. A weak unknown value rotates — and is reported accurately."""
        with app.app_context():
            result = neutralise_default_admin(db.session,
                                              replacement_password='hunter2')
            assert result['status'] == STATUS_ROTATED
