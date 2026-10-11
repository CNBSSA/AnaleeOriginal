"""Pre-deploy runner must fail loudly when the legacy default admin stays live."""
from __future__ import annotations

import pytest

from models import User, db
from security.default_admin_guard import LEGACY_ADMIN_EMAIL, LEGACY_ADMIN_PASSWORD


@pytest.fixture
def legacy_admin_on_default(app):
    with app.app_context():
        user = User(username='festusa', email=LEGACY_ADMIN_EMAIL,
                    subscription_status='active')
        user.set_password(LEGACY_ADMIN_PASSWORD)
        db.session.add(user)
        db.session.commit()
        yield user.id


def test_unneutralizable_default_admin_exits_nonzero_with_clear_message(
        app, legacy_admin_on_default, monkeypatch, capsys):
    import app as app_module
    import neutralise_default_admin

    monkeypatch.setattr(app_module, 'create_app', lambda: app)
    monkeypatch.setenv('ADMIN_PASSWORD', LEGACY_ADMIN_PASSWORD)
    monkeypatch.delenv('ADMIN_PASSWORD_FORCE_RESET', raising=False)

    assert neutralise_default_admin.main() != 0
    captured = capsys.readouterr()
    assert 'REFUSED' in captured.err or 'STILL' in captured.err
    with app.app_context():
        user = User.query.filter_by(email=LEGACY_ADMIN_EMAIL).first()
        assert user.check_password(LEGACY_ADMIN_PASSWORD) is True


def test_successful_neutralisation_exits_zero(app, legacy_admin_on_default, monkeypatch, capsys):
    import app as app_module
    import neutralise_default_admin

    monkeypatch.setattr(app_module, 'create_app', lambda: app)
    monkeypatch.setenv('ADMIN_PASSWORD', 'Chosen-By-Test-2026!')
    monkeypatch.delenv('ADMIN_PASSWORD_FORCE_RESET', raising=False)

    assert neutralise_default_admin.main() == 0
    captured = capsys.readouterr()
    assert 'WAS on the default password' in captured.out
    with app.app_context():
        user = User.query.filter_by(email=LEGACY_ADMIN_EMAIL).first()
        assert user.check_password(LEGACY_ADMIN_PASSWORD) is False


def test_no_legacy_row_exits_zero(app, monkeypatch, capsys):
    import app as app_module
    import neutralise_default_admin

    monkeypatch.setattr(app_module, 'create_app', lambda: app)
    monkeypatch.delenv('ADMIN_PASSWORD', raising=False)
    monkeypatch.delenv('ADMIN_PASSWORD_FORCE_RESET', raising=False)
    monkeypatch.delenv('ADMIN_EMAIL', raising=False)

    assert neutralise_default_admin.main() == 0
    captured = capsys.readouterr()
    assert 'nothing to do' in captured.out
