"""create_admin.py goes through the default-admin guard's refusals (work-orders #69, R10).

The manual script reset the password of WHATEVER account held ADMIN_EMAIL,
promoted it to administrator and activated it — with none of the refusals the
pre-deploy guard enforces (a known-default password; an ordinary account is
never promoted). It now calls ensure_admin_account(force=True): a missing
administrator is created, an existing administrator's password is reset to
ADMIN_PASSWORD (what the script did before), and the two refusals hold. This
NARROWS a password-rotation path; it never widens it. No password is printed.
"""
import os
import tempfile

import pytest

pytest.importorskip("flask_sqlalchemy")

ADMIN = "owner@analee.test"
STRONG = "Correct-Horse-Battery-9"


@pytest.fixture
def script_env(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db", prefix="create_admin_")
    os.close(fd)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.setenv("FLASK_SECRET_KEY", "create-admin-test")
    monkeypatch.setenv("ADMIN_EMAIL", ADMIN)
    monkeypatch.setenv("ADMIN_PASSWORD", STRONG)
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    yield
    try:
        os.remove(path)
    except OSError:
        pass


def _row(email=ADMIN):
    from app import create_app
    from models import User
    app = create_app()
    with app.app_context():
        return User.query.filter_by(email=email).first()


def _seed(email, password, is_admin):
    from app import create_app
    from models import db, User
    app = create_app()
    with app.app_context():
        u = User(username="seeded", email=email, is_admin=is_admin,
                 subscription_status="active")
        u.set_password(password)
        db.session.add(u)
        db.session.commit()


def test_a_missing_administrator_is_created(script_env, capsys):
    from create_admin import create_admin_user
    assert create_admin_user() == 0
    row = _row()
    assert row is not None and row.is_admin is True
    assert row.check_password(STRONG)
    assert STRONG not in capsys.readouterr().out


def test_an_existing_administrator_is_reset_to_admin_password(script_env):
    _seed(ADMIN, "Old-Admin-Secret-1", is_admin=True)
    from create_admin import create_admin_user
    assert create_admin_user() == 0
    assert _row().check_password(STRONG)


def test_an_ordinary_account_is_never_promoted_or_reset(script_env):
    _seed(ADMIN, "Customer-Secret-1", is_admin=False)
    from create_admin import create_admin_user
    assert create_admin_user() != 0
    row = _row()
    assert row.is_admin is False
    assert row.check_password("Customer-Secret-1")


def test_a_known_default_password_is_refused(script_env, monkeypatch, capsys):
    monkeypatch.setenv("ADMIN_PASSWORD", "admin123")
    from create_admin import create_admin_user
    assert create_admin_user() != 0
    assert _row() is None
    assert "admin123" not in capsys.readouterr().out


def test_the_script_uses_the_guard(script_env):
    import inspect
    import create_admin
    assert "ensure_admin_account(" in inspect.getsource(create_admin.create_admin_user)
    assert "set_password(" not in inspect.getsource(create_admin.create_admin_user)
