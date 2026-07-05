"""The statement-upload page must SAY when AI reading is unconfigured.

Without ANTHROPIC_API_KEY, scanned-PDF uploads bounce back to the same page
("the upload doesn't go anywhere"). The page now shows a warning banner naming
the missing variable so the failure is self-explanatory at the point of use.
"""
import os

import pytest

pytest.importorskip("flask_sqlalchemy")
pytest.importorskip("flask_login")

from flask import Flask

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from flask_login import LoginManager
from flask_wtf import CSRFProtect

from models import db, User
from ocr import ocr as ocr_bp


def _make_app():
    app = Flask(__name__, template_folder=os.path.join(_REPO_ROOT, "templates"))
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY="test",
        WTF_CSRF_ENABLED=False,
        LOGIN_DISABLED=True,
    )
    db.init_app(app)
    CSRFProtect(app)  # provides the csrf_token() template global
    lm = LoginManager()
    lm.init_app(app)

    @lm.user_loader
    def _load(uid):
        return db.session.get(User, int(uid))

    app.register_blueprint(ocr_bp)
    with app.app_context():
        db.create_all()
    return app


BANNER = "AI reading is not configured"


def test_banner_shown_when_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = _make_app().test_client().get("/ocr/statement")
    assert resp.status_code == 200
    assert BANNER in resp.get_data(as_text=True)


def test_banner_hidden_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    resp = _make_app().test_client().get("/ocr/statement")
    assert resp.status_code == 200
    assert BANNER not in resp.get_data(as_text=True)
