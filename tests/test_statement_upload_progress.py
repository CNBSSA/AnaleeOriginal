"""The statement-upload page must show an in-progress message while extraction
runs. Claude Vision reads of scanned statements take 1-2+ minutes; without any
on-page feedback the upload looks dead ("nothing indicating that extraction is
taking place"). The page ships a progress panel + submit-time script that
reveals it, disables the button, and updates the wording as time passes.
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


def test_upload_page_ships_progress_feedback():
    body = _make_app().test_client().get("/ocr/statement").get_data(as_text=True)
    # The progress panel exists (hidden until submit) ...
    assert 'id="extract-progress"' in body
    assert "Reading your statement" in body
    # ... the submit button is wired for the working state ...
    assert 'id="extract-btn"' in body
    # ... and the script escalates the message for long scanned reads.
    assert "scanned statement" in body
    assert "pageshow" in body  # Back-button restore resets the working state
