"""
Integration tests for the OCR confirm route's per-row include filter (Phase 2.1).

Only rows the user keeps ticked (their index present in 'include') should be
imported when 'has_include_filter' is set; without that hidden field, all rows
import (backward compatibility).
"""
import pytest

pytest.importorskip("flask_sqlalchemy")
pytest.importorskip("flask_login")

from flask import Flask
from flask_login import LoginManager
from models import db, User, Transaction
from ocr import ocr as ocr_bp


def _make_app():
    app = Flask(__name__, template_folder="templates")
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY="test",
        WTF_CSRF_ENABLED=False,
        LOGIN_DISABLED=True,
    )
    db.init_app(app)
    lm = LoginManager()
    lm.init_app(app)

    @lm.user_loader
    def _load(uid):
        return db.session.get(User, int(uid))

    app.register_blueprint(ocr_bp)
    # Stub the redirect target so url_for('main.upload') can build.
    app.add_url_rule('/upload', endpoint='main.upload', view_func=lambda: 'ok')
    return app


def _login(client, user_id):
    with client.session_transaction() as sess:
        sess['_user_id'] = str(user_id)


def _seed_user(app):
    with app.app_context():
        db.create_all()
        user = User(username='t', email='t@e.com', password_hash='x')
        db.session.add(user)
        db.session.commit()
        return user.id


def test_confirm_imports_only_included_rows():
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)

    # Two rows submitted, but only row index 1 ticked -> only that one imported.
    resp = client.post('/ocr/receipt/confirm', data={
        'date': ['2026-03-01', '2026-03-02'],
        'description': ['Dup Coffee', 'New Lunch'],
        'amount': ['4.50', '12.30'],
        'include': ['1'],
        'has_include_filter': '1',
        'filename': 'statement.pdf',
    })
    assert resp.status_code in (301, 302)
    with app.app_context():
        rows = Transaction.query.all()
        assert len(rows) == 1
        assert rows[0].description == 'New Lunch'


def test_confirm_without_filter_imports_all_rows():
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)

    # No has_include_filter -> legacy behavior: import everything.
    resp = client.post('/ocr/receipt/confirm', data={
        'date': ['2026-03-01', '2026-03-02'],
        'description': ['A', 'B'],
        'amount': ['1.00', '2.00'],
        'filename': 'receipt.jpg',
    })
    assert resp.status_code in (301, 302)
    with app.app_context():
        assert Transaction.query.count() == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
