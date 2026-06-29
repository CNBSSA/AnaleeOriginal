"""Integration: the no-login client-explain POST must work with CSRF enabled.

The client wizard form carries no CSRF token (clients have no session), so without
a blueprint-level CSRF exemption the save POST would be rejected with
"The CSRF token is missing". This boots the real app with WTF_CSRF_ENABLED on and
verifies a client can save an explanation through the token-scoped endpoint.
"""
import os
import sys
import types
import tempfile

# Stub optional extensions that app.py imports at module load.
for _name in ("flask_migrate", "flask_apscheduler"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["flask_migrate"].Migrate = lambda *a, **k: types.SimpleNamespace(
    init_app=lambda *a, **k: None)


class _Sched:
    def __init__(self, *a, **k):
        pass

    def init_app(self, *a, **k):
        pass


sys.modules["flask_apscheduler"].APScheduler = _Sched

from datetime import datetime  # noqa: E402


def _boot_app():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["DATABASE_URL"] = f"sqlite:///{path}"
    os.environ["FLASK_SECRET_KEY"] = "client-explain-csrf-test-secret"
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    # NOTE: deliberately leave WTF_CSRF_ENABLED at its real value (True) so this
    # test actually exercises the CSRF path the client hits in production.
    assert app.config["WTF_CSRF_ENABLED"] is True
    return app


def test_client_can_save_explanation_without_csrf_token():
    app = _boot_app()
    from models import db, User, UploadedFile, Transaction
    from client_explain_tokens import create_client_explain_token

    with app.app_context():
        user = User(username="owner", email="owner@example.com",
                    subscription_status="active")
        user.set_password("secret")
        db.session.add(user)
        db.session.commit()
        uid = user.id
        uploaded = UploadedFile(filename="march.csv", user_id=uid,
                                upload_date=datetime.utcnow())
        db.session.add(uploaded)
        db.session.flush()
        fid = uploaded.id
        txn = Transaction(date=datetime(2026, 3, 1), description="Builder Supply",
                          amount=-500.0, user_id=uid, file_id=fid,
                          explanation="", explanation_source="")
        db.session.add(txn)
        db.session.commit()
        tid = txn.id
        token = create_client_explain_token(fid, uid, secret_key=app.config["SECRET_KEY"])

    client = app.test_client()
    # GET renders the wizard (no login).
    assert client.get(f"/client-explain/{token}/").status_code == 200

    # POST with NO csrf token must succeed (200), not be rejected (400).
    resp = client.post(
        f"/client-explain/{token}/",
        data={"transaction_id": tid, "explanation": "Bought timber for a job"},
    )
    assert resp.status_code == 200, (
        f"client save POST blocked (status {resp.status_code}) — CSRF exemption missing?")

    with app.app_context():
        from models import Transaction as T
        saved = T.query.get(tid)
        assert saved.explanation == "Bought timber for a job"
        assert saved.explanation_source == "client"
