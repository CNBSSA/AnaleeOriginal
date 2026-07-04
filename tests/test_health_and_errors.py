"""Tests for GET /health diagnostics and the friendly 500 handler.

/health lets anyone verify, in one browser visit, which build is live and
whether the DB/schema/env are healthy — turning "Internal Server Error, no
idea why" into a named cause. The 500 handler replaces the bare Werkzeug page
with a friendly one and writes an [ANALEE-500]-tagged traceback to the logs.
"""
import logging
import os
import sys
import tempfile
import types

# app.py imports these optional extensions at module load; stub them so the
# module imports on any environment (same pattern as the other boot tests).
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


def _boot_app():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["DATABASE_URL"] = f"sqlite:///{path}"
    os.environ["FLASK_SECRET_KEY"] = "health-test-secret"
    from app import create_app
    app = create_app()
    assert app is not None, "create_app() returned None"
    app.config["TESTING"] = False  # 500 handler only runs outside TESTING
    app.config["PROPAGATE_EXCEPTIONS"] = False
    return app


def test_health_reports_ok_on_healthy_boot():
    app = _boot_app()
    resp = app.test_client().get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["db_ok"] is True
    assert data["schema_missing"] == {}
    assert data["env"]["FLASK_SECRET_KEY"] is True
    assert data["env"]["DATABASE_URL"] is True
    assert "commit" in data and "booted_at" in data
    # No secret VALUES anywhere in the payload.
    assert "health-test-secret" not in resp.get_data(as_text=True)


def test_500_handler_friendly_page_and_log_marker(caplog):
    app = _boot_app()

    @app.route("/boom")
    def _boom():
        raise RuntimeError("kaboom for testing")

    client = app.test_client()
    with caplog.at_level(logging.ERROR):
        resp = client.get("/boom")
    assert resp.status_code == 500
    body = resp.get_data(as_text=True)
    assert "Something went wrong on our side" in body
    assert "Internal Server Error" not in body  # bare Werkzeug page replaced
    assert any("[ANALEE-500]" in r.message for r in caplog.records)

    # XHR/JSON callers get JSON, not HTML.
    resp = client.get("/boom", headers={"X-Requested-With": "XMLHttpRequest"})
    assert resp.status_code == 500
    assert resp.get_json()["success"] is False
