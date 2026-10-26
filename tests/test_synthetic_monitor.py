"""Synthetic monitor must fail when unconfigured and probe /health + auth paths."""
import importlib.util
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("requests")
from werkzeug.serving import make_server

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_monitor():
    spec = importlib.util.spec_from_file_location(
        "synthetic_monitor", _SCRIPTS / "synthetic_monitor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synthetic_monitor():
    return _load_monitor()


def test_unconfigured_monitor_fails_loudly(synthetic_monitor, monkeypatch, capsys):
    for name in ("MONITOR_BASE_URL", "MONITOR_EMAIL", "MONITOR_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    code = synthetic_monitor.check()
    captured = capsys.readouterr()
    assert code == 1
    assert "NOT CONFIGURED" in captured.err
    assert "zero probes" in captured.err.lower()


def test_failure_posts_to_alert_webhook(synthetic_monitor, monkeypatch):
    monkeypatch.setenv("MONITOR_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("MONITOR_EMAIL", "monitor@test.example")
    monkeypatch.setenv("MONITOR_PASSWORD", "wrong")
    monkeypatch.setenv("MONITOR_ALERT_WEBHOOK", "https://hooks.example.test/alert")
    monkeypatch.setenv("MONITOR_TIMEOUT", "1")

    posts = []

    def _fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return MagicMock(status_code=200)

    monkeypatch.setattr(synthetic_monitor.requests, "post", _fake_post)

    assert synthetic_monitor.check() == 1
    assert len(posts) == 1
    assert posts[0][0] == "https://hooks.example.test/alert"
    assert "FAILED" in posts[0][1]["json"]["text"]


@pytest.fixture
def live_app_for_monitor(monkeypatch):
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="syn_mon_")
    os.close(fd)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("FLASK_SECRET_KEY", "synthetic-monitor-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from app import create_app
    from models import User, db

    app = create_app()
    assert app is not None
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["TESTING"] = True

    email = "synthetic-monitor@test.example"
    password = "Synthetic-Monitor-Test-9"
    with app.app_context():
        user = User(
            username="synmon",
            email=email,
            subscription_status="active",
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setenv("MONITOR_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("MONITOR_EMAIL", email)
    monkeypatch.setenv("MONITOR_PASSWORD", password)
    monkeypatch.setenv("MONITOR_MAX_SECONDS", "30")
    monkeypatch.delenv("MONITOR_ALERT_WEBHOOK", raising=False)

    yield

    server.shutdown()


def test_configured_monitor_passes_when_app_is_up(
    synthetic_monitor, live_app_for_monitor, capsys
):
    code = synthetic_monitor.check()
    captured = capsys.readouterr()
    assert code == 0
    assert "OK   /health" in captured.out
    assert "OK   /dashboard" in captured.out
    assert "all checks passed" in captured.out
