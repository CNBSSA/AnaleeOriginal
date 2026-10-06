"""Login abuse protection — a database-backed failed-attempt ledger (work-orders #69, R1).

There was no throttle, back-off or lockout on password login: thirty wrong
passwords for an administrator's address were answered as fast as the server
could, each with the same "Invalid email or password". With two-factor
authentication ruled out corporation-wide, this is the compensating control that
has to exist.

Why a DATABASE ledger and not an in-process counter: the deploy runs several
gunicorn workers, so a per-process counter would count per worker and an
advertised ten attempts would really be ten times the worker count.
"""
import logging

import pytest

pytest.importorskip("flask_sqlalchemy")

EMAIL = "throttle@example.com"
PASSWORD = "Sup3rSecret!"


def _register(client, email=EMAIL, username="throttled"):
    return client.post("/auth/register", data={
        "username": username, "email": email,
        "password": PASSWORD, "confirm_password": PASSWORD,
    }, follow_redirects=True)


def _login(client, email=EMAIL, password="wrong-password", ip=None):
    kwargs = {}
    if ip:
        kwargs["environ_base"] = {"REMOTE_ADDR": ip}
    return client.post("/auth/login", data={"email": email, "password": password},
                       **kwargs)


def test_an_email_is_locked_after_ten_failures_and_recovers_after_the_window(canary_app):
    from security import login_throttle

    client = canary_app.test_client()
    _register(client)

    for _ in range(login_throttle.EMAIL_LIMIT):
        resp = _login(client)
        assert resp.status_code == 200, "a failure inside the limit is answered normally"
        assert b"Invalid email or password" in resp.data

    # The eleventh attempt is refused even with the RIGHT password.
    resp = _login(client, password=PASSWORD)
    assert resp.status_code == 429
    assert login_throttle.LOCKED_MESSAGE.encode() in resp.data
    assert resp.headers.get("Retry-After")
    assert "/dashboard" not in resp.headers.get("Location", "")

    # Attempts older than the window stop counting: the user gets back in.
    from models import db, LoginAttempt
    from datetime import datetime, timedelta
    with canary_app.app_context():
        old = datetime.utcnow() - login_throttle.WINDOW - timedelta(seconds=1)
        for row in LoginAttempt.query.all():
            row.attempted_at = old
        db.session.commit()
    resp = _login(client, password=PASSWORD)
    assert resp.status_code == 302 and resp.headers["Location"].endswith("/dashboard")


def test_a_success_clears_that_emails_failures(canary_app):
    from security import login_throttle

    client = canary_app.test_client()
    _register(client)
    for _ in range(login_throttle.EMAIL_LIMIT - 1):
        _login(client)
    resp = _login(client, password=PASSWORD)
    assert resp.status_code == 302
    client.get("/auth/logout")

    # The slate is clean: the same number of failures is allowed again.
    for _ in range(login_throttle.EMAIL_LIMIT - 1):
        assert _login(client).status_code == 200
    assert _login(client, password=PASSWORD).status_code == 302


def test_one_ip_is_locked_after_thirty_failures_across_many_emails(canary_app):
    from security import login_throttle

    client = canary_app.test_client()
    for i in range(login_throttle.IP_LIMIT):
        resp = _login(client, email=f"guess{i}@example.com", ip="203.0.113.7")
        assert resp.status_code == 200

    resp = _login(client, email="another@example.com", ip="203.0.113.7")
    assert resp.status_code == 429

    # A different client address is not affected by that one's failures.
    resp = _login(client, email="another@example.com", ip="203.0.113.8")
    assert resp.status_code == 200


def test_the_ledger_stores_a_hash_never_the_address(canary_app):
    from models import LoginAttempt

    client = canary_app.test_client()
    _login(client, email="who.am.i@example.com", ip="203.0.113.9")
    with canary_app.app_context():
        rows = LoginAttempt.query.all()
        assert rows, "a failed attempt is recorded"
        for row in rows:
            assert "who.am.i" not in (row.key_hash or "")
            assert "203.0.113.9" not in (row.key_hash or "")
            assert len(row.key_hash) == 64  # sha256 hex


def test_the_throttle_fails_open_when_the_ledger_is_unavailable(canary_app, monkeypatch, caplog):
    """A broken ledger must never lock everyone out: login proceeds as before."""
    from security import login_throttle

    def _boom(*a, **k):
        raise RuntimeError("ledger table unavailable")

    monkeypatch.setattr(login_throttle, "_count_failures", _boom)
    monkeypatch.setattr(login_throttle, "_insert", _boom)

    client = canary_app.test_client()
    _register(client)
    with caplog.at_level(logging.ERROR):
        resp = _login(client, password=PASSWORD)
    assert resp.status_code == 302 and resp.headers["Location"].endswith("/dashboard")
    assert any("login throttle" in r.getMessage() for r in caplog.records)


def test_the_client_ip_is_the_last_forwarded_for_entry(canary_app):
    """Railway terminates TLS in front of gunicorn and this app has no ProxyFix,
    so remote_addr is the proxy. The per-IP limit must follow the real client
    (the LAST X-Forwarded-For entry, which the edge appended), or one busy
    proxy address would lock every customer out together."""
    from security.login_throttle import client_ip

    with canary_app.test_request_context(
            headers={"X-Forwarded-For": "198.51.100.1, 203.0.113.50"},
            environ_base={"REMOTE_ADDR": "10.0.0.2"}):
        assert client_ip() == "203.0.113.50"
    with canary_app.test_request_context(environ_base={"REMOTE_ADDR": "10.0.0.2"}):
        assert client_ip() == "10.0.0.2"


def test_the_migration_exists_for_environments_that_run_upgrade():
    import glob
    import os
    files = glob.glob(os.path.join("migrations", "versions", "*login_attempt*.py"))
    assert files, "the additive table needs a migration beside the boot create_all()"
