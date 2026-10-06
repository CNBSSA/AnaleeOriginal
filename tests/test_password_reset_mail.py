"""The password-reset link is DELIVERED when a mail server is configured
(work-orders #69, R2).

"Forgot password" created a valid token and then only logged it (and only when
ANALEE_PASSWORD_RESET_LOG_LINK=1), while telling the customer "If nothing
arrives, contact support". Flask-Mail is pinned, MAIL_* names existed in
config.py, and templates/email/password_reset.html existed — none of it was
wired. Now: with MAIL_SERVER set the template is sent to the user; with it
unset, behaviour is exactly what it was.
"""
import logging
import os
import tempfile

import pytest

pytest.importorskip("flask_sqlalchemy")
pytest.importorskip("flask_mail")

EMAIL = "locked.out@example.com"
PASSWORD = "Sup3rSecret!"


def _boot(monkeypatch, mail_server):
    fd, path = tempfile.mkstemp(suffix=".db", prefix="mail_")
    os.close(fd)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.setenv("FLASK_SECRET_KEY", "mail-test-secret")
    monkeypatch.delenv("ANALEE_PASSWORD_RESET_LOG_LINK", raising=False)
    for name in ("MAIL_SERVER", "MAIL_PORT", "MAIL_USE_TLS", "MAIL_USE_SSL",
                 "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_DEFAULT_SENDER"):
        monkeypatch.delenv(name, raising=False)
    if mail_server:
        monkeypatch.setenv("MAIL_SERVER", "smtp.example.test")
        monkeypatch.setenv("MAIL_PORT", "2525")
        monkeypatch.setenv("MAIL_USERNAME", "no-reply@example.test")
        monkeypatch.setenv("MAIL_PASSWORD", "not-a-real-secret")
        monkeypatch.setenv("MAIL_DEFAULT_SENDER", "Analee <no-reply@example.test>")
    from app import create_app
    app = create_app()
    assert app is not None
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["TESTING"] = True
    client = app.test_client()
    client.post("/auth/register", data={
        "username": "lockedout", "email": EMAIL,
        "password": PASSWORD, "confirm_password": PASSWORD,
    })
    return app, client


def test_without_a_mail_server_nothing_is_sent_and_nothing_changes(monkeypatch, caplog):
    app, client = _boot(monkeypatch, mail_server=False)
    assert app.config.get("MAIL_SERVER") is None
    assert "mail" not in app.extensions

    with caplog.at_level(logging.INFO):
        resp = client.post("/auth/reset_password_request", data={"email": EMAIL})
    assert resp.status_code == 302
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "/auth/reset_password/" not in joined, "the link is never logged by default"
    assert "link not logged" in joined


def test_with_a_mail_server_the_template_is_sent_to_the_user(monkeypatch, caplog):
    app, client = _boot(monkeypatch, mail_server=True)
    mail = app.extensions["mail"]
    mail.suppress = True  # record instead of connecting to smtp.example.test

    with mail.record_messages() as outbox, caplog.at_level(logging.INFO):
        resp = client.post("/auth/reset_password_request", data={"email": EMAIL})
    assert resp.status_code == 302
    assert len(outbox) == 1
    msg = outbox[0]
    assert msg.recipients == [EMAIL]
    assert msg.sender == "Analee <no-reply@example.test>"
    assert "/auth/reset_password/" in (msg.html or "")
    assert "lockedout" in (msg.html or ""), "the template greets the user by name"
    assert "expire in 1 hour" in (msg.html or "")

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "/auth/reset_password/" not in joined, "the link is never logged unless asked"
    assert EMAIL not in joined


def test_the_log_link_switch_still_works_exactly_as_before(monkeypatch, caplog):
    app, client = _boot(monkeypatch, mail_server=True)
    app.extensions["mail"].suppress = True
    monkeypatch.setenv("ANALEE_PASSWORD_RESET_LOG_LINK", "1")
    with caplog.at_level(logging.WARNING):
        client.post("/auth/reset_password_request", data={"email": EMAIL})
    assert any("PASSWORD RESET LINK" in r.getMessage() for r in caplog.records)


def test_a_mail_failure_is_logged_without_the_link_and_the_page_still_answers(monkeypatch, caplog):
    app, client = _boot(monkeypatch, mail_server=True)
    mail = app.extensions["mail"]

    def _boom(message):
        raise ConnectionRefusedError("smtp down")

    monkeypatch.setattr(mail, "send", _boom)
    with caplog.at_level(logging.ERROR):
        resp = client.post("/auth/reset_password_request", data={"email": EMAIL})
    assert resp.status_code == 302
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("could not send" in m.lower() for m in errors)
    assert not any("/auth/reset_password/" in m for m in errors)


def test_an_unknown_address_sends_nothing(monkeypatch):
    app, client = _boot(monkeypatch, mail_server=True)
    mail = app.extensions["mail"]
    mail.suppress = True
    with mail.record_messages() as outbox:
        client.post("/auth/reset_password_request", data={"email": "nobody@example.com"})
    assert outbox == []
