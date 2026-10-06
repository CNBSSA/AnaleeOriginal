"""The login page tells nobody who was a customer, and the logs carry no address
(work-orders #69, R4 + R6).

R4: a soft-deleted address used to be answered "This account has been deleted…"
while an unknown one got "Invalid email or password" — a free check of who had
been a customer. Both now get the same generic answer; the re-registration path
in auth.register still explains the situation to the person it concerns.

R6: every failed login logged the submitted e-mail address, and successful
login/logout/registration logged it too. Railway logs are read by several
people and kept. Log lines now carry the user id or a short hash.
"""
import logging

import pytest

pytest.importorskip("flask_sqlalchemy")

EMAIL = "former.customer@example.com"
PASSWORD = "Sup3rSecret!"


def _register(client, email=EMAIL, username="former"):
    return client.post("/auth/register", data={
        "username": username, "email": email,
        "password": PASSWORD, "confirm_password": PASSWORD,
    }, follow_redirects=True)


def test_a_deleted_account_gets_the_same_answer_as_an_unknown_one(canary_app):
    client = canary_app.test_client()
    _register(client)
    from models import db, User
    with canary_app.app_context():
        User.query.filter_by(email=EMAIL).first().soft_delete()
        db.session.commit()

    deleted = client.post("/auth/login", data={"email": EMAIL, "password": "nope"})
    unknown = client.post("/auth/login", data={"email": "nobody@example.com",
                                               "password": "nope"})
    assert deleted.status_code == unknown.status_code == 200
    assert b"deleted" not in deleted.data.lower()
    assert b"Invalid email or password" in deleted.data
    assert b"Invalid email or password" in unknown.data


def test_a_deleted_account_cannot_log_in_with_its_old_password(canary_app):
    client = canary_app.test_client()
    _register(client)
    from models import db, User
    with canary_app.app_context():
        User.query.filter_by(email=EMAIL).first().soft_delete()
        db.session.commit()
    resp = client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})
    assert resp.status_code == 200 and b"Invalid email or password" in resp.data


def test_login_log_lines_carry_no_email_address(canary_app, caplog):
    client = canary_app.test_client()
    _register(client)
    with caplog.at_level(logging.INFO):
        client.post("/auth/login", data={"email": "ghost@example.com", "password": "x"})
        client.post("/auth/login", data={"email": EMAIL, "password": "wrong"})
        client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})
        client.get("/auth/logout")
    auth_lines = [r.getMessage() for r in caplog.records if r.name.startswith("auth")]
    assert auth_lines, "the auth routes still log what happened"
    joined = "\n".join(auth_lines)
    assert "ghost@example.com" not in joined
    assert EMAIL not in joined
    assert "@" not in joined


def test_registration_log_lines_carry_no_email_address(canary_app, caplog):
    client = canary_app.test_client()
    with caplog.at_level(logging.INFO):
        _register(client)
    joined = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("auth"))
    assert EMAIL not in joined and "@" not in joined
