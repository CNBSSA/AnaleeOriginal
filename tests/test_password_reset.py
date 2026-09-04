"""Password recovery — the token decides whose password changes.

QA audit, 2026-09-04. The previous route accepted a ``<token>`` and never read
it: it looked up a user by an email posted in the form and reset THAT account.
It survived only because ``ResetPasswordForm`` has no email field, so every
submission raised and returned 500 — broken, not exploitable. The danger was
that anyone "fixing the 500" by adding the missing field would have shipped
unauthenticated account takeover of any user, including an admin.

``test_the_takeover_trap_stays_shut`` is the guard against exactly that, and is
the most important test in this file. Do not delete it to make a change pass.
"""
import re

import pytest
from itsdangerous import URLSafeTimedSerializer

from password_reset_tokens import (
    PASSWORD_RESET_SALT,
    create_password_reset_token,
    token_matches_current_password,
    verify_password_reset_token,
    BadSignature,
    SignatureExpired,
)

VICTIM = "victim@example.com"
ATTACKER = "attacker@example.com"
PASSWORD = "Sup3rSecret!"
NEW_PASSWORD = "Brand-New-Pass1"


def _register(client, username, email):
    return client.post("/auth/register", data={
        "username": username, "email": email,
        "password": PASSWORD, "confirm_password": PASSWORD,
    }, follow_redirects=True)


def _user(app, email):
    from models import User
    with app.app_context():
        return User.query.filter_by(email=email).first()


def _token_for(app, email):
    with app.app_context():
        from models import User
        user = User.query.filter_by(email=email).first()
        return create_password_reset_token(
            user, secret_key=app.config['SECRET_KEY'])


# ---- the trap ---------------------------------------------------------------

def test_the_takeover_trap_stays_shut():
    """The reset view must never resolve its target from posted form data.

    If this fails, read the module docstring before changing anything: adding an
    email/user field back to this route re-creates an account-takeover hole.
    """
    import inspect
    import auth.routes as auth_routes

    src = inspect.getsource(auth_routes.reset_password)
    for forbidden in ('form.email', 'filter_by(email', 'request.form.get("email"',
                      "request.form.get('email'"):
        assert forbidden not in src, (
            f"reset_password() reads {forbidden!r} — the target must come from "
            "the signed token, never from the request. See "
            "tests/test_password_reset.py."
        )


# ---- the token itself -------------------------------------------------------

def test_token_is_bound_to_one_user(canary_app):
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    _register(client, "attacker", ATTACKER)

    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        attacker = User.query.filter_by(email=ATTACKER).first()
        key = canary_app.config['SECRET_KEY']

        token = create_password_reset_token(attacker, secret_key=key)
        assert verify_password_reset_token(token, secret_key=key) == attacker.id
        assert verify_password_reset_token(token, secret_key=key) != victim.id


def test_tampered_and_foreign_tokens_are_rejected(canary_app):
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)

    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        key = canary_app.config['SECRET_KEY']
        token = create_password_reset_token(victim, secret_key=key)

        with pytest.raises(BadSignature):
            verify_password_reset_token(token + "x", secret_key=key)

        # Signed with a different secret — e.g. lifted from another deployment.
        forged = URLSafeTimedSerializer(
            'not-our-secret', salt=PASSWORD_RESET_SALT).dumps(
                {'user_id': victim.id, 'pw': victim.password_hash[:24]})
        with pytest.raises(BadSignature):
            verify_password_reset_token(forged, secret_key=key)


def test_token_expires(canary_app):
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        key = canary_app.config['SECRET_KEY']
        token = create_password_reset_token(victim, secret_key=key)
        with pytest.raises(SignatureExpired):
            verify_password_reset_token(token, secret_key=key, max_age=-1)


def test_token_dies_once_the_password_changes(canary_app):
    """This is what makes it single-use, with no schema change."""
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    with canary_app.app_context():
        from models import db, User
        victim = User.query.filter_by(email=VICTIM).first()
        key = canary_app.config['SECRET_KEY']
        token = create_password_reset_token(victim, secret_key=key)
        assert token_matches_current_password(token, victim, secret_key=key)

        victim.set_password(NEW_PASSWORD)
        db.session.commit()
        assert not token_matches_current_password(token, victim, secret_key=key)


# ---- the routes -------------------------------------------------------------

def test_request_does_not_reveal_whether_an_account_exists(canary_app):
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    client.get("/auth/logout")

    known = client.post("/auth/reset_password_request",
                        data={"email": VICTIM}, follow_redirects=True)
    unknown = client.post("/auth/reset_password_request",
                          data={"email": "nobody@example.com"},
                          follow_redirects=True)

    assert known.status_code == 200 and unknown.status_code == 200
    for page in (known.data, unknown.data):
        assert b"not found" not in page.lower()
    assert b"has an account" in known.data
    assert b"has an account" in unknown.data


def test_a_valid_token_resets_that_user_and_then_stops_working(canary_app):
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    client.get("/auth/logout")

    token = _token_for(canary_app, VICTIM)

    assert client.get(f"/auth/reset_password/{token}").status_code == 200

    resp = client.post(f"/auth/reset_password/{token}",
                       data={"password": NEW_PASSWORD,
                             "confirm_password": NEW_PASSWORD},
                       follow_redirects=True)
    assert resp.status_code == 200

    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        assert victim.check_password(NEW_PASSWORD)
        assert not victim.check_password(PASSWORD)

    # Replay must fail.
    replay = client.post(f"/auth/reset_password/{token}",
                         data={"password": "Another-Pass9",
                               "confirm_password": "Another-Pass9"},
                         follow_redirects=True)
    assert b"no longer valid" in replay.data
    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        assert victim.check_password(NEW_PASSWORD), "replayed token changed the password"


def test_a_garbage_token_never_renders_a_working_form(canary_app):
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    client.get("/auth/logout")

    resp = client.get("/auth/reset_password/not-a-real-token",
                      follow_redirects=True)
    assert b"no longer valid" in resp.data

    resp = client.post("/auth/reset_password/not-a-real-token",
                       data={"password": NEW_PASSWORD,
                             "confirm_password": NEW_PASSWORD},
                       follow_redirects=True)
    assert b"no longer valid" in resp.data
    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        assert victim.check_password(PASSWORD), "an unsigned token reset a password"


def test_one_users_token_cannot_reset_another_user(canary_app):
    """The headline defect, end to end."""
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    client.get("/auth/logout")
    _register(client, "attacker", ATTACKER)
    client.get("/auth/logout")

    attacker_token = _token_for(canary_app, ATTACKER)

    client.post(f"/auth/reset_password/{attacker_token}",
                data={"password": NEW_PASSWORD,
                      "confirm_password": NEW_PASSWORD},
                follow_redirects=True)

    with canary_app.app_context():
        from models import User
        victim = User.query.filter_by(email=VICTIM).first()
        attacker = User.query.filter_by(email=ATTACKER).first()
        assert victim.check_password(PASSWORD), "the victim's password was changed"
        assert attacker.check_password(NEW_PASSWORD)


def test_login_does_not_swallow_the_message_that_sent_you_there(canary_app):
    """login() starts with session.clear() for session-fixation hygiene.

    That also emptied the flash queue, so every message redirecting to the login
    page — including "your password has been reset" — was destroyed before the
    page rendered and the user saw nothing. Found during the 2026-09-04 QA work.
    """
    client = canary_app.test_client()
    _register(client, "victim", VICTIM)
    client.get("/auth/logout")

    token = _token_for(canary_app, VICTIM)
    resp = client.post(f"/auth/reset_password/{token}",
                       data={"password": NEW_PASSWORD,
                             "confirm_password": NEW_PASSWORD},
                       follow_redirects=True)

    assert b"password has been reset" in resp.data, (
        "the confirmation was swallowed on the way to the login page — check "
        "login() still preserves session['_flashes'] across session.clear()")
