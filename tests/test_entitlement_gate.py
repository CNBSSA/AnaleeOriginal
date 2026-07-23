"""Slice 5 — Analee entitlement gate tests (Festus 2026-07-13).

Rule: Analee is available only to a **Practice Club member** OR an
**Accountants / Analee subscriber**. The gate ships dark behind
``ANALEE_ENTITLEMENT_ENFORCED``.

Live-code note (updated for B-A2): a new self-registered user now defaults to
``subscription_status='login_only'`` — they CAN log in (``is_active`` includes
it) but are NOT Analee-entitled (``is_subscriber`` excludes it). So the gate's
block-path is now reachable by a real logged-in non-subscriber (as well as a
future Club-SSO login), and is also exercised deterministically by forcing the
entitlement helper. These tests cover:

* the pure ``entitlement.analee_entitled`` helper (subscriber / Club / anon);
* the enforcement flag;
* the before_request gate wiring — off = never consulted, on = admins &
  entitled pass, non-entitled redirect to the friendly page;
* the gate page is public (never traps a user).
"""

EMAIL = "gateuser@example.com"
PASSWORD = "Sup3rSecret!"


def _register_and_login(client):
    client.post("/auth/register", data={
        "username": "gateuser", "email": EMAIL,
        "password": PASSWORD, "confirm_password": PASSWORD,
    })
    client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})


def _make_admin(app):
    from models import db, User
    with app.app_context():
        u = User.query.filter_by(email=EMAIL).first()
        u.is_admin = True
        db.session.commit()


def _make_subscriber(app):
    """Promote the freshly-registered user to a full subscriber. Needed because
    a new sign-up now defaults to 'login_only' (can log in, not Analee-entitled)
    — B-A2. Setting 'active' simulates them subscribing / being provisioned."""
    from models import db, User
    with app.app_context():
        u = User.query.filter_by(email=EMAIL).first()
        u.subscription_status = "active"
        db.session.commit()


# ---- pure helper -----------------------------------------------------------

class _U:
    is_authenticated = True

    def __init__(self, status="active"):
        self.subscription_status = status


def test_helper_subscriber_entitled(canary_app):
    import entitlement
    with canary_app.test_request_context("/"):
        assert entitlement.analee_entitled(_U("active")) is True
        assert entitlement.analee_entitled(_U("pending")) is True


def test_helper_non_subscriber_not_entitled(canary_app):
    import entitlement
    with canary_app.test_request_context("/"):
        assert entitlement.analee_entitled(_U("inactive")) is False
        # 'login_only' (the new-signup default) can log in but is NOT entitled.
        assert entitlement.analee_entitled(_U("login_only")) is False


def test_helper_club_member_entitled_without_subscription(canary_app):
    import entitlement
    from flask import session
    with canary_app.test_request_context("/"):
        session["club_session"] = True
        assert entitlement.analee_entitled(_U("inactive")) is True
    # club_member_id is an equally-valid Club marker
    with canary_app.test_request_context("/"):
        session["club_member_id"] = 42
        assert entitlement.analee_entitled(_U("inactive")) is True


def test_helper_anonymous_not_entitled(canary_app):
    import entitlement
    with canary_app.test_request_context("/"):
        assert entitlement.analee_entitled(None) is False


def test_enforcement_flag(canary_app, monkeypatch):
    import entitlement
    monkeypatch.delenv("ANALEE_ENTITLEMENT_ENFORCED", raising=False)
    assert entitlement.enforcement_enabled() is False
    monkeypatch.setenv("ANALEE_ENTITLEMENT_ENFORCED", "True")
    assert entitlement.enforcement_enabled() is True


# ---- gate integration ------------------------------------------------------

def test_gate_off_never_consults_helper(canary_app, monkeypatch):
    """Flag off = complete no-op: the helper is not even called."""
    import entitlement
    monkeypatch.delenv("ANALEE_ENTITLEMENT_ENFORCED", raising=False)

    def _boom(_user):
        raise AssertionError("entitlement helper called while gate is OFF")

    monkeypatch.setattr(entitlement, "analee_entitled", _boom)
    client = canary_app.test_client()
    _register_and_login(client)
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_gate_on_allows_subscriber(canary_app, monkeypatch):
    monkeypatch.setenv("ANALEE_ENTITLEMENT_ENFORCED", "True")
    client = canary_app.test_client()
    _register_and_login(client)      # new sign-up defaults to 'login_only'
    _make_subscriber(canary_app)     # ...promote to a full subscriber ('active')
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_gate_on_blocks_non_entitled(canary_app, monkeypatch):
    """When the helper says not-entitled, the gate redirects to the notice."""
    import entitlement
    monkeypatch.setenv("ANALEE_ENTITLEMENT_ENFORCED", "True")
    monkeypatch.setattr(entitlement, "analee_entitled", lambda _user: False)
    client = canary_app.test_client()
    _register_and_login(client)
    resp = client.get("/dashboard")
    assert resp.status_code == 302
    assert "/entitlement-required" in resp.headers.get("Location", "")


def test_gate_on_admin_bypasses_helper(canary_app, monkeypatch):
    """Admins are always allowed, even if the helper would say no."""
    import entitlement
    monkeypatch.setenv("ANALEE_ENTITLEMENT_ENFORCED", "True")
    monkeypatch.setattr(entitlement, "analee_entitled", lambda _user: False)
    client = canary_app.test_client()
    _register_and_login(client)
    _make_admin(canary_app)
    resp = client.get("/dashboard", follow_redirects=False)
    # admin index/dashboard may redirect to the admin area, but NEVER to the gate
    assert "/entitlement-required" not in resp.headers.get("Location", "")
    assert resp.status_code in (200, 302)


def test_gate_page_is_public(canary_app):
    client = canary_app.test_client()
    resp = client.get("/entitlement-required")
    assert resp.status_code == 200
    assert b"Practice Club" in resp.data
