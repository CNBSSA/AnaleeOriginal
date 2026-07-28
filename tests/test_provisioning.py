"""Slice 5b — Analee cross-product provisioning endpoint (dark, fail-closed).

THE ACCOUNTANTS calls POST /api/provisioning/analee (server-to-server, bearer
secret) to grant/revoke full Analee access by email. Ships dark behind
ANALEE_PROVISIONING_ENABLED + ANALEE_PROVISIONING_SECRET. Tests cover: dark 404,
missing-secret 503, wrong-bearer 401, missing-email 400, activate/deactivate an
existing user by email, and the not-found (found=False) path.
"""

URL = "/api/provisioning/analee"
SECRET = "s3cr3t-provisioning-key"


def _enable(monkeypatch):
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", SECRET)


def _make_user(app, email, status="inactive"):
    from models import db, User
    with app.app_context():
        u = User(username="prov", email=email)
        # set a password if the model requires one; tolerate either API
        if hasattr(u, "set_password"):
            u.set_password("Sup3rSecret!")
        u.subscription_status = status
        db.session.add(u)
        db.session.commit()
        return u.id


def _status(app, email):
    from models import User
    with app.app_context():
        u = User.query.filter_by(email=email).first()
        return u.subscription_status if u else None


def test_dark_returns_404_when_disabled(canary_app, monkeypatch):
    monkeypatch.delenv("ANALEE_PROVISIONING_ENABLED", raising=False)
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "x@e.com"})
    assert r.status_code == 404


def test_503_when_secret_unset(canary_app, monkeypatch):
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.delenv("ANALEE_PROVISIONING_SECRET", raising=False)
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "x@e.com"},
                    headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 503


def test_401_on_wrong_bearer(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "x@e.com"},
                    headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_400_on_missing_email(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(URL, json={"entitled": True},
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 400


def test_activates_existing_user_by_email(canary_app, monkeypatch):
    _enable(monkeypatch)
    _make_user(canary_app, "grant@e.com", status="inactive")
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "GRANT@e.com", "entitled": True},
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["found"] is True
    assert body["subscription_status"] == "active"
    assert _status(canary_app, "grant@e.com") == "active"


def test_deactivates_on_entitled_false(canary_app, monkeypatch):
    _enable(monkeypatch)
    _make_user(canary_app, "revoke@e.com", status="active")
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "revoke@e.com", "entitled": False},
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200
    assert r.get_json()["subscription_status"] == "inactive"
    assert _status(canary_app, "revoke@e.com") == "inactive"


def test_first_time_buyer_account_is_created(canary_app, monkeypatch):
    """G10 / B-A1: an entitled email with no existing account is CREATED
    (active), so the 'buy Accountants → get Analee' promise holds for a
    first-time buyer."""
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "NewBuyer@e.com", "entitled": True},
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["found"] is False          # no PRE-EXISTING account...
    assert body["created"] is True          # ...but one was created
    assert body["subscription_status"] == "active"
    # the account now exists and is entitled (email stored as given)
    assert _status(canary_app, "NewBuyer@e.com") == "active"


def test_unknown_email_not_entitled_creates_nothing(canary_app, monkeypatch):
    """A deactivation for an unknown email is a no-op — never creates an
    account just to mark it inactive."""
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(URL, json={"email": "ghost@e.com", "entitled": False},
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["found"] is False
    assert body.get("created") is not True
    assert _status(canary_app, "ghost@e.com") is None  # nothing created


def test_workspace_alias_email_is_refused_for_creation(canary_app, monkeypatch):
    """A workspace-alias email is owned exclusively by ensure_workspace — the
    buyer-create path must refuse to mint one."""
    _enable(monkeypatch)
    client = canary_app.test_client()
    alias = "client+acme@ws.theaccountants.local"
    r = client.post(URL, json={"email": alias, "entitled": True},
                    headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["found"] is False
    assert body["created"] is False
    assert _status(canary_app, alias) is None  # not created here
