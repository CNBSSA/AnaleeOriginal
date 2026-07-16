"""Client workspaces (scoped re-open, Festus 2026-07-15).

THE ACCOUNTANTS orchestrates one Analee workspace per firm client through the
sealed provisioning seam: idempotent ensure (deterministic alias email +
CompanySettings + entity-correct chart via the frozen service), short-TTL
signed login links, and the browser entry route. Same dark flag + fail-closed
bearer as the base endpoint; no schema change.
"""

ENSURE_URL = "/api/provisioning/analee/workspace"
LINK_URL = "/api/provisioning/analee/workspace/login-link"
ENTER_URL = "/workspace/enter"
SECRET = "s3cr3t-provisioning-key"


def _enable(monkeypatch):
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", SECRET)


def _auth():
    return {"Authorization": f"Bearer {SECRET}"}


def _ensure(client, **payload):
    body = {"client_ref": "acc-7-42", "client_name": "Mokoena Traders"}
    body.update(payload)
    return client.post(ENSURE_URL, json=body, headers=_auth())


# ---------------------------------------------------------------- dark / auth

def test_workspace_routes_dark_when_disabled(canary_app, monkeypatch):
    monkeypatch.delenv("ANALEE_PROVISIONING_ENABLED", raising=False)
    client = canary_app.test_client()
    assert client.post(ENSURE_URL, json={}).status_code == 404
    assert client.post(LINK_URL, json={}).status_code == 404
    assert client.get(ENTER_URL + "?token=x").status_code == 404


def test_workspace_503_when_secret_unset(canary_app, monkeypatch):
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.delenv("ANALEE_PROVISIONING_SECRET", raising=False)
    client = canary_app.test_client()
    r = client.post(ENSURE_URL, json={}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 503
    # Browser route behaves dark (nothing could have been minted).
    assert client.get(ENTER_URL + "?token=x").status_code == 404


def test_workspace_401_on_wrong_bearer(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(ENSURE_URL, json={}, headers={"Authorization": "Bearer no"})
    assert r.status_code == 401
    r = client.post(LINK_URL, json={}, headers={"Authorization": "Bearer no"})
    assert r.status_code == 401


def test_workspace_400_on_bad_client_ref(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = _ensure(client, client_ref="   ***   ")
    assert r.status_code == 400


# ------------------------------------------------------------------- ensure

def test_ensure_creates_workspace_with_chart(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = _ensure(client)
    assert r.status_code == 200
    body = r.get_json()
    assert body["created"] is True
    assert body["client_ref"] == "acc-7-42"
    assert body["email"] == "client+acc-7-42@ws.theaccountants.local"
    assert body["company"] == "Mokoena Traders"
    assert body["chart_provisioned"] is True

    from models import Account, CompanySettings, User
    with canary_app.app_context():
        user = User.query.filter_by(email=body["email"]).first()
        assert user is not None
        assert user.subscription_status == "active"
        settings = CompanySettings.query.filter_by(user_id=user.id).first()
        assert settings.company_name == "Mokoena Traders"
        # Entity-correct chart copied by the FROZEN service (called, not changed).
        assert Account.query.filter_by(user_id=user.id).count() > 0


def test_ensure_is_idempotent_and_reactivates(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    first = _ensure(client).get_json()

    # Simulate a revocation, then re-ensure with a renamed client.
    from models import db, User
    with canary_app.app_context():
        u = User.query.get(first["workspace_user_id"])
        u.subscription_status = "inactive"
        db.session.commit()

    again = _ensure(client, client_name="Mokoena Traders (Pty) Ltd").get_json()
    assert again["created"] is False
    assert again["workspace_user_id"] == first["workspace_user_id"]
    assert again["company"] == "Mokoena Traders (Pty) Ltd"
    from models import User as U2
    with canary_app.app_context():
        assert U2.query.get(first["workspace_user_id"]).subscription_status == "active"


def test_ensure_respects_entity_name(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    body = _ensure(client, client_ref="acc-7-99", client_name="J Naidoo",
                   entity_name="Sole Proprietor").get_json()
    assert body["entity"] == "Sole Proprietor"
    from models import Account, User
    with canary_app.app_context():
        user = User.query.filter_by(email=body["email"]).first()
        names = {a.name for a in Account.query.filter_by(user_id=user.id)}
        assert "Drawings Account" in names          # sole-prop equity, not shares
        assert "Share Capital - Ordinary" not in names


# --------------------------------------------------------------- login link

def test_login_link_refuses_regular_accounts(canary_app, monkeypatch):
    _enable(monkeypatch)
    from models import db, User
    with canary_app.app_context():
        u = User(username="human", email="human@example.com")
        u.set_password("Sup3rSecret!")
        db.session.add(u)
        db.session.commit()
    client = canary_app.test_client()
    r = client.post(LINK_URL, json={"email": "human@example.com"},
                    headers=_auth())
    assert r.status_code == 400  # never a takeover vector for human accounts


def test_login_link_unknown_workspace_found_false(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(LINK_URL, json={"client_ref": "no-such-client"},
                    headers=_auth())
    assert r.status_code == 200
    assert r.get_json()["found"] is False


def test_login_link_includes_absolute_url_when_public_base_set(
        canary_app, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("ANALEE_PUBLIC_BASE_URL", "https://analee.test")
    client = canary_app.test_client()
    _ensure(client)
    link = client.post(LINK_URL, json={"client_ref": "acc-7-42"},
                       headers=_auth()).get_json()
    assert link["login_url"].startswith(
        "https://analee.test/workspace/enter?token=")


def test_login_link_round_trip_logs_into_workspace(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    _ensure(client)
    link = client.post(LINK_URL, json={"client_ref": "acc-7-42"},
                       headers=_auth()).get_json()
    assert link["found"] is True
    assert link["client_ref"] == "acc-7-42"
    assert link["url_path"].startswith(ENTER_URL + "?token=")

    r = client.get(link["url_path"])
    assert r.status_code == 302
    assert "login" not in r.headers["Location"]
    with client.session_transaction() as sess:
        assert sess.get("workspace_session") is True
        assert sess["workspace_email"].endswith("@ws.theaccountants.local")


def test_expired_link_redirects_to_login(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    _ensure(client)
    link = client.post(LINK_URL, json={"client_ref": "acc-7-42"},
                       headers=_auth()).get_json()
    monkeypatch.setenv("ANALEE_WORKSPACE_LINK_TTL", "-1")  # everything expired
    r = client.get(link["url_path"])
    assert r.status_code == 302
    assert "login" in r.headers["Location"]
    with client.session_transaction() as sess:
        assert "workspace_session" not in sess


def test_tampered_link_redirects_to_login(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.get(ENTER_URL + "?token=forged.token.value")
    assert r.status_code == 302
    assert "login" in r.headers["Location"]


def test_revoked_workspace_cannot_enter(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    made = _ensure(client).get_json()
    link = client.post(LINK_URL, json={"client_ref": "acc-7-42"},
                       headers=_auth()).get_json()
    from models import db, User
    with canary_app.app_context():
        u = User.query.get(made["workspace_user_id"])
        u.subscription_status = "inactive"
        db.session.commit()
    r = client.get(link["url_path"])
    assert r.status_code == 302
    assert "login" in r.headers["Location"]
