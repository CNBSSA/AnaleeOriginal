"""One-Login Practice Layer P1 (Festus 2026-07-18).

One accountant login → My Clients page listing the practice's client
workspaces (the hidden alias accounts the 07-15 seam creates) → server-side
switch into a client and back. Dark behind ANALEE_PRACTICE_LAYER_ENABLED;
the S2S bind endpoint lives in the sealed provisioning seam with the same
fail-closed bearer. Frozen analysis engine + chart/TB core: called, never
changed — nothing in these tests touches them beyond the existing chart
provisioning the seam already performs.
"""

PRACTICE_URL = "/practice"
RETURN_URL = "/practice/return"
BIND_URL = "/api/provisioning/analee/practice"
SECRET = "s3cr3t-provisioning-key"


def _enable(monkeypatch):
    monkeypatch.setenv("ANALEE_PRACTICE_LAYER_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", SECRET)


def _auth():
    return {"Authorization": f"Bearer {SECRET}"}


def _mk_accountant(app, email="acct@firm.co.za", firm_ref="acc-7",
                   firm_name="Firm A"):
    from models import db, User, PracticeLink
    with app.app_context():
        user = User(username=email, email=email, subscription_status="active")
        user.set_password("pw-123456")
        db.session.add(user)
        db.session.flush()
        db.session.add(PracticeLink(accountant_user_id=user.id,
                                    firm_ref=firm_ref, firm_name=firm_name))
        db.session.commit()
        return user.id


def _mk_workspace(app, client_ref, name):
    from provisioning import ensure_workspace
    with app.app_context():
        result = ensure_workspace(client_ref, name)
        assert "error" not in result
        return result["workspace_user_id"]


def _login(client, email="acct@firm.co.za", password="pw-123456"):
    return client.post("/auth/login",
                       data={"email": email, "password": password},
                       follow_redirects=True)


# ---------------------------------------------------------------- dark by default

def test_practice_dark_when_disabled(canary_app, monkeypatch):
    monkeypatch.delenv("ANALEE_PRACTICE_LAYER_ENABLED", raising=False)
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", SECRET)
    _mk_accountant(canary_app)
    client = canary_app.test_client()
    _login(client)
    assert client.get(PRACTICE_URL).status_code == 404
    assert client.post(RETURN_URL).status_code == 404
    assert client.post("/practice/open/1").status_code == 404


def test_bind_endpoint_dark_when_provisioning_disabled(canary_app, monkeypatch):
    monkeypatch.delenv("ANALEE_PROVISIONING_ENABLED", raising=False)
    client = canary_app.test_client()
    assert client.post(BIND_URL, json={}).status_code == 404


# ---------------------------------------------------------------- S2S bind

def test_bind_fail_closed_auth(canary_app, monkeypatch):
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.delenv("ANALEE_PROVISIONING_SECRET", raising=False)
    client = canary_app.test_client()
    assert client.post(BIND_URL, json={},
                       headers={"Authorization": "Bearer x"}).status_code == 503
    monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", SECRET)
    assert client.post(BIND_URL, json={},
                       headers={"Authorization": "Bearer no"}).status_code == 401


def test_bind_validates_inputs(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(BIND_URL, json={"firm_ref": "acc-7"}, headers=_auth())
    assert r.status_code == 400
    r = client.post(BIND_URL, json={"accountant_email": "a@b.co"}, headers=_auth())
    assert r.status_code == 400
    # Workspace alias emails are refused — the bind can never hijack a hidden
    # workspace identity into an accountant login.
    r = client.post(BIND_URL, json={
        "accountant_email": "client+acc-7-1@ws.theaccountants.local",
        "firm_ref": "acc-7"}, headers=_auth())
    assert r.status_code == 400


def test_bind_existing_user_and_idempotent_rebind(canary_app, monkeypatch):
    _enable(monkeypatch)
    from models import db, User, PracticeLink
    with canary_app.app_context():
        user = User(username="ex@firm.co.za", email="ex@firm.co.za",
                    subscription_status="active")
        user.set_password("pw-123456")
        db.session.add(user)
        db.session.commit()
        uid = user.id
    client = canary_app.test_client()
    r = client.post(BIND_URL, json={
        "accountant_email": "EX@firm.co.za", "firm_ref": "acc-9",
        "firm_name": "Firm B"}, headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body["linked"] is True and body["created_user"] is False
    assert body["accountant_user_id"] == uid
    # Re-bind updates, never duplicates.
    r = client.post(BIND_URL, json={
        "accountant_email": "ex@firm.co.za", "firm_ref": "acc-9",
        "firm_name": "Firm B (Pty) Ltd"}, headers=_auth())
    assert r.status_code == 200
    with canary_app.app_context():
        links = PracticeLink.query.filter_by(accountant_user_id=uid).all()
        assert len(links) == 1
        assert links[0].firm_name == "Firm B (Pty) Ltd"


def test_bind_creates_missing_accountant(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(BIND_URL, json={
        "accountant_email": "new@firm.co.za", "firm_ref": "acc-11"},
        headers=_auth())
    assert r.status_code == 200
    assert r.get_json()["created_user"] is True
    from models import db, User
    with canary_app.app_context():
        user = User.query.filter(
            db.func.lower(User.email) == "new@firm.co.za").first()
        assert user is not None and user.subscription_status == "active"


# ---------------------------------------------------------------- My Clients

def test_my_clients_lists_only_own_firm(canary_app, monkeypatch):
    _enable(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    _mk_workspace(canary_app, "acc-7-1", "Mokoena Trading")
    _mk_workspace(canary_app, "acc-7-2", "Dlamini Stores")
    _mk_workspace(canary_app, "acc-70-9", "Prefix Collision Co")  # acc-70 ≠ acc-7
    _mk_workspace(canary_app, "acc-8-1", "Other Firm Client")
    client = canary_app.test_client()
    _login(client)
    r = client.get(PRACTICE_URL)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Mokoena Trading" in html and "Dlamini Stores" in html
    assert "Prefix Collision Co" not in html
    assert "Other Firm Client" not in html


def test_my_clients_404_for_user_without_link(canary_app, monkeypatch):
    _enable(monkeypatch)
    from models import db, User
    with canary_app.app_context():
        user = User(username="plain@x.co.za", email="plain@x.co.za",
                    subscription_status="active")
        user.set_password("pw-123456")
        db.session.add(user)
        db.session.commit()
    client = canary_app.test_client()
    _login(client, email="plain@x.co.za")
    assert client.get(PRACTICE_URL).status_code == 404


# ---------------------------------------------------------------- switching

def test_open_switches_and_return_comes_back(canary_app, monkeypatch):
    _enable(monkeypatch)
    acct_id = _mk_accountant(canary_app, firm_ref="acc-7")
    ws_id = _mk_workspace(canary_app, "acc-7-1", "Mokoena Trading")
    client = canary_app.test_client()
    _login(client)

    r = client.post(f"/practice/open/{ws_id}")
    assert r.status_code == 302 and "/dashboard" in r.headers["Location"]
    with client.session_transaction() as sess:
        assert sess.get("workspace_session") is True
        assert sess.get("practice_return_uid") == acct_id
        assert sess.get("_user_id") == str(ws_id)

    r = client.post(RETURN_URL)
    assert r.status_code == 302 and r.headers["Location"].endswith(PRACTICE_URL)
    with client.session_transaction() as sess:
        assert sess.get("workspace_session") is None
        assert sess.get("practice_return_uid") is None
        assert sess.get("_user_id") == str(acct_id)
    assert client.get(PRACTICE_URL).status_code == 200


def test_open_refuses_foreign_workspace(canary_app, monkeypatch):
    _enable(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    foreign_id = _mk_workspace(canary_app, "acc-8-1", "Other Firm Client")
    client = canary_app.test_client()
    _login(client)
    r = client.post(f"/practice/open/{foreign_id}")
    assert r.status_code == 302 and r.headers["Location"].endswith(PRACTICE_URL)
    with client.session_transaction() as sess:
        assert sess.get("workspace_session") is None  # identity unchanged


def test_return_without_switch_redirects_to_login(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(RETURN_URL)
    assert r.status_code == 302 and "/login" in r.headers["Location"]


# ---------------------------------------------------- P2: ACC numbers + add

API = "https://acc.example/api/practice"


def _enable_api(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("ACCOUNTANTS_PRACTICE_API_URL", API)


class _FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}

    def json(self):
        return self._body


def test_ensure_seam_upserts_client_number(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post("/api/provisioning/analee/workspace", json={
        "client_ref": "acc-7-42", "client_name": "Mokoena Traders",
        "client_number": "ACC-0042"}, headers=_auth())
    assert r.status_code == 200
    from models import PracticeClientMeta
    with canary_app.app_context():
        meta = PracticeClientMeta.query.filter_by(
            workspace_user_id=r.get_json()["workspace_user_id"]).first()
        assert meta is not None and meta.client_number == "ACC-0042"
    # Re-ensure with an updated number keeps it in step, never duplicates.
    r2 = client.post("/api/provisioning/analee/workspace", json={
        "client_ref": "acc-7-42", "client_name": "Mokoena Traders",
        "client_number": "ACC-0099"}, headers=_auth())
    assert r2.status_code == 200
    with canary_app.app_context():
        metas = PracticeClientMeta.query.filter_by(
            workspace_user_id=r.get_json()["workspace_user_id"]).all()
        assert len(metas) == 1 and metas[0].client_number == "ACC-0099"


def test_my_clients_shows_acc_number(canary_app, monkeypatch):
    _enable(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    from provisioning import ensure_workspace
    with canary_app.app_context():
        ensure_workspace("acc-7-1", "Mokoena Trading", client_number="ACC-0001")
    client = canary_app.test_client()
    _login(client)
    html = client.get(PRACTICE_URL).get_data(as_text=True)
    assert "ACC-0001" in html


def test_add_client_creates_both_sides(canary_app, monkeypatch):
    _enable_api(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    client = canary_app.test_client()
    _login(client)
    from unittest import mock
    fake = _FakeResponse(body={
        "created": True, "client_ref": "acc-7-55", "client_number": "ACC-0003",
        "name": "Dlamini Stores", "entity_type": "cc",
        "analee_entity_name": "Close Corporation"})
    with mock.patch("practice_layer.requests.post", return_value=fake) as post:
        r = client.post("/practice/add",
                        data={"name": "Dlamini Stores", "entity_type": "cc"})
    assert r.status_code == 302
    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == API + "/clients"
    assert kwargs["json"] == {"firm_ref": "acc-7", "name": "Dlamini Stores",
                              "entity_type": "cc"}
    assert kwargs["headers"]["Authorization"] == f"Bearer {SECRET}"
    # The workspace now exists locally, with the number, and is listed.
    html = client.get(PRACTICE_URL).get_data(as_text=True)
    assert "Dlamini Stores" in html and "ACC-0003" in html


def test_add_client_s2s_failure_is_friendly_and_creates_nothing(canary_app, monkeypatch):
    _enable_api(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    client = canary_app.test_client()
    _login(client)
    from unittest import mock
    with mock.patch("practice_layer.requests.post",
                    return_value=_FakeResponse(status=503)):
        r = client.post("/practice/add", data={"name": "Ghost Client"},
                        follow_redirects=True)
    assert b"could not complete" in r.data
    from models import db, User
    with canary_app.app_context():
        assert User.query.filter(
            User.email.like("%ghost%")).first() is None


def test_add_client_hidden_without_api_env(canary_app, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.delenv("ACCOUNTANTS_PRACTICE_API_URL", raising=False)
    _mk_accountant(canary_app, firm_ref="acc-7")
    client = canary_app.test_client()
    _login(client)
    html = client.get(PRACTICE_URL).get_data(as_text=True)
    assert "Add a client" not in html


# ------------------------------------------------------------ P3: Send TB

def test_send_tb_hands_share_url_to_accountants(canary_app, monkeypatch):
    _enable_api(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    ws_id = _mk_workspace(canary_app, "acc-7-1", "Mokoena Trading")
    client = canary_app.test_client()
    _login(client)
    assert client.post(f"/practice/open/{ws_id}").status_code == 302

    from unittest import mock
    fake = _FakeResponse(body={"received": True, "client_number": "ACC-0001"})
    with mock.patch("practice_layer.requests.post", return_value=fake) as post:
        r = client.post("/practice/send-tb", follow_redirects=True)
    assert post.call_args.args[0] == API + "/tb-drop"
    payload = post.call_args.kwargs["json"]
    assert payload["client_ref"] == "acc-7-1"
    assert "/api/trial-balance/shared/" in payload["share_url"]
    assert payload["share_url"].startswith("http")
    assert b"Trial balance sent" in r.data


def test_send_tb_refused_outside_workspace_session(canary_app, monkeypatch):
    _enable_api(monkeypatch)
    _mk_accountant(canary_app, firm_ref="acc-7")
    client = canary_app.test_client()
    _login(client)
    from unittest import mock
    with mock.patch("practice_layer.requests.post") as post:
        r = client.post("/practice/send-tb")
    post.assert_not_called()
    assert r.status_code == 302
