"""QA PC-3 (Festus, 2026-09-27): Send TB -> THE ACCOUNTANTS from a workspace
the Practice Club provisioned (ref ``club-<member>-<client>``).

The button renders in every workspace session; these tests pin that it hands
THE ACCOUNTANTS what it needs to find the client (the client's name, an HTTPS
link) and that every refusal names its real cause in plain words instead of
"not configured" / "could not complete that right now".
Frozen analysis engine + chart/TB core untouched: the share link is minted by
the same token helper Copy-share-link uses.
"""
from unittest import mock

API = "https://acc.example/api/practice"
SECRET = "s3cr3t-provisioning-key"


class _FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = str(self._body)

    def json(self):
        return self._body


def _enable(monkeypatch, api=API, secret=SECRET):
    monkeypatch.setenv("ANALEE_PRACTICE_LAYER_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    if secret:
        monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", secret)
    else:
        monkeypatch.delenv("ANALEE_PROVISIONING_SECRET", raising=False)
    if api:
        monkeypatch.setenv("ACCOUNTANTS_PRACTICE_API_URL", api)
    else:
        monkeypatch.delenv("ACCOUNTANTS_PRACTICE_API_URL", raising=False)


def _club_workspace_client(app, monkeypatch, ref="club-41-7",
                           name="Mokoena Trading", approve=True):
    """A test client sitting in a Club-provisioned workspace session, entered
    through the real signed ``/workspace/enter`` door."""
    from provisioning import ensure_workspace
    monkeypatch.setenv("ANALEE_PROVISIONING_ENABLED", "True")
    monkeypatch.setenv("ANALEE_PROVISIONING_SECRET", SECRET)
    with app.app_context():
        assert "error" not in ensure_workspace(ref, name, "Close Corporation")
    client = app.test_client()
    link = client.post("/api/provisioning/analee/workspace/login-link",
                       json={"client_ref": ref},
                       headers={"Authorization": f"Bearer {SECRET}"}).get_json()
    assert link["found"]
    assert client.get(link["url_path"]).status_code == 302
    if approve:
        _approve_workspace_tb(app, ref)
    return client


def _approve_workspace_tb(app, ref):
    """Festus 2026-09-28: an administrator approves the trial balance before
    Send TB may hand it over. These tests are about the hand-over itself."""
    from models import Account, Transaction, UploadedFile, User, db
    from provisioning import WORKSPACE_EMAIL_DOMAIN
    from reports import tb_approval
    from reports.trial_balance_service import load_trial_balance
    from datetime import datetime
    with app.app_context():
        ws = User.query.filter_by(email=f"client+{ref}@{WORKSPACE_EMAIL_DOMAIN}").one()
        if not Transaction.query.filter_by(user_id=ws.id).count():
            bank = Account.query.filter_by(user_id=ws.id, link="ca.810.001").first()
            sales = Account.query.filter_by(user_id=ws.id, link="i.100.000").first()
            f = UploadedFile(filename="s.csv", user_id=ws.id, bank_account_id=bank.id)
            db.session.add(f)
            db.session.flush()
            db.session.add(Transaction(date=datetime.now(), description="Sale", amount=100.0,
                                       user_id=ws.id, account_id=sales.id, file_id=f.id))
            db.session.commit()
        admin = User.query.filter_by(username="tb-admin").first()
        if admin is None:
            admin = User(username="tb-admin", email="tb-admin@example.com",
                         subscription_status="active", is_admin=True)
            admin.set_password("password")
            db.session.add(admin)
            db.session.commit()
        ctx = load_trial_balance(ws.id)
        record = tb_approval.request_approval(ws.id, ctx, requested_by=ws.id)
        tb_approval.decide(record, admin_id=admin.id, approve=True)


def test_club_workspace_sends_name_and_https_link(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch)
    fake = _FakeResponse(body={"received": True, "client_number": "ACC-0003"})
    with mock.patch("practice_layer.requests.post", return_value=fake) as post:
        r = client.post("/practice/send-tb", follow_redirects=True)
    assert post.call_args.args[0] == API + "/tb-drop"
    payload = post.call_args.kwargs["json"]
    assert payload["client_ref"] == "club-41-7"
    assert payload["client_name"] == "Mokoena Trading"
    assert payload["share_url"].startswith("https://")
    assert "/api/trial-balance/shared/" in payload["share_url"]
    assert b"Trial balance sent" in r.data


def test_no_firm_reason_from_accountants_is_shown(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch)
    msg = ("Your Workstation identity has no firm in THE ACCOUNTANTS yet. Open "
           "THE ACCOUNTANTS from the Workstation once and press Create firm.")
    fake = _FakeResponse(status=404, body={"error": "no firm", "message": msg})
    with mock.patch("practice_layer.requests.post", return_value=fake):
        r = client.post("/practice/send-tb", follow_redirects=True)
    assert b"no firm in THE ACCOUNTANTS" in r.data
    assert b"could not complete" not in r.data


def test_raw_payload_is_never_shown(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch)
    fake = _FakeResponse(status=404, body={"error": "not found",
                                           "message": '{"error": "x"}'})
    with mock.patch("practice_layer.requests.post", return_value=fake):
        r = client.post("/practice/send-tb", follow_redirects=True)
    assert b'{&#34;error' not in r.data and b'{"error' not in r.data
    assert b"ANALEE_PROVISIONING_ENABLED" in r.data  # the 404 translation


def test_key_mismatch_is_named(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch)
    with mock.patch("practice_layer.requests.post",
                    return_value=_FakeResponse(status=401,
                                               body={"error": "unauthorized"})):
        r = client.post("/practice/send-tb", follow_redirects=True)
    assert b"ANALEE_PROVISIONING_SECRET" in r.data
    assert b"unauthorized" not in r.data


def test_missing_url_names_the_variable(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch, api="")
    with mock.patch("practice_layer.requests.post") as post:
        r = client.post("/practice/send-tb", follow_redirects=True)
    post.assert_not_called()
    assert b"ACCOUNTANTS_PRACTICE_API_URL" in r.data


def test_missing_secret_names_the_variable(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _enable(monkeypatch, secret="")
    with mock.patch("practice_layer.requests.post") as post:
        r = client.post("/practice/send-tb", follow_redirects=True)
    post.assert_not_called()
    assert b"ANALEE_PROVISIONING_SECRET" in r.data


def test_an_unapproved_trial_balance_is_not_sent(canary_app, monkeypatch):
    """Festus 2026-09-28: Send TB refuses until an administrator has approved
    the trial balance, and says so; THE ACCOUNTANTS is never called."""
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _enable(monkeypatch)
    with mock.patch("practice_layer.requests.post") as post:
        r = client.post("/practice/send-tb", follow_redirects=True)
    post.assert_not_called()
    assert b"has not been approved for sending" in r.data
