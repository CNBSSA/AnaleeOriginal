"""Analee → FileIt trial-balance tunnel (Festus 2026-09-29, SEND direction).

Dark by default; workspace sessions only; the same admin approval gate as the
share link / Send TB; the SAME workbook as Download Excel; the folder comes
from the Practice Club roster with the workspace's own ref; idempotent; every
refusal is a plain message, never a 500. HTTP is mocked — no network.
"""
import io
from unittest import mock

import openpyxl

from test_practice_send_tb_club import _club_workspace_client

REF = "club-41-7"


def _on(monkeypatch, base="https://fileit.test"):
    monkeypatch.setenv("ANALEE_FILEIT_PUSH_ENABLED", "True")
    monkeypatch.setenv("FILEIT_API_BASE_URL", base)
    monkeypatch.setenv("FILEIT_API_TOKEN", "osk_test")
    monkeypatch.setenv("CLUB_PRACTICE_SYNC_URL", "https://hub.test/practice/sync/")
    monkeypatch.setenv("CLUB_PRACTICE_SYNC_TOKEN", "hub-token")


class _R:
    def __init__(self, status=200, json=None):
        self.status_code, self._json = status, json or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def _wire(monkeypatch, *, upload_status=201, found=True, folders=None, credential=None):
    """One dispatcher for every POST: hub resolve, FileIt token, FileIt upload."""
    folders = folders or {REF: "folder-9"}

    def post(url, **kw):
        if url == "https://hub.test/practice/resolve/":
            ref = kw["json"]["external_ref"]
            body = {"found": found and ref in folders, "external_ref": folders.get(ref, "")}
            if credential:
                body["credential"] = credential(ref)
            return _R(json=body)
        if url.endswith("/auth/token"):
            return _R(json={"access_token": "tok-" + kw["json"]["client_id"], "expires_in": 600})
        if "/folders/" in url and url.endswith("/documents"):
            return _R(status=upload_status(url, kw) if callable(upload_status) else upload_status,
                      json={"id": 77})
        raise AssertionError(url)
    m = mock.Mock(side_effect=post)
    monkeypatch.setattr("fileit_pull.requests.post", m)
    return m


def _uploads(m):
    return [c for c in m.call_args_list if c.args[0].endswith("/documents")]


def test_dark_by_default_404_and_no_button(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    assert client.post("/fileit/trial-balance").status_code == 404
    assert b"File in FileIt" not in client.get("/trial-balance").data


def test_button_shows_in_a_workspace_when_on(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch)
    assert b"File in FileIt" in client.get("/trial-balance").data


def test_outside_a_workspace_nothing_is_sent(canary_app, monkeypatch):
    from test_tb_approval_gate import _client
    client, _ = _client(canary_app, n=91)
    _on(monkeypatch)
    m = _wire(monkeypatch)
    assert b"File in FileIt" not in client.get("/trial-balance").data
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert b"Open a client first" in r.data
    assert m.call_count == 0


def test_an_unapproved_trial_balance_is_refused_with_the_gate_message(canary_app, monkeypatch):
    from reports import tb_approval
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    # Figures exist but nobody approved them.
    from models import Account, Transaction, UploadedFile, User, db
    from provisioning import WORKSPACE_EMAIL_DOMAIN
    from datetime import datetime
    with canary_app.app_context():
        ws = User.query.filter_by(email=f"client+{REF}@{WORKSPACE_EMAIL_DOMAIN}").one()
        bank = Account.query.filter_by(user_id=ws.id, link="ca.810.001").first()
        sales = Account.query.filter_by(user_id=ws.id, link="i.100.000").first()
        f = UploadedFile(filename="s.csv", user_id=ws.id, bank_account_id=bank.id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(date=datetime.now(), description="Sale", amount=100.0,
                                   user_id=ws.id, account_id=sales.id, file_id=f.id))
        db.session.commit()
    _on(monkeypatch)
    m = _wire(monkeypatch)
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert r.status_code == 200
    assert tb_approval.NOT_APPROVED_MESSAGE.encode() in r.data
    assert _uploads(m) == []


def test_happy_path_files_the_same_workbook_as_download_excel(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch)
    m = _wire(monkeypatch)
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert b"in the client&#39;s FileIt folder" in r.data and b"Filed analee-trial-balance-" in r.data
    resolve = m.call_args_list[0]
    assert resolve.kwargs["json"] == {"product": "analee", "external_ref": REF,
                                      "target_product": "bookstorage"}
    (up,) = _uploads(m)
    assert up.args[0] == "https://fileit.test/api/integration/v1/folders/folder-9/documents"
    assert up.kwargs["allow_redirects"] is False and up.kwargs["timeout"] <= 30
    assert up.kwargs["headers"]["Authorization"] == "Bearer osk_test"
    key = up.kwargs["headers"]["Idempotency-Key"]
    assert key.startswith("analee-tb-") and len(key) <= 128
    assert up.kwargs["data"]["document_kind"] == "general"
    assert up.kwargs["data"]["source_app"] == "analee"
    name, content, ctype = up.kwargs["files"]["file"]
    assert name.endswith(".xlsx") and ctype.endswith("spreadsheetml.sheet")

    exported = client.get("/trial-balance/export").data

    def cells(raw):
        wb = openpyxl.load_workbook(io.BytesIO(raw))
        return [[c.value for c in row] for ws in wb.worksheets for row in ws.iter_rows()]
    assert cells(content) == cells(exported)


def test_the_same_trial_balance_is_never_filed_twice(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch)
    seen = set()

    def status(url, kw):
        k = kw["headers"]["Idempotency-Key"]
        if k in seen:
            return 200
        seen.add(k)
        return 201
    m = _wire(monkeypatch, upload_status=status)
    client.post("/fileit/trial-balance")
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert b"not filed twice" in r.data
    keys = [c.kwargs["headers"]["Idempotency-Key"] for c in _uploads(m)]
    assert len(keys) == 2 and keys[0] == keys[1]


def test_refusals_are_plain_messages(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch)
    for status, words in ((401, b"did not accept"), (403, b"may not write"),
                          (404, b"could not find the client"), (500, b"status 500")):
        _wire(monkeypatch, upload_status=status)
        r = client.post("/fileit/trial-balance", follow_redirects=True)
        assert r.status_code == 200 and words in r.data, status


def test_an_unlinked_client_is_refused_without_an_upload(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch)
    m = _wire(monkeypatch, found=False)
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert b"not linked to a FileIt folder" in r.data
    assert _uploads(m) == []


def test_fileit_unreachable_is_not_a_500(canary_app, monkeypatch):
    import requests
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch)
    m = _wire(monkeypatch)
    orig = m.side_effect

    def boom(url, **kw):
        if url.endswith("/documents"):
            raise requests.ConnectionError("down")
        return orig(url, **kw)
    m.side_effect = boom
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert r.status_code == 200 and b"could not be reached" in r.data


def test_a_non_https_fileit_address_is_refused(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch)
    _on(monkeypatch, base="http://fileit.test")
    m = _wire(monkeypatch)
    r = client.post("/fileit/trial-balance", follow_redirects=True)
    assert b"over HTTPS" in r.data
    assert m.call_count == 0


def test_firm_a_never_reaches_firm_b_folder_or_key(canary_app, monkeypatch):
    """Two practices' clients: each send resolves with its OWN ref, goes to its
    OWN folder, and uses the credential the hub issued for THAT practice."""
    ref_b = "club-52-3"
    a = _club_workspace_client(canary_app, monkeypatch)
    b = _club_workspace_client(canary_app, monkeypatch, ref=ref_b, name="Dlamini Holdings")
    _on(monkeypatch)
    monkeypatch.delenv("FILEIT_API_TOKEN")
    import fileit_pull
    fileit_pull._tokens_by_client.clear()
    m = _wire(monkeypatch, folders={REF: "folder-A", ref_b: "folder-B"},
              credential=lambda ref: {"client_id": f"cid-{ref}", "client_secret": f"s-{ref}"})
    a.post("/fileit/trial-balance")
    b.post("/fileit/trial-balance")
    ups = _uploads(m)
    assert [u.args[0].rsplit("/", 2)[-2] for u in ups] == ["folder-A", "folder-B"]
    assert ups[0].kwargs["headers"]["Authorization"] == f"Bearer tok-cid-{REF}"
    assert ups[1].kwargs["headers"]["Authorization"] == f"Bearer tok-cid-{ref_b}"
    assert ups[0].kwargs["headers"]["Idempotency-Key"] != ups[1].kwargs["headers"]["Idempotency-Key"]
    resolved = [c.kwargs["json"]["external_ref"] for c in m.call_args_list
                if c.args[0].endswith("/resolve/")]
    assert resolved == [REF, ref_b]
