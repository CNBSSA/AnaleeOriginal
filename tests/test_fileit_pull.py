"""FileIt → Analee bank-statement tunnel (Festus 2026-09-28, tunnel 2).

Dark by default; workspace sessions only; the folder comes from the Practice
Club roster; a document is fetched only if it is in the client's own folder;
CSV goes through the upload service, PDF through the review screen.
"""
from unittest import mock

import pytest

from test_practice_send_tb_club import SECRET, _club_workspace_client

REF = "club-41-7"
CSV = b"Date,Description,Amount\n2026-03-01,Sale,150.00\n2026-03-02,Rent,-50.00\n"
PDF = b"%PDF-1.4 fake"


def _on(monkeypatch):
    monkeypatch.setenv("ANALEE_FILEIT_PULL_ENABLED", "True")
    monkeypatch.setenv("FILEIT_API_BASE_URL", "https://fileit.test")
    monkeypatch.setenv("FILEIT_API_TOKEN", "osk_test")
    monkeypatch.setenv("CLUB_PRACTICE_SYNC_URL", "https://hub.test/practice/sync/")
    monkeypatch.setenv("CLUB_PRACTICE_SYNC_TOKEN", "hub-token")


class _R:
    def __init__(self, status=200, json=None, content=b""):
        self.status_code, self._json, self.content = status, json or {}, content

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


DOCS = [
    {"id": "d-csv", "original_filename": "FNB March.csv", "category": "other",
     "created_at": "2026-04-01T00:00:00Z", "document_kind": "bank_statement"},
    {"id": "d-pdf", "original_filename": "Capitec Feb.pdf", "category": "pdf",
     "created_at": "2026-03-01T00:00:00Z", "document_kind": "general"},
    {"id": "d-doc", "original_filename": "lease.docx", "category": "word",
     "created_at": "2026-05-01T00:00:00Z", "document_kind": "general"},
]


def _wire(monkeypatch, content=b"", found=True):
    post = mock.Mock(return_value=_R(json={"found": found, "external_ref": "folder-9"}))

    def get(url, **kw):
        if url.endswith("/folders/folder-9/documents"):
            return _R(json={"items": DOCS, "total": 3})
        if url.endswith("/content"):
            return _R(content=content)
        raise AssertionError(url)
    monkeypatch.setattr("fileit_pull.requests.post", post)
    monkeypatch.setattr("fileit_pull.requests.get", mock.Mock(side_effect=get))
    return post


def _account_id(app):
    from models import Account, User
    from provisioning import WORKSPACE_EMAIL_DOMAIN
    with app.app_context():
        ws = User.query.filter_by(email=f"client+{REF}@{WORKSPACE_EMAIL_DOMAIN}").one()
        return Account.query.filter_by(user_id=ws.id, link="ca.810.001").one().id, ws.id


def test_dark_by_default_404_and_no_card(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    assert client.get("/fileit/statements").status_code == 404
    assert b"Choose from FileIt" not in client.get("/bank-statements/upload").data


def test_card_shows_in_a_workspace_when_on(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _on(monkeypatch)
    assert b"Choose from FileIt" in client.get("/bank-statements/upload").data
    assert b"Choose from FileIt" in client.get("/ocr/statement").data


def test_folder_is_resolved_with_the_workspace_ref(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _on(monkeypatch)
    post = _wire(monkeypatch)
    page = client.get("/fileit/statements")
    assert page.status_code == 200
    assert post.call_args.kwargs["json"] == {"product": "analee", "external_ref": REF,
                                             "target_product": "bookstorage"}
    assert post.call_args.args[0] == "https://hub.test/practice/resolve/"
    body = page.data
    assert b"FNB March.csv" in body and b"Capitec Feb.pdf" in body
    assert b"lease.docx" not in body
    assert body.index(b"FNB March.csv") < body.index(b"Capitec Feb.pdf")  # statements first


def test_not_linked_is_said_plainly(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _on(monkeypatch)
    _wire(monkeypatch, found=False)
    assert b"not linked to a FileIt folder" in client.get("/fileit/statements").data


def test_a_csv_is_imported_through_the_upload_service(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _on(monkeypatch)
    monkeypatch.setenv("ANALEE_AUTOPROCESS_ON_IMPORT", "0")
    _wire(monkeypatch, content=CSV)
    account_id, ws_id = _account_id(canary_app)
    r = client.post("/fileit/statements/import",
                    data={"document_id": "d-csv", "account_id": str(account_id)},
                    follow_redirects=True)
    assert b"Imported FNB March.csv from FileIt" in r.data
    from models import Transaction, UploadedFile
    with canary_app.app_context():
        assert Transaction.query.filter_by(user_id=ws_id).count() == 2
        assert UploadedFile.query.filter_by(user_id=ws_id).one().bank_account_id == account_id


def test_a_pdf_opens_the_review_screen(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _on(monkeypatch)
    _wire(monkeypatch, content=PDF)
    account_id, _ = _account_id(canary_app)
    outcome = mock.Mock(ok=True, rows=[{"date": "2026-02-01", "description": "Sale",
                                        "amount": 100.0, "confidence": 0.9}],
                        header=None, report_card=None, method="digital", error="")
    with mock.patch("ocr.routes.extract_bank_statement", return_value=outcome) as ex:
        r = client.post("/fileit/statements/import",
                        data={"document_id": "d-pdf", "account_id": str(account_id)})
    assert r.status_code == 200
    assert ex.call_args.args[0] == PDF
    assert b"Capitec Feb.pdf" in r.data


def test_a_document_outside_the_folder_is_refused(canary_app, monkeypatch):
    client = _club_workspace_client(canary_app, monkeypatch, approve=False)
    _on(monkeypatch)
    _wire(monkeypatch, content=CSV)
    account_id, ws_id = _account_id(canary_app)
    r = client.post("/fileit/statements/import",
                    data={"document_id": "someone-elses", "account_id": str(account_id)},
                    follow_redirects=True)
    assert b"not in this client" in r.data
    from models import Transaction
    with canary_app.app_context():
        assert Transaction.query.filter_by(user_id=ws_id).count() == 0


def test_outside_a_workspace_there_is_nothing_new(canary_app, monkeypatch):
    _on(monkeypatch)
    from models import User, db
    with canary_app.app_context():
        u = User(username="plain", email="plain@x.com", subscription_status="active")
        u.set_password("pw12345678")
        db.session.add(u)
        db.session.commit()
    client = canary_app.test_client()
    client.post("/auth/login", data={"email": "plain@x.com", "password": "pw12345678"})
    assert b"Choose from FileIt" not in client.get("/bank-statements/upload").data
    r = client.get("/fileit/statements")
    assert r.status_code == 302
