"""Unit 4.1: /icountant duplicate transaction-processing path removed.

Transaction work belongs on Analyze Data; conversation belongs on AI Assistant Chat.
TEST data only.
"""
import os
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from test_asf_degradation_guard import EMAIL, _register_and_login

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _templates_with_icountant_hrefs():
    hits = []
    for root, _dirs, files in os.walk(os.path.join(HERE, "templates")):
        for name in files:
            if not name.endswith(".html"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if "icountant" in text.lower():
                hits.append(os.path.relpath(path, HERE))
    return hits


def test_icountant_url_redirects_to_analyze_list(canary_app):
    client = canary_app.test_client()
    _register_and_login(client)
    resp = client.get("/icountant", follow_redirects=False)
    assert resp.status_code in (301, 302, 303, 307, 308)
    assert "/analyze" in resp.headers["Location"]


def test_icountant_post_also_redirects(canary_app):
    client = canary_app.test_client()
    _register_and_login(client)
    resp = client.post("/icountant", data={}, follow_redirects=False)
    assert resp.status_code in (301, 302, 303, 307, 308)
    assert "/analyze" in resp.headers["Location"]


def test_icountant_insights_api_is_gone(canary_app):
    from models import Transaction, User, db

    client = canary_app.test_client()
    _register_and_login(client)
    with canary_app.app_context():
        uid = User.query.filter_by(email=EMAIL).one().id
        txn = Transaction(
            date=datetime(2026, 3, 1),
            description="TEST",
            amount=-1.0,
            user_id=uid,
        )
        db.session.add(txn)
        db.session.commit()
        txn_id = txn.id
    resp = client.get(f"/api/icountant/{txn_id}/insights")
    assert resp.status_code == 404


def test_no_template_links_to_icountant():
    assert _templates_with_icountant_hrefs() == []


def test_nav_has_no_ask_analee_duplicate_path():
    base = open(os.path.join(HERE, "templates/base.html"), encoding="utf-8").read()
    assert "icountant" not in base.lower()
    assert base.count("AI Assistant Chat") == 1


def test_analyze_and_chat_still_reachable(canary_app):
    client = canary_app.test_client()
    _register_and_login(client)
    assert client.get("/analyze").status_code == 200
    assert client.get("/chat/interface").status_code == 200


def test_analyze_still_shows_a_line_still_on_the_bank(canary_app):
    """Bank-stamped rows remain visible on Analyze (replaces the old /icountant queue)."""
    from models import Account, Transaction, UploadedFile, User, db

    client = canary_app.test_client()
    _register_and_login(client)
    with canary_app.app_context():
        u = User.query.filter_by(email=EMAIL).one()
        bank = Account(
            link="ca.810.001",
            name="TEST Bank Cheque",
            category="Assets",
            user_id=u.id,
        )
        db.session.add(bank)
        db.session.flush()
        f = UploadedFile(filename="TEST.csv", user_id=u.id, upload_date=datetime.utcnow())
        db.session.add(f)
        db.session.flush()
        db.session.add(
            Transaction(
                date=datetime(2026, 3, 1),
                description="TEST still on the bank",
                amount=-50.0,
                user_id=u.id,
                account_id=bank.id,
                file_id=f.id,
            )
        )
        db.session.commit()
        file_id = f.id
    page = client.get(f"/analyze/{file_id}").get_data(as_text=True)
    assert "TEST still on the bank" in page
