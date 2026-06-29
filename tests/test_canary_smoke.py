"""
Canary smoke suite — walks a brand-new user through the whole journey against the
real application (Flask test client, file-backed SQLite), so a broken deploy is
caught before users are: register -> login -> company settings -> import a file ->
verify the transaction count -> render every report -> exercise an AI feature
(deterministic offline fallback).

Designed for CI (every deploy + nightly). Boots create_app via the canary_app
fixture in conftest.py.
"""
import io
from datetime import datetime

import pytest

openpyxl = pytest.importorskip("openpyxl")
from openpyxl import Workbook  # noqa: E402

EMAIL = "canary@example.com"
PASSWORD = "Sup3rSecret!"


def _xlsx_bytes(rows):
    wb = Workbook()
    ws = wb.active
    ws.append(["Date", "Description", "Amount"])
    for date_str, description, amount in rows:
        ws.append([date_str, description, amount])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_canary_full_journey(canary_app):
    app = canary_app
    client = app.test_client()
    from models import db, User, Account, Transaction

    # 1. Register a brand-new user.
    resp = client.post("/auth/register", data={
        "username": "canary",
        "email": EMAIL,
        "password": PASSWORD,
        "confirm_password": PASSWORD,
    })
    assert resp.status_code in (301, 302), "registration should redirect"
    with app.app_context():
        user = User.query.filter_by(email=EMAIL).first()
        assert user is not None, "user was not created"
        user_id = user.id

    # 2. Log in.
    resp = client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})
    assert resp.status_code in (301, 302)
    assert "/login" not in resp.headers.get("Location", ""), "login bounced back to login"

    # 3. Configure company settings (Dec year-end -> calendar-year financial year).
    # Entity type is now a required field (entity-chart feature); pick a seeded one.
    from models import Entity
    with app.app_context():
        entity = Entity.query.order_by(Entity.name).first()
        assert entity is not None, "no entity types seeded on boot"
        entity_id = entity.id
    resp = client.post("/company-settings", data={
        "company_name": "Canary Co",
        "registration_number": "R1",
        "tax_number": "T1",
        "vat_number": "V1",
        "address": "1 Test Street",
        "financial_year_end": "12",
        "entity_id": str(entity_id),
    }, follow_redirects=True)
    assert resp.status_code == 200

    # 4. Get a bank account to import into. Selecting an entity type provisions
    # a chart of accounts, so prefer an existing bank account (link ca.810*);
    # only create one if none was provisioned.
    with app.app_context():
        account = (Account.query
                   .filter(Account.user_id == user_id,
                           Account.link.like('ca.810%'),
                           Account.is_active.is_(True))
                   .order_by(Account.link)
                   .first())
        if account is None:
            account = Account(name="Bank", link="ca.810.001", category="Assets",
                              user_id=user_id, is_active=True)
            db.session.add(account)
            db.session.commit()
        account_id = account.id

    # 5. Import a fixture spreadsheet (3 rows, dated today -> current FY).
    today = datetime.utcnow().strftime("%Y-%m-%d")
    xlsx = _xlsx_bytes([
        (today, "Coffee Shop", 4.50),
        (today, "Lunch", 12.30),
        (today, "Taxi", 20.00),
    ])
    # /upload is now a legacy redirect to the consolidated bank-statement
    # importer, so exercise the real endpoint users hit.
    resp = client.post("/bank-statements/upload", data={
        "account": str(account_id),
        "file": (io.BytesIO(xlsx), "canary.xlsx"),
    }, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200, "upload did not complete"

    with app.app_context():
        count = Transaction.query.filter_by(user_id=user_id).count()
        assert count == 3, f"expected 3 imported transactions, got {count}"

    # 6. Every report renders without a server error.
    for path in ["/trial-balance", "/general-ledger", "/cashbook",
                 "/income-statement", "/financial-position"]:
        report = client.get(path)
        assert report.status_code == 200, f"{path} returned {report.status_code}"

    # 7. The trial balance reflects the imported account.
    tb = client.get("/trial-balance")
    assert b"Bank" in tb.data, "imported account missing from trial balance"

    # 8. An AI feature degrades gracefully with no API key (must not 500).
    ai = client.post("/analyze/suggest-explanation", json={"description": "Coffee Shop"})
    assert ai.status_code == 200, f"AI endpoint returned {ai.status_code} (should fall back)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
