"""
Multi-tenant isolation test (Analee tenancy == user_id).

The Django-flavoured spec asks: "authenticated as Company B, viewing Company A's
asset must return an object error / 404." Analee is Flask + SQLAlchemy and scopes
every owned entity by user_id, so the equivalent is: logged in as user B, every
attempt to read or mutate user A's account / file / transaction / goal must be
refused (404 or redirect) AND must leave A's data unchanged.

Boots the real app via the canary_app fixture (conftest.py).
"""
import io

import pytest

openpyxl = pytest.importorskip("openpyxl")


def _seed_two_tenants(app):
    """Create users A and B; give A an account, file, transaction and goal."""
    from models import db, User, Account, UploadedFile, Transaction, FinancialGoal
    from datetime import datetime

    with app.app_context():
        a = User(username="tenant_a", email="a@example.com")
        a.set_password("pw-a-123456")
        b = User(username="tenant_b", email="b@example.com")
        b.set_password("pw-b-123456")
        db.session.add_all([a, b])
        db.session.flush()

        account = Account(name="A Bank", link="ca.810.001", category="Assets",
                          user_id=a.id, is_active=True)
        uploaded = UploadedFile(filename="a.xlsx", user_id=a.id, upload_date=datetime.utcnow())
        db.session.add_all([account, uploaded])
        db.session.flush()

        txn = Transaction(date=datetime(2026, 3, 1), description="A SECRET TXN",
                          amount=100.0, user_id=a.id, file_id=uploaded.id, account_id=account.id)
        goal = FinancialGoal(user_id=a.id, name="A Goal", target_amount=1000.0,
                             current_amount=10.0, status="in_progress")
        db.session.add_all([txn, goal])
        db.session.commit()

        return {
            "a_id": a.id, "b_id": b.id,
            "account_id": account.id, "file_id": uploaded.id,
            "txn_id": txn.id, "goal_id": goal.id,
        }


def _login(client, user_id):
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user_id)


def test_tenant_b_cannot_access_tenant_a_data(canary_app):
    app = canary_app
    ids = _seed_two_tenants(app)
    client = app.test_client()
    _login(client, ids["b_id"])  # authenticated as B

    leaks = []

    # 1. View A's uploaded file / its transactions.
    resp = client.get(f"/analyze/{ids['file_id']}")
    if resp.status_code == 200 and b"A SECRET TXN" in resp.data:
        leaks.append("GET /analyze/<A file> leaked A's transaction data")

    # 2. Edit A's account.
    client.post(f"/account/{ids['account_id']}/edit",
                data={"link": "HACKED", "name": "HACKED", "category": "x", "sub_category": ""})

    # 3. Delete A's account.
    client.post(f"/account/{ids['account_id']}/delete")

    # 4. Delete A's uploaded file.
    resp = client.post(f"/file/{ids['file_id']}/delete")

    # 5. Reassign/relabel A's transaction.
    client.post(f"/analyze/save-transaction/{ids['txn_id']}",
                json={"account_id": 999999, "explanation": "HACKED"})

    # 6. Tamper with A's goal.
    client.post(f"/goals/{ids['goal_id']}/update", data={"current_amount": "999999"})

    # Re-assert A's data is intact and unmodified.
    from models import db, Account, UploadedFile, Transaction, FinancialGoal
    with app.app_context():
        account = db.session.get(Account, ids["account_id"])
        uploaded = db.session.get(UploadedFile, ids["file_id"])
        txn = db.session.get(Transaction, ids["txn_id"])
        goal = db.session.get(FinancialGoal, ids["goal_id"])

        if account is None:
            leaks.append("A's account was DELETED by tenant B")
        else:
            if account.name == "HACKED":
                leaks.append("A's account was EDITED by tenant B")
        if uploaded is None:
            leaks.append("A's uploaded file was DELETED by tenant B")
        if txn is None:
            leaks.append("A's transaction was DELETED by tenant B")
        elif txn.explanation == "HACKED" or txn.account_id == 999999:
            leaks.append("A's transaction was MODIFIED by tenant B")
        if goal is not None and goal.current_amount == 999999:
            leaks.append("A's goal was MODIFIED by tenant B")

    assert not leaks, "Multi-tenant isolation FAILED:\n- " + "\n- ".join(leaks)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
