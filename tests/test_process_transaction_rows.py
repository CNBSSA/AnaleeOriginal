"""
Regression test for routes.process_transaction_rows.

The function uses pd.to_datetime but routes.py imported pandas only locally inside
process_uploaded_file, so process_transaction_rows raised
"name 'pd' is not defined" for every row (0 imported). pandas is now imported at
module scope. This test imports the full routes module (heavy deps) and is skipped
if any of those deps are unavailable in the environment.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask_sqlalchemy")
# routes pulls in the wider app stack; skip cleanly if it can't be imported here.
routes = pytest.importorskip("routes", reason="routes module deps unavailable")

from flask import Flask
from models import db, User, UploadedFile, Transaction


def _app():
    app = Flask(__name__)
    app.config.update(SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
                      SQLALCHEMY_TRACK_MODIFICATIONS=False)
    db.init_app(app)
    return app


def test_process_transaction_rows_persists_rows():
    import pandas as pd
    from datetime import datetime

    app = _app()
    with app.app_context():
        db.create_all()
        user = User(username="t", email="t@e.com", password_hash="x")
        db.session.add(user)
        db.session.commit()
        uf = UploadedFile(filename="x.xlsx", user_id=user.id, upload_date=datetime.utcnow())
        db.session.add(uf)
        db.session.commit()

        df = pd.DataFrame([
            {"Date": datetime(2026, 3, 1), "Description": "Coffee", "Amount": 4.5},
            {"Date": datetime(2026, 3, 2), "Description": "Lunch", "Amount": 12.3},
        ])
        processed, errors = routes.process_transaction_rows(df, uf, user)
        assert processed == 2, f"expected 2 processed, got {processed} (errors={errors})"
        assert errors == []
        assert Transaction.query.filter_by(file_id=uf.id).count() == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
