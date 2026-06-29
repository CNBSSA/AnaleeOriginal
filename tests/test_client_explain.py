"""Client no-login ERF wizard tests."""
from datetime import datetime

import pytest

from client_explain_tokens import create_client_explain_token, verify_client_explain_token
from models import Transaction, UploadedFile, User, db
from services.client_explanation import SOURCE_ACCOUNTANT, SOURCE_CLIENT, save_explanation
from services.client_erf import find_similar_unexplained


@pytest.fixture
def owner(app):
    with app.app_context():
        user = User(username='owner1', email='owner1@example.com', subscription_status='active')
        user.set_password('secret')
        db.session.add(user)
        db.session.commit()
        return user.id


@pytest.fixture
def bank_file(app, owner):
    with app.app_context():
        uploaded = UploadedFile(
            filename='march-fnb.csv',
            user_id=owner,
            upload_date=datetime.utcnow(),
        )
        db.session.add(uploaded)
        db.session.flush()
        for desc, amount in (
            ('Builder Supply Co', -500.0),
            ('Builder Supply Co materials', -520.0),
            ('Client deposit', 1000.0),
        ):
            db.session.add(Transaction(
                date=datetime(2026, 3, 15),
                description=desc,
                amount=amount,
                user_id=owner,
                file_id=uploaded.id,
                explanation='',
                explanation_source='',
            ))
        db.session.commit()
        return uploaded.id


def test_token_round_trip():
    token = create_client_explain_token(7, 3, secret_key='test-secret-key-analee-erf')
    assert verify_client_explain_token(token, secret_key='test-secret-key-analee-erf') == (7, 3)


def test_accountant_cannot_overwrite_client(app, bank_file, owner):
    with app.app_context():
        txn = Transaction.query.filter_by(file_id=bank_file).first()
        save_explanation(txn, 'Client knows best', SOURCE_CLIENT)
        db.session.commit()
        ok, reason = save_explanation(txn, 'Accountant guess', SOURCE_ACCOUNTANT)
        assert not ok
        assert reason == 'client_locked'


def test_erf_finds_similar(app, bank_file, owner):
    with app.app_context():
        first = Transaction.query.filter_by(file_id=bank_file).order_by(Transaction.id).first()
        similar = find_similar_unexplained(
            bank_file, owner, 'Builder Supply Co', exclude_id=first.id,
        )
        assert len(similar) >= 1
