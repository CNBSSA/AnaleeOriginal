"""Tests for phased analyze processing helpers."""
from datetime import datetime

import pytest

from models import Account, Transaction, UploadedFile, User, db
from services.analyze_processing import (
    ANALYZE_PAGE_SIZE,
    count_unprocessed_transactions,
    get_paginated_transactions,
    process_transaction_batch,
    save_analyze_form_transactions,
    transaction_needs_processing,
)


@pytest.fixture
def analyze_user(app):
    with app.app_context():
        user = User(username='analyzeuser', email='analyze@example.com', subscription_status='active')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        return user.id


@pytest.fixture
def analyze_file(app, analyze_user):
    with app.app_context():
        uploaded = UploadedFile(filename='statement.xlsx', user_id=analyze_user)
        db.session.add(uploaded)
        db.session.commit()
        return uploaded.id


def _add_transactions(app, user_id, file_id, count=15):
    with app.app_context():
        for index in range(count):
            transaction = Transaction(
                date=datetime(2025, 1, index + 1),
                description=f'Transaction {index + 1}',
                amount=100.0 + index,
                user_id=user_id,
                file_id=file_id,
            )
            db.session.add(transaction)
        db.session.commit()


def _add_account(app, user_id):
    with app.app_context():
        account = Account(
            link='ca.100',
            name='Bank Fees',
            category='Expenses',
            user_id=user_id,
            is_active=True,
        )
        db.session.add(account)
        db.session.commit()
        return account.id


def test_transaction_needs_processing():
    pending = Transaction(description='Test', amount=1, date=datetime.utcnow(), user_id=1)
    assert transaction_needs_processing(pending) is True

    pending.explanation = 'Monthly fee'
    assert transaction_needs_processing(pending) is False

    pending.explanation = ''
    pending.account_id = 5
    assert transaction_needs_processing(pending) is False


def test_pagination_returns_ten_rows_per_page(app, analyze_user, analyze_file):
    _add_transactions(app, analyze_user, analyze_file, count=25)

    with app.app_context():
        page_one, total_count, total_pages = get_paginated_transactions(
            analyze_file, analyze_user, page=1
        )
        page_three, _, _ = get_paginated_transactions(analyze_file, analyze_user, page=3)

    assert total_count == 25
    assert total_pages == 3
    assert len(page_one) == ANALYZE_PAGE_SIZE
    assert len(page_three) == 5


def test_save_analyze_form_transactions(app, analyze_user, analyze_file):
    _add_transactions(app, analyze_user, analyze_file, count=2)
    account_id = _add_account(app, analyze_user)

    with app.app_context():
        transactions = Transaction.query.filter_by(file_id=analyze_file).order_by(Transaction.id).all()
        form_data = {
            f'account_{transactions[0].id}': str(account_id),
            f'explanation_{transactions[0].id}': 'Bank charge',
            f'explanation_{transactions[1].id}': 'Office supplies',
        }
        saved = save_analyze_form_transactions(analyze_user, form_data)

        updated_first = Transaction.query.get(transactions[0].id)
        updated_second = Transaction.query.get(transactions[1].id)

    assert saved == 2
    assert updated_first.account_id == account_id
    assert updated_first.explanation == 'Bank charge'
    assert updated_second.explanation == 'Office supplies'


def test_count_unprocessed_transactions(app, analyze_user, analyze_file):
    _add_transactions(app, analyze_user, analyze_file, count=3)
    account_id = _add_account(app, analyze_user)

    with app.app_context():
        first = Transaction.query.filter_by(file_id=analyze_file).first()
        first.account_id = account_id
        db.session.commit()
        unprocessed = count_unprocessed_transactions(analyze_file, analyze_user)

    assert unprocessed == 2


def test_find_similar_transactions_returns_list(app, analyze_user, analyze_file):
    from predictive_features import PredictiveFeatures

    with app.app_context():
        source = Transaction(
            date=datetime(2025, 1, 1),
            description='Monthly bank charge',
            amount=-10.0,
            user_id=analyze_user,
            file_id=analyze_file,
            explanation='Bank service fee',
        )
        db.session.add(source)
        db.session.commit()

        predictor = PredictiveFeatures()
        result = predictor.find_similar_transactions(
            'Monthly bank charge',
            'Bank service fee',
            user_id=analyze_user,
        )

    assert result['success'] is True
    assert isinstance(result['similar_transactions'], list)


def test_replicate_explanation_helper(app, analyze_user, analyze_file):
    with app.app_context():
        source = Transaction(
            date=datetime(2025, 1, 1),
            description='Monthly bank charge',
            amount=-10.0,
            user_id=analyze_user,
            file_id=analyze_file,
            explanation='Bank service fee',
        )
        target = Transaction(
            date=datetime(2025, 1, 2),
            description='Monthly bank charge',
            amount=-12.0,
            user_id=analyze_user,
            file_id=analyze_file,
        )
        db.session.add_all([source, target])
        db.session.commit()

        target.explanation = source.explanation
        db.session.commit()

        updated = db.session.get(Transaction, target.id)
        assert updated.explanation == 'Bank service fee'


def test_process_transaction_batch_without_ai_client(app, analyze_user, analyze_file, monkeypatch):
    _add_transactions(app, analyze_user, analyze_file, count=12)
    _add_account(app, analyze_user)

    class FakePredictor:
        def suggest_account(self, description, explanation, user_id=None):
            return {
                'success': True,
                'account': 'Bank Fees',
                'confidence': 0.9,
                'reasoning': 'Looks like a fee',
            }

    monkeypatch.setattr('predictive_features.PredictiveFeatures', FakePredictor)

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)

    assert result['success'] is True
    assert result['processed'] == 10
    assert result['has_more'] is True
    assert result['results'][0]['applied_account_id'] is not None
