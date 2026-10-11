"""QA D2/D3 — Analyze screen UI must match persisted batch state (TEST data only)."""
import os
import re
from datetime import datetime

import pytest

from models import Account, Transaction, UploadedFile, User, db

pytest.importorskip('flask_sqlalchemy')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def analyze_user(app):
    with app.app_context():
        user = User(username='d2d3user', email='d2d3@example.com', subscription_status='active')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        return user.id


@pytest.fixture
def analyze_file(app, analyze_user):
    with app.app_context():
        uploaded = UploadedFile(filename='TEST-d2d3.xlsx', user_id=analyze_user)
        db.session.add(uploaded)
        db.session.commit()
        return uploaded.id


def _read(rel):
    with open(os.path.join(HERE, rel), encoding='utf-8') as fh:
        return fh.read()


def test_d3_suggest_account_button_label_after_request():
    """After ASF fetch, the row button must read 'Suggest account', not internal jargon."""
    js = _read('static/js/analyze/accountSuggestions.js')
    match = re.search(
        r'resetButtonState\(button\)\s*\{([^}]+)\}',
        js,
        re.DOTALL,
    )
    assert match, 'resetButtonState not found'
    reset_block = match.group(1)
    assert 'Suggest account' in reset_block
    assert re.search(r'innerHTML\s*=\s*[^;]*\bASF\b', reset_block) is None


def test_d2_batch_ui_only_applies_persisted_accounts():
    """Batch JS must not pre-fill the dropdown from low-confidence suggestions."""
    js = _read('static/js/analyze/batchProcessor.js')
    match = re.search(
        r'applyBatchResults\(results\)\s*\{([\s\S]*?)\n    \}',
        js,
    )
    assert match, 'applyBatchResults not found'
    apply_fn = match.group(1)
    assert 'applied_account_id' in apply_fn
    assert 'suggestion.account' not in apply_fn
    assert 'border-info' not in apply_fn


def test_d2_api_low_confidence_suggestion_without_applied_account(
    app, analyze_user, analyze_file, monkeypatch,
):
    """Server returns suggestion metadata but does not persist sub-threshold accounts."""
    from services.analyze_processing import process_transaction_batch
    from services.bulk_suggestions import RowSuggestion

    with app.app_context():
        txn = Transaction(
            date=datetime(2025, 1, 1),
            description='TEST M-CHOICE',
            amount=-50.0,
            user_id=analyze_user,
            file_id=analyze_file,
        )
        db.session.add(txn)
        account = Account(
            link='e.400.000',
            name='TEST Supplies',
            category='Expenses',
            user_id=analyze_user,
            is_active=True,
        )
        db.session.add(account)
        db.session.commit()
        account_id = account.id

    def _low_confidence(rows, accounts, client=None, **kwargs):
        return {
            row['index']: RowSuggestion(
                index=row['index'],
                account_id=account_id,
                account_name='TEST Supplies',
                confidence=0.10,
                explanation='TEST low confidence',
            )
            for row in rows
        }

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _low_confidence)

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)
        row = result['results'][0]
        persisted = db.session.get(Transaction, row['transaction_id'])

    assert row['suggestion']['success'] is True
    assert row['suggestion']['account'] == 'TEST Supplies'
    assert row['applied_account_id'] is None
    assert persisted.account_id is None
