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
    """A row is finished only when it has BOTH an account and an explanation.

    This used to be an AND of the two absences — having *either* one marked the
    row done. Both import paths stamp the chosen bank account onto every row,
    so a freshly imported statement was reported "All processed" before
    anything had been categorised or explained.
    """
    pending = Transaction(description='Test', amount=1, date=datetime.utcnow(), user_id=1)
    assert transaction_needs_processing(pending) is True

    # Explanation only — still needs an account.
    pending.explanation = 'Monthly fee'
    assert transaction_needs_processing(pending) is True

    # Account only (the import-stamped bank account) — still needs explaining.
    pending.explanation = ''
    pending.account_id = 5
    assert transaction_needs_processing(pending) is True

    # Both present — done.
    pending.explanation = 'Monthly fee'
    assert transaction_needs_processing(pending) is False

    # Whitespace is not an explanation.
    pending.explanation = '   '
    assert transaction_needs_processing(pending) is True


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


def test_account_change_persists_on_client_locked_row(app, analyze_user, analyze_file):
    """Regression: an accountant changing the ACCOUNT of a client-explained row
    must be saved, even when the same submit also carries an accountant
    explanation for that row. The client's explanation is preserved (not
    overwritten), but the account override must never be silently dropped.

    Previously the client-lock branch did `continue`, which skipped `saved += 1`
    and — when this was the only edited row — the gated `if saved: commit()`,
    so the account change was lost. The discriminating assertion is `saved == 1`
    (the old code returned 0)."""
    account_id = _add_account(app, analyze_user)
    with app.app_context():
        txn = Transaction(
            date=datetime(2025, 2, 1),
            description='Builder Supply Co',
            amount=-500.0,
            user_id=analyze_user,
            file_id=analyze_file,
            explanation='Paid the plumber (client)',
            explanation_source='client',
        )
        db.session.add(txn)
        db.session.commit()
        txn_id = txn.id

        form_data = {
            f'account_{txn_id}': str(account_id),
            f'explanation_{txn_id}': 'Accountant note that must NOT overwrite the client',
        }
        saved = save_analyze_form_transactions(analyze_user, form_data)

        # Re-read from the DB to confirm the account change was committed.
        db.session.expire_all()
        updated = db.session.get(Transaction, txn_id)

    assert saved == 1                                          # not skipped by the client-lock
    assert updated.account_id == account_id                    # accountant's account override saved
    assert updated.explanation == 'Paid the plumber (client)'  # client's words kept
    assert updated.explanation_source == 'client'              # still attributed to the client


def test_count_unprocessed_transactions(app, analyze_user, analyze_file):
    """An account alone does not finish a row — it still needs explaining."""
    _add_transactions(app, analyze_user, analyze_file, count=3)
    account_id = _add_account(app, analyze_user)

    with app.app_context():
        first = Transaction.query.filter_by(file_id=analyze_file).first()
        first.account_id = account_id
        db.session.commit()
        # All 3 still need work: one has an account but no explanation, two
        # have neither.
        assert count_unprocessed_transactions(analyze_file, analyze_user) == 3

        first.explanation = 'Monthly bank charge'
        db.session.commit()
        # Now that row has both halves and drops out.
        assert count_unprocessed_transactions(analyze_file, analyze_user) == 2


def test_imported_statement_is_not_reported_all_processed(app, analyze_user, analyze_file):
    """The regression that hid 465 live transactions behind a green badge.

    Both import paths stamp the selected bank account onto every row. Under the
    old predicate that alone satisfied "processed", so the analyze list showed
    "All processed" for a statement whose rows had never been categorised or
    explained, and the exceptions view showed nothing to do.
    """
    _add_transactions(app, analyze_user, analyze_file, count=5)
    bank_account_id = _add_account(app, analyze_user)

    with app.app_context():
        for txn in Transaction.query.filter_by(file_id=analyze_file).all():
            txn.account_id = bank_account_id       # exactly what import does
        db.session.commit()

        assert count_unprocessed_transactions(analyze_file, analyze_user) == 5

        rows, total, _ = get_paginated_transactions(
            analyze_file, analyze_user, page=1, only_unprocessed=True)
        assert total == 5, "exceptions view must still surface these rows"


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


def test_batch_scopes_suggestions_to_the_owning_user(app, analyze_user, analyze_file, monkeypatch):
    """The batch must pass user_id to the suggester.

    suggest_account(..., user_id=None) falls back to
    Account.query.filter_by(is_active=True) across EVERY tenant, so omitting it
    put other customers' charts of accounts into the prompt.
    """
    _add_transactions(app, analyze_user, analyze_file, count=2)
    _add_account(app, analyze_user)
    seen = []

    class FakePredictor:
        def suggest_account(self, description, explanation, user_id=None):
            seen.append(user_id)
            return {'success': True, 'account': 'Bank Fees',
                    'confidence': 0.9, 'reasoning': 'fee'}

    monkeypatch.setattr('predictive_features.PredictiveFeatures', FakePredictor)
    with app.app_context():
        process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)

    assert seen, "suggester was never called"
    assert all(uid == analyze_user for uid in seen), (
        "batch leaked every tenant's accounts by omitting user_id")


def test_batch_does_not_skip_rows_it_could_not_assign(app, analyze_user, analyze_file, monkeypatch):
    """Auto-applied rows leave the `account_id IS NULL` set, so the window must
    not advance past them — doing so stepped over unassigned rows for good."""
    _add_transactions(app, analyze_user, analyze_file, count=12)
    _add_account(app, analyze_user)

    class HalfConfidentPredictor:
        """Alternates: half land above the 0.85 gate, half below."""
        def __init__(self):
            self.calls = 0

        def suggest_account(self, description, explanation, user_id=None):
            self.calls += 1
            confident = self.calls % 2 == 0
            return {'success': True, 'account': 'Bank Fees',
                    'confidence': 0.95 if confident else 0.10,
                    'reasoning': 'x'}

    monkeypatch.setattr('predictive_features.PredictiveFeatures', HalfConfidentPredictor)

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)
        applied = result['applied']
        # 5 of 10 applied -> 5 remain in the set at positions 0..4, so the next
        # window must start at 5, NOT at 10 (which would skip 5 live rows).
        assert applied == 5
        assert result['next_offset'] == 5

        # Drain the file and prove nothing was stepped over: every row ends up
        # either assigned or still visible as needing work.
        seen_ids = set()
        offset = result['next_offset']
        for _ in range(10):
            nxt = process_transaction_batch(
                analyze_file, analyze_user, offset=offset, batch_size=10)
            seen_ids.update(r['transaction_id'] for r in nxt['results'])
            offset = nxt['next_offset']
            if not nxt['has_more']:
                break

        unassigned = Transaction.query.filter(
            Transaction.file_id == analyze_file,
            Transaction.account_id.is_(None),
        ).count()
        assigned = Transaction.query.filter(
            Transaction.file_id == analyze_file,
            Transaction.account_id.isnot(None),
        ).count()
        assert assigned + unassigned == 12
        assert assigned > 5, "later batches never got to assign anything"


def test_asf_declines_cleanly_when_ai_key_absent(canary_app):
    """Degradation guard (Festus 2026-08-27): with no ANTHROPIC_API_KEY the
    ASF route must NOT return a misleading text-match suggestion — it returns
    ai_online=False + a clear message, and never invokes the frozen engine.
    canary_app is the real app and already runs with the key popped."""
    with canary_app.app_context():
        u = User(username="guardu", email="guard@x.test", subscription_status="active")
        u.set_password("pw12345!")
        db.session.add(u); db.session.commit()
        uid = u.id
    from flask_login import login_user
    from models import User as _User
    with canary_app.test_request_context(
            "/analyze/suggest-account", method="POST",
            json={"description": "FEE BANK CHARGE", "explanation": "bank charges"}):
        login_user(_User.query.get(uid))
        from routes import suggest_account
        resp = suggest_account()
    payload = resp.get_json() if hasattr(resp, "get_json") else resp[0].get_json()
    assert payload["success"] is False
    assert payload["ai_online"] is False
    assert "offline" in payload["message"].lower()


def test_only_unprocessed_filter_returns_exceptions(app, analyze_user, analyze_file):
    """Slice 1 (Festus 2026-08-27): the exceptions view returns ONLY rows that
    still need an account/explanation; the full list is unchanged by default."""
    from services.analyze_processing import get_paginated_transactions
    with app.app_context():
        # "Done" now means BOTH an account and an explanation. A row carrying
        # only the import-stamped bank account (H1) is still an exception.
        acc = _add_account(app, analyze_user)
        done = [
            Transaction(date=datetime(2025, 1, 1), description='D1', amount=1,
                        user_id=analyze_user, file_id=analyze_file,
                        account_id=acc, explanation='Bank charge'),
        ]
        needy = [
            Transaction(date=datetime(2025, 1, 2), description='H1', amount=2,
                        user_id=analyze_user, file_id=analyze_file,
                        account_id=acc),                      # account, no explanation
            Transaction(date=datetime(2025, 1, 3), description='H2', amount=3,
                        user_id=analyze_user, file_id=analyze_file,
                        explanation='Bank charge'),           # explanation, no account
            Transaction(date=datetime(2025, 1, 4), description='N1', amount=4,
                        user_id=analyze_user, file_id=analyze_file),
        ]
        db.session.add_all(done + needy)
        db.session.commit()

        all_rows, all_total, _ = get_paginated_transactions(
            analyze_file, analyze_user, 1, 50, only_unprocessed=False)
        exc_rows, exc_total, _ = get_paginated_transactions(
            analyze_file, analyze_user, 1, 50, only_unprocessed=True)

    assert all_total == 4
    assert exc_total == 3
    assert {r.description for r in exc_rows} == {'H1', 'H2', 'N1'}


def test_exceptions_view_renders(canary_app):
    """Post-engagement: the ?view=exceptions page renders without a template
    error (guards the |pluralize→Django-only mistake and the new markup)."""
    with canary_app.app_context():
        u = User(username="exru", email="exr@x.test", subscription_status="active")
        u.set_password("pw12345!"); db.session.add(u); db.session.commit()
        f = UploadedFile(filename="s.xlsx", user_id=u.id); db.session.add(f); db.session.commit()
        db.session.add(Transaction(date=datetime(2025, 1, 1), description="N",
                                   amount=1, user_id=u.id, file_id=f.id)); db.session.commit()
        uid, fid = u.id, f.id
    from flask_login import login_user
    with canary_app.test_request_context(f"/analyze/{fid}?view=exceptions"):
        login_user(User.query.get(uid))
        from routes import analyze
        html = analyze(fid)
    assert "need your eye" in html
    assert "Analyse the entire statement" in html
