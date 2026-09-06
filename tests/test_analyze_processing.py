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


def test_batch_costs_one_ai_call_for_the_whole_batch(app, analyze_user, analyze_file, monkeypatch):
    """The Tier 1 point: N rows cost ONE call, not N.

    The old path called Claude once per transaction, each prompt carrying the
    user's entire chart of accounts.
    """
    _add_transactions(app, analyze_user, analyze_file, count=12)
    account_id = _add_account(app, analyze_user)
    calls = []

    from services.bulk_suggestions import RowSuggestion

    def _suggest(rows, accounts, client=None):
        calls.append(len(rows))
        return {r['index']: RowSuggestion(
            index=r['index'], account_id=account_id, account_name='Bank Fees',
            confidence=0.9, explanation='Monthly bank charge') for r in rows}

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _suggest)

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)

    assert result['success'] is True
    assert result['processed'] == 10
    assert result['has_more'] is True
    assert result['results'][0]['applied_account_id'] is not None
    assert calls == [10], f"expected one batched call for 10 rows, got {calls}"


def test_batch_sends_only_the_owning_users_accounts(app, analyze_user, analyze_file, monkeypatch):
    """The chart put in the prompt must be scoped to this user.

    The per-row suggester defaulted user_id to None and then queried
    Account.query.filter_by(is_active=True) across EVERY tenant. The batched
    path passes an explicitly scoped list instead — this pins that.
    """
    _add_transactions(app, analyze_user, analyze_file, count=2)
    _add_account(app, analyze_user)

    with app.app_context():
        other = User(username='other', email='other@x.test', subscription_status='active')
        other.set_password('pw12345!')
        db.session.add(other)
        db.session.commit()
        db.session.add(Account(link='e.999.000', name='OTHER TENANT SECRET ACCOUNT',
                               category='Expenses', user_id=other.id))
        db.session.commit()

    seen_charts = []

    def _suggest(rows, accounts, client=None):
        seen_charts.append([a.name for a in accounts])
        return {}

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _suggest)
    with app.app_context():
        process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)

    assert seen_charts, "suggester was never called"
    for chart in seen_charts:
        assert 'OTHER TENANT SECRET ACCOUNT' not in chart, (
            "another tenant's accounts reached the prompt")


def test_batch_does_not_skip_rows_it_could_not_complete(app, analyze_user, analyze_file, monkeypatch):
    """Completed rows leave the result set, so the window must advance only
    past the rows that remain — otherwise live rows are stepped over for good."""
    _add_transactions(app, analyze_user, analyze_file, count=12)
    account_id = _add_account(app, analyze_user)

    from services.bulk_suggestions import RowSuggestion

    def _half_confident(rows, accounts, client=None):
        out = {}
        for position, row in enumerate(rows):
            confident = position % 2 == 1
            out[row['index']] = RowSuggestion(
                index=row['index'],
                account_id=account_id,
                account_name='Bank Fees',
                confidence=0.95 if confident else 0.10,
                explanation='Reviewed automatically',
            )
        return out

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _half_confident)

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0, batch_size=10)
        # 5 of 10 got an account (and all 10 got an explanation), so 5 rows are
        # now complete and drop out; the next window starts at 5, not 10.
        assert result['applied'] == 5
        assert result['next_offset'] == 5

        # Drain the file; nothing may be stepped over.
        offset = result['next_offset']
        for _ in range(10):
            nxt = process_transaction_batch(
                analyze_file, analyze_user, offset=offset, batch_size=10)
            offset = nxt['next_offset']
            if not nxt['has_more']:
                break

        assigned = Transaction.query.filter(
            Transaction.file_id == analyze_file,
            Transaction.account_id.isnot(None),
        ).count()
        assert assigned > 5, "later batches never got to assign anything"


def _fake_bulk(mapping):
    """Patch the batched suggester with a canned index -> RowSuggestion map."""
    from services.bulk_suggestions import RowSuggestion

    def _suggest(rows, accounts, client=None):
        out = {}
        for row in rows:
            spec = mapping.get(row['index'])
            if spec is None:
                continue
            out[row['index']] = RowSuggestion(index=row['index'], **spec)
        return out

    return _suggest


def test_batch_writes_an_explanation_as_well_as_an_account(app, analyze_user, analyze_file, monkeypatch):
    """A processed row must come back COMPLETE. The old batch wrote only the
    account, so every row it touched was left without an explanation."""
    _add_transactions(app, analyze_user, analyze_file, count=1)
    account_id = _add_account(app, analyze_user)

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _fake_bulk({
        0: {'account_id': account_id, 'account_name': 'Bank Fees',
            'confidence': 0.95, 'explanation': 'Monthly bank charge'},
    }))

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0)
        txn = Transaction.query.filter_by(file_id=analyze_file).first()

        assert result['applied'] == 1
        assert result['explained'] == 1
        assert txn.account_id == account_id
        assert txn.explanation == 'Monthly bank charge'
        assert txn.explanation_source == 'ai', "machine writing must be attributed"


def test_batch_never_overwrites_a_human_explanation(app, analyze_user, analyze_file, monkeypatch):
    """Re-running bulk processing over a partly-worked file is safe."""
    _add_transactions(app, analyze_user, analyze_file, count=1)
    account_id = _add_account(app, analyze_user)

    with app.app_context():
        txn = Transaction.query.filter_by(file_id=analyze_file).first()
        txn.explanation = 'Client says: annual insurance premium'
        txn.explanation_source = 'client'
        db.session.commit()

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _fake_bulk({
        0: {'account_id': account_id, 'account_name': 'Bank Fees',
            'confidence': 0.99, 'explanation': 'Monthly bank charge'},
    }))

    with app.app_context():
        process_transaction_batch(analyze_file, analyze_user, offset=0)
        txn = Transaction.query.filter_by(file_id=analyze_file).first()

        assert txn.explanation == 'Client says: annual insurance premium'
        assert txn.explanation_source == 'client'
        # The account was still legitimately assigned into its empty slot.
        assert txn.account_id == account_id


def test_batch_never_overwrites_an_existing_account(app, analyze_user, analyze_file, monkeypatch):
    """Import stamps the bank account onto every row; a suggestion must not
    silently replace it."""
    _add_transactions(app, analyze_user, analyze_file, count=1)
    bank_id = _add_account(app, analyze_user)

    with app.app_context():
        txn = Transaction.query.filter_by(file_id=analyze_file).first()
        txn.account_id = bank_id
        db.session.commit()

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _fake_bulk({
        0: {'account_id': 999999, 'account_name': 'Something Else',
            'confidence': 0.99, 'explanation': 'Reclassified'},
    }))

    with app.app_context():
        process_transaction_batch(analyze_file, analyze_user, offset=0)
        txn = Transaction.query.filter_by(file_id=analyze_file).first()

        assert txn.account_id == bank_id, "an existing assignment was overwritten"
        # ...but the empty explanation slot was still filled.
        assert txn.explanation == 'Reclassified'


def test_batch_applies_nothing_below_the_confidence_gate(app, analyze_user, analyze_file, monkeypatch):
    _add_transactions(app, analyze_user, analyze_file, count=1)
    account_id = _add_account(app, analyze_user)

    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows', _fake_bulk({
        0: {'account_id': account_id, 'account_name': 'Bank Fees',
            'confidence': 0.42, 'explanation': 'Possibly a bank charge'},
    }))

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0)
        txn = Transaction.query.filter_by(file_id=analyze_file).first()

        assert result['applied'] == 0
        assert txn.account_id is None
        # A low-confidence ACCOUNT does not suppress the explanation.
        assert txn.explanation == 'Possibly a bank charge'


def test_batch_offline_changes_nothing(app, analyze_user, analyze_file, monkeypatch):
    """With no suggestions at all, the rows are left exactly as they were."""
    _add_transactions(app, analyze_user, analyze_file, count=3)
    _add_account(app, analyze_user)
    monkeypatch.setattr('services.bulk_suggestions.suggest_for_rows',
                        lambda rows, accounts, client=None: {})

    with app.app_context():
        result = process_transaction_batch(analyze_file, analyze_user, offset=0)
        assert result['applied'] == 0
        assert result['explained'] == 0
        assert all(t.account_id is None
                   for t in Transaction.query.filter_by(file_id=analyze_file).all())


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
