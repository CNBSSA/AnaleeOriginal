"""History-first matching: reuse what this practice already decided.

The compounding tier. A recurring payee should be filed the way this firm filed
it last month — free, instant, consistent — and only genuinely new payees
should cost an AI call. These tests pin the payee-key normalisation (SA bank
narrations carry per-transaction reference/date noise) and, more importantly,
the refusal to guess when a payee's history is genuinely split.
"""
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from models import Account, Transaction, UploadedFile, User, db
from services.history_matching import (
    HistoryEntry,
    build_history_index,
    match_rows,
    normalize_description,
)


# --- payee key -------------------------------------------------------------

def test_reference_numbers_and_dates_are_stripped():
    """The real narration from Festus's FNB statement."""
    assert normalize_description('Magtape Credit Medihelp Smh0363383 20210204') \
        == 'magtape credit medihelp'


def test_the_same_payee_matches_across_different_references():
    a = normalize_description('Card Purchase Checkers Sandton 4021')
    b = normalize_description('Card Purchase Checkers Sandton 9987')
    assert a == b == 'card purchase checkers sandton'


def test_a_description_with_nothing_identifying_yields_no_key():
    assert normalize_description('') == ''
    assert normalize_description('12345 6789') == ''
    assert normalize_description(None) == ''
    assert normalize_description('fee') == '', "too generic to identify a payee"


def test_case_and_punctuation_do_not_split_a_payee():
    assert normalize_description('WOOLWORTHS, SANDTON.') \
        == normalize_description('woolworths sandton')


# --- verdict ---------------------------------------------------------------

def test_a_consistently_filed_payee_is_trusted():
    entry = HistoryEntry(key='k', account_counts={7: 4}, account_names={7: 'Bank Charges'},
                         explanation='Monthly bank fee', total=4)
    account_id, name, confidence = entry.verdict()
    assert (account_id, name) == (7, 'Bank Charges')
    assert confidence >= 0.95


def test_a_payee_seen_once_is_trusted_a_little_less():
    entry = HistoryEntry(key='k', account_counts={7: 1}, account_names={7: 'Bank Charges'},
                         total=1)
    _id, _name, confidence = entry.verdict()
    assert 0.85 <= confidence < 0.97


def test_a_dominant_account_still_clears_the_gate():
    entry = HistoryEntry(key='k', account_counts={7: 9, 8: 1},
                         account_names={7: 'Bank Charges', 8: 'Sundry'}, total=10)
    account_id, _name, confidence = entry.verdict()
    assert account_id == 7
    assert confidence >= 0.85


def test_a_genuinely_split_payee_is_refused_not_averaged():
    """A card used for two purposes must go to the accountant, not be guessed."""
    entry = HistoryEntry(key='k', account_counts={7: 5, 8: 4},
                         account_names={7: 'Travel', 8: 'Entertainment'}, total=9)
    _id, _name, confidence = entry.verdict()
    assert confidence < 0.85, "an ambiguous payee must not be auto-applied"


# --- index over the database ----------------------------------------------

@pytest.fixture
def seeded(app):
    with app.app_context():
        user = User(username='hist', email='hist@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()

        account = Account(link='e.321.000', name='Bank Charges',
                          category='Expenses', user_id=user.id)
        old_file = UploadedFile(filename='jan.pdf', user_id=user.id,
                                upload_date=datetime(2026, 1, 31))
        new_file = UploadedFile(filename='feb.pdf', user_id=user.id,
                                upload_date=datetime(2026, 2, 28))
        db.session.add_all([account, old_file, new_file])
        db.session.commit()

        # Settled practice: filed and explained on a PREVIOUS statement.
        db.session.add_all([
            Transaction(date=datetime(2026, 1, 5), description='FNB FEE 0011',
                        amount=-59.0, user_id=user.id, file_id=old_file.id,
                        account_id=account.id, explanation='Monthly bank fee',
                        explanation_source='accountant'),
            Transaction(date=datetime(2026, 1, 20), description='FNB FEE 0042',
                        amount=-59.0, user_id=user.id, file_id=old_file.id,
                        account_id=account.id, explanation='Monthly bank fee',
                        explanation_source='accountant'),
        ])
        db.session.commit()
        return {'user_id': user.id, 'account_id': account.id,
                'old_file': old_file.id, 'new_file': new_file.id}


def test_index_learns_from_completed_rows_only(app, seeded):
    with app.app_context():
        # An incomplete row must not teach the index.
        db.session.add(Transaction(
            date=datetime(2026, 1, 25), description='MYSTERY PAYEE',
            amount=-10.0, user_id=seeded['user_id'], file_id=seeded['old_file']))
        db.session.commit()

        index = build_history_index(seeded['user_id'])

    assert 'fnb fee' in index
    assert index['fnb fee'].total == 2
    assert 'mystery payee' not in index, "an unfinished row must not become precedent"


def test_a_recurring_payee_is_matched_without_any_ai(app, seeded):
    with app.app_context():
        index = build_history_index(seeded['user_id'], exclude_file_id=seeded['new_file'])
        matches = match_rows(
            [{'index': 0, 'description': 'FNB FEE 9999', 'amount': -59.0}], index)

    assert 0 in matches, "a payee filed twice before was not recognised"
    assert matches[0].account_id == seeded['account_id']
    assert matches[0].confidence >= 0.95
    assert matches[0].explanation == 'Monthly bank fee'


def test_a_new_payee_gets_no_history_match(app, seeded):
    with app.app_context():
        index = build_history_index(seeded['user_id'])
        matches = match_rows(
            [{'index': 0, 'description': 'BRAND NEW SUPPLIER', 'amount': -10.0}], index)
    assert matches == {}, "an unseen payee must fall through to the AI"


def test_a_statement_does_not_teach_itself(app, seeded):
    """Rows on the file being processed must not become their own precedent —
    otherwise one early mistake propagates through the rest of the statement."""
    with app.app_context():
        db.session.add(Transaction(
            date=datetime(2026, 2, 3), description='NEW PAYEE 001', amount=-25.0,
            user_id=seeded['user_id'], file_id=seeded['new_file'],
            account_id=seeded['account_id'], explanation='Guessed earlier in this file',
            explanation_source='ai'))
        db.session.commit()

        index = build_history_index(seeded['user_id'], exclude_file_id=seeded['new_file'])

    assert 'new payee' not in index
