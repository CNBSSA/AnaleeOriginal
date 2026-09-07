"""Automation writes at scale — the accountant must be able to see what it wrote.

Every explanation write is tagged (``ai`` for the machine, accountant/client for
people), but until now the tag was recorded and displayed nowhere. In a
professional ledger an accountant has to be able to tell a machine-written line
from their own, and to review the machine's output specifically rather than
re-reading a whole statement.
"""
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from models import Account, Transaction, UploadedFile, User, db
from services.analyze_processing import get_paginated_transactions, provenance_summary


@pytest.fixture
def mixed_file(app):
    """One statement explained by all three authors, plus an untouched row."""
    with app.app_context():
        user = User(username='prov', email='prov@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()

        account = Account(link='e.321.000', name='Bank Charges',
                          category='Expenses', user_id=user.id)
        uploaded = UploadedFile(filename='feb.pdf', user_id=user.id)
        db.session.add_all([account, uploaded])
        db.session.commit()

        def _row(day, source, text):
            return Transaction(
                date=datetime(2026, 2, day), description=f'ROW {day}', amount=-10.0,
                user_id=user.id, file_id=uploaded.id, account_id=account.id,
                explanation=text, explanation_source=source)

        db.session.add_all([
            _row(1, 'ai', 'Monthly bank fee'),
            _row(2, 'ai', 'Card purchase'),
            _row(3, 'accountant', 'Reclassified after review'),
            _row(4, 'client', 'Client says: personal'),
            _row(5, 'client_erf', 'Client says: personal'),
            _row(6, '', 'Explained before provenance existed'),
            # Untouched: no explanation at all.
            Transaction(date=datetime(2026, 2, 7), description='ROW 7', amount=-10.0,
                        user_id=user.id, file_id=uploaded.id),
        ])
        db.session.commit()
        return {'user_id': user.id, 'file_id': uploaded.id}


def test_summary_counts_each_author(app, mixed_file):
    with app.app_context():
        counts = provenance_summary(mixed_file['file_id'], mixed_file['user_id'])

    assert counts['ai'] == 2
    assert counts['accountant'] == 1
    assert counts['client'] == 2, "client and client_erf both count as the client"
    assert counts['unattributed'] == 1
    assert counts['explained'] == 6
    assert counts['total'] == 7, "the unexplained row still belongs to the statement"


def test_the_ai_review_filter_returns_only_machine_written_rows(app, mixed_file):
    """Spot-checking automation must not mean re-reading the whole statement."""
    with app.app_context():
        rows, total, _pages = get_paginated_transactions(
            mixed_file['file_id'], mixed_file['user_id'], page=1, per_page=50,
            only_ai_written=True)

    assert total == 2
    assert {r.description for r in rows} == {'ROW 1', 'ROW 2'}
    assert all(r.explanation_source == 'ai' for r in rows)


def test_the_default_view_is_unchanged(app, mixed_file):
    with app.app_context():
        _rows, total, _pages = get_paginated_transactions(
            mixed_file['file_id'], mixed_file['user_id'], page=1, per_page=50)
    assert total == 7


def test_a_statement_nobody_has_explained_summarises_cleanly(app):
    with app.app_context():
        user = User(username='empty', email='empty@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()
        uploaded = UploadedFile(filename='new.pdf', user_id=user.id)
        db.session.add(uploaded)
        db.session.commit()
        db.session.add(Transaction(date=datetime(2026, 2, 1), description='X',
                                   amount=-1.0, user_id=user.id, file_id=uploaded.id))
        db.session.commit()

        counts = provenance_summary(uploaded.id, user.id)

    assert counts['explained'] == 0
    assert counts['total'] == 1


def test_the_page_actually_shows_provenance():
    """Recorded-but-never-displayed is the defect being fixed; guard the wiring."""
    with open('templates/analyze.html', encoding='utf-8') as handle:
        markup = handle.read()

    assert 'explanation_source' in markup, 'no per-row author badge'
    assert 'Who explained what' in markup, 'no provenance summary'
    assert "view='ai'" in markup, 'no way to review what the machine wrote'
