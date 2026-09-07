"""One accountant decision applied to every matching row on the statement.

Exceptions arrive in families — forty card purchases at the same supermarket,
twelve identical debit orders. Deciding one and retyping it thirty-nine times
is the largest remaining piece of manual work in the product. The client wizard
has had a fan-out for a while; the accountant, who does the professional bulk
of the work, has had none.

The rules that matter here are the refusals: a row a person already decided is
never touched, and ids in the request are re-validated rather than trusted.
"""
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from models import Account, Transaction, UploadedFile, User, db
from services.accountant_fanout import (
    MAX_FANOUT,
    apply_to_rows,
    find_matching_rows,
    serialize_rows,
)


@pytest.fixture
def statement(app):
    """One statement: four rows from the same supermarket, one from elsewhere."""
    with app.app_context():
        user = User(username='fan', email='fan@x.test', subscription_status='active')
        user.set_password('pw12345!')
        db.session.add(user)
        db.session.commit()

        account = Account(link='e.334.000', name='Consumables',
                          category='Expenses', user_id=user.id)
        uploaded = UploadedFile(filename='feb.pdf', user_id=user.id,
                                upload_date=datetime(2026, 2, 28))
        db.session.add_all([account, uploaded])
        db.session.commit()

        rows = [
            Transaction(date=datetime(2026, 2, d),
                        description=f'Card Purchase Checkers Sandton {1000 + d}',
                        amount=-120.0, user_id=user.id, file_id=uploaded.id)
            for d in (3, 9, 17, 24)
        ]
        rows.append(Transaction(
            date=datetime(2026, 2, 11), description='FNB FEE 0042', amount=-59.0,
            user_id=user.id, file_id=uploaded.id))
        db.session.add_all(rows)
        db.session.commit()

        return {'user_id': user.id, 'file_id': uploaded.id,
                'account_id': account.id, 'ids': [r.id for r in rows]}


def test_it_finds_the_family_and_not_the_stranger(app, statement):
    with app.app_context():
        found = find_matching_rows(
            statement['file_id'], statement['user_id'], statement['ids'][0])

    assert found['key'] == 'card purchase checkers sandton'
    # 3 siblings; the source row itself and the unrelated bank fee are excluded.
    assert len(found['rows']) == 3
    assert all('Checkers' in r.description for r in found['rows'])


def test_a_row_a_person_already_decided_is_never_offered(app, statement):
    """The core refusal: a fan-out must not overwrite considered judgement."""
    with app.app_context():
        decided = db.session.get(Transaction, statement['ids'][1])
        decided.explanation = 'Client says this was a staff function, not stock'
        decided.explanation_source = 'client'
        db.session.commit()

        found = find_matching_rows(
            statement['file_id'], statement['user_id'], statement['ids'][0])

    offered = [r.id for r in found['rows']]
    assert statement['ids'][1] not in offered
    assert len(offered) == 2


def test_a_description_with_no_payee_offers_nothing(app, statement):
    with app.app_context():
        blank = Transaction(date=datetime(2026, 2, 5), description='12345 6789',
                            amount=-1.0, user_id=statement['user_id'],
                            file_id=statement['file_id'])
        db.session.add(blank)
        db.session.commit()
        found = find_matching_rows(
            statement['file_id'], statement['user_id'], blank.id)

    assert found['key'] == ''
    assert found['rows'] == [], "must not fan out on a key that identifies nothing"


def test_applying_sets_both_halves(app, statement):
    targets = statement['ids'][1:4]
    with app.app_context():
        result = apply_to_rows(
            statement['file_id'], statement['user_id'], targets,
            statement['account_id'], 'Weekly supermarket consumables')

        assert result['accounts_set'] == 3
        assert result['explanations_set'] == 3

        for tid in targets:
            txn = db.session.get(Transaction, tid)
            assert txn.account_id == statement['account_id']
            assert txn.explanation == 'Weekly supermarket consumables'
            assert txn.explanation_source == 'accountant'


def test_applying_skips_a_client_decided_row_even_if_asked(app, statement):
    """Ids come from the browser; the rule is enforced on apply, not just on
    preview."""
    with app.app_context():
        locked = db.session.get(Transaction, statement['ids'][1])
        locked.explanation = 'Client: personal, do not claim'
        locked.explanation_source = 'client'
        db.session.commit()

        result = apply_to_rows(
            statement['file_id'], statement['user_id'],
            [statement['ids'][1], statement['ids'][2]],
            statement['account_id'], 'Weekly supermarket consumables')

        untouched = db.session.get(Transaction, statement['ids'][1])
        assert untouched.explanation == 'Client: personal, do not claim'
        assert untouched.account_id is None, "a client-decided row was written to"
        assert result['skipped'] == 1
        assert result['accounts_set'] == 1


def test_rows_from_another_user_are_ignored(app, statement):
    """Ids are re-validated against the file AND the user."""
    with app.app_context():
        other = User(username='other', email='o@x.test', subscription_status='active')
        other.set_password('pw12345!')
        db.session.add(other)
        db.session.commit()
        theirs = Transaction(date=datetime(2026, 2, 3), description='Card Purchase Checkers',
                             amount=-50.0, user_id=other.id)
        db.session.add(theirs)
        db.session.commit()
        stolen_id = theirs.id

        result = apply_to_rows(
            statement['file_id'], statement['user_id'], [stolen_id],
            statement['account_id'], 'Not mine')

        assert result['accounts_set'] == 0
        assert db.session.get(Transaction, stolen_id).account_id is None


def test_an_account_belonging_to_someone_else_is_ignored(app, statement):
    with app.app_context():
        other = User(username='other2', email='o2@x.test', subscription_status='active')
        other.set_password('pw12345!')
        db.session.add(other)
        db.session.commit()
        foreign = Account(link='e.999.000', name='Their Account',
                          category='Expenses', user_id=other.id)
        db.session.add(foreign)
        db.session.commit()

        result = apply_to_rows(
            statement['file_id'], statement['user_id'], [statement['ids'][2]],
            foreign.id, 'Explanation still applies')

        txn = db.session.get(Transaction, statement['ids'][2])
        assert txn.account_id is None, "a foreign account was assigned"
        assert txn.explanation == 'Explanation still applies'
        assert result['accounts_set'] == 0


def test_no_ids_changes_nothing(app, statement):
    with app.app_context():
        assert apply_to_rows(statement['file_id'], statement['user_id'], [],
                             statement['account_id'], 'x')['accounts_set'] == 0


def test_serialisation_shape(app, statement):
    with app.app_context():
        found = find_matching_rows(
            statement['file_id'], statement['user_id'], statement['ids'][0])
        payload = serialize_rows(found['rows'])

    assert payload
    for row in payload:
        assert {'id', 'date', 'description', 'amount',
                'has_account', 'has_explanation'} <= set(row)


def test_fanout_is_bounded():
    assert 1 < MAX_FANOUT <= 500


def test_the_button_is_actually_reachable_from_the_page():
    """A feature nobody can click is not shipped. Guards the wiring: the button,
    its handler module, and the import that loads it."""
    with open('templates/analyze.html', encoding='utf-8') as handle:
        markup = handle.read()
    assert 'fanout-btn' in markup, 'no Apply to similar button on the row'
    assert 'FanoutHandler' in markup, 'the handler is never imported'
    assert 'bindUI' in markup

    with open('static/js/analyze/fanout.js', encoding='utf-8') as handle:
        script = handle.read()
    assert 'similar-rows' in script and 'apply-to-similar' in script


def test_routes_are_registered(canary_app):
    rules = {rule.rule for rule in canary_app.url_map.iter_rules()}
    assert '/analyze/<int:file_id>/similar-rows/<int:transaction_id>' in rules
    assert '/analyze/<int:file_id>/apply-to-similar' in rules
