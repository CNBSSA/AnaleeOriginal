"""Work order #115 item 4: where an earlier year's result goes in the trial balance.

The rule the Chairman approved on 2026-09-28: earlier years' income and expenses
roll into Retained Earnings (``q.200.000``) when the chart has it, else into
``q.100.000``. Cursor's #145 dropped the "else" so that a Private Company chart
without ``q.200.000`` would no longer pile that result onto Share Capital - a real
problem - but it also took the fallback away from the two entities whose chart has
NO ``q.200.000`` by design and whose ``q.100.000`` IS the owner's/organisation's
accumulated equity:

    Sole Proprietor   q.100.000 = Capital Account      (Chairman, 2026-10-12: option (a))
    NPO               q.100.000 = Accumulated Funds

The rule now:

    chart has q.200.000                  -> q.200.000                       (as today)
    Sole Proprietor / NPO without it     -> q.100.000
    anything else without it             -> a "Retained Earnings" line (q.200.000),
                                            never Share Capital / Members' Contribution

No totals change: only which equity line carries the earlier years' result.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from models import (Account, CompanySettings, Entity, Transaction, UploadedFile,
                    User, db)
from reports.trial_balance_service import load_trial_balance

FY2026 = datetime(2026, 6, 30)   # Feb year-end: FY2026 runs 1 Mar 2026 - 28 Feb 2027
IN_FY2025 = datetime(2025, 9, 1)
IN_FY2026 = datetime(2026, 9, 1)
_seq = iter(range(1, 10_000))


def _entity(name):
    entity = Entity.query.filter_by(name=name).first()
    if entity is None:
        entity = Entity(name=name)
        db.session.add(entity)
        db.session.commit()
    return entity.id


def _client(entity_name=None):
    n = next(_seq)
    user = User(username=f'ret{n}', email=f'ret{n}@example.com',
                subscription_status='active')
    user.set_password('password')
    db.session.add(user)
    db.session.commit()
    db.session.add(CompanySettings(
        user_id=user.id, company_name='TEST', financial_year_end=2,
        entity_id=_entity(entity_name) if entity_name else None))
    db.session.commit()
    return user.id


def _account(uid, link, name, category):
    acc = Account(link=link, name=name, category=category, sub_category=category,
                  user_id=uid)
    db.session.add(acc)
    db.session.flush()
    return acc


def _statement(uid, bank):
    f = UploadedFile(filename='s.csv', user_id=uid, bank_account_id=bank.id)
    db.session.add(f)
    db.session.flush()
    return f


def _line(uid, statement, account, amount, when):
    db.session.add(Transaction(date=when, description='line', amount=amount,
                               user_id=uid, account_id=account.id,
                               file_id=statement.id))


def _rows(ctx):
    return {r.link: r.amount for r in ctx.rows}


def _names(ctx):
    return {r.link: r.account_name for r in ctx.rows}


def _prior_year_expenses(entity_name, capital_name, with_capital=True):
    """A client with 25,301.24 of EARLIER-year expenses and 8,101.11 this year."""
    uid = _client(entity_name)
    bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
    if with_capital:
        _account(uid, 'q.100.000', capital_name, 'Equity')
    rent = _account(uid, 'e.400.000', 'Lease Rental - Premises', 'Expenses')
    utilities = _account(uid, 'e.410.000', 'Electricity', 'Expenses')
    s = _statement(uid, bank)
    _line(uid, s, rent, -20_000.0, IN_FY2025)
    _line(uid, s, utilities, -5_301.24, IN_FY2025)
    _line(uid, s, rent, -8_101.11, IN_FY2026)
    db.session.commit()
    return uid


PRIOR = Decimal('25301.24')


# ---- the two entities whose q.100.000 IS the accumulated equity ------------------

def test_sole_proprietor_prior_years_roll_into_the_capital_account(app):
    with app.app_context():
        uid = _prior_year_expenses('Sole Proprietor', 'Capital Account')
        ctx = load_trial_balance(uid, as_at=FY2026)
    rows = _rows(ctx)
    assert rows['q.100.000'] == PRIOR, 'earlier years belong on the Capital Account'
    assert _names(ctx)['q.100.000'] == 'Capital Account'
    assert 'q.200.000' not in rows, 'no made-up Retained Earnings line for a sole proprietor'
    assert rows['e.400.000'] == Decimal('8101.11')
    assert sum(rows.values()) == 0


def test_npo_prior_years_roll_into_accumulated_funds(app):
    with app.app_context():
        uid = _prior_year_expenses('NPO', 'Accumulated Funds')
        ctx = load_trial_balance(uid, as_at=FY2026)
    rows = _rows(ctx)
    assert rows['q.100.000'] == PRIOR
    assert _names(ctx)['q.100.000'] == 'Accumulated Funds'
    assert 'q.200.000' not in rows
    assert sum(rows.values()) == 0


def test_the_entity_name_is_matched_ignoring_case_and_spaces(app):
    with app.app_context():
        uid = _prior_year_expenses('sole proprietor ', 'Capital Account')
        ctx = load_trial_balance(uid, as_at=FY2026)
    assert _rows(ctx).get('q.100.000') == PRIOR


def test_a_sole_proprietors_own_capital_postings_and_the_roll_share_the_line(app):
    with app.app_context():
        uid = _client('Sole Proprietor')
        bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
        capital = _account(uid, 'q.100.000', 'Capital Account', 'Equity')
        sales = _account(uid, 'i.100.000', 'Sales', 'Income')
        s = _statement(uid, bank)
        _line(uid, s, capital, 50_000.0, IN_FY2025)    # owner puts money in
        _line(uid, s, sales, 1_000.0, IN_FY2025)       # earlier-year trading (a credit)
        _line(uid, s, sales, 200.0, IN_FY2026)
        db.session.commit()
        ctx = load_trial_balance(uid, as_at=FY2026)
    rows = _rows(ctx)
    assert rows['q.100.000'] == Decimal('-51000.00'), 'contribution and earlier profit together'
    assert rows['i.100.000'] == Decimal('-200.00')
    assert 'q.200.000' not in rows
    assert sum(rows.values()) == 0


def test_a_sole_proprietor_with_no_capital_account_in_the_chart_gets_the_retained_line(app):
    with app.app_context():
        uid = _prior_year_expenses('Sole Proprietor', '', with_capital=False)
        ctx = load_trial_balance(uid, as_at=FY2026)
    rows = _rows(ctx)
    assert rows['q.200.000'] == PRIOR
    assert _names(ctx)['q.200.000'] == 'Retained Earnings'
    assert sum(rows.values()) == 0


# ---- everyone else: #145's fix is kept ----------------------------------------------

def test_a_private_company_without_retained_earnings_never_uses_share_capital(app):
    with app.app_context():
        uid = _prior_year_expenses('Private Company', 'Share Capital - Ordinary')
        ctx = load_trial_balance(uid, as_at=FY2026)
    rows = _rows(ctx)
    assert 'q.100.000' not in rows, 'Share Capital must not absorb the earlier years'
    assert rows['q.200.000'] == PRIOR
    assert _names(ctx)['q.200.000'] == 'Retained Earnings'
    assert sum(rows.values()) == 0


def test_close_corporation_partnership_and_unknown_entities_use_the_retained_line(app):
    for entity in ('Close Corporation', 'Partnership', None):
        with app.app_context():
            uid = _prior_year_expenses(entity, 'Members Contribution')
            ctx = load_trial_balance(uid, as_at=FY2026)
        rows = _rows(ctx)
        assert 'q.100.000' not in rows, entity
        assert rows['q.200.000'] == PRIOR, entity
        assert sum(rows.values()) == 0, entity


# ---- when the chart HAS q.200.000 it is always used ---------------------------------

def test_retained_earnings_in_the_chart_is_used_for_every_entity(app):
    for entity in ('Sole Proprietor', 'NPO', 'Private Company', None):
        with app.app_context():
            uid = _client(entity)
            bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            _account(uid, 'q.100.000', 'Capital', 'Equity')
            retained = _account(uid, 'q.200.000', 'Accumulated profit', 'Equity')
            rent = _account(uid, 'e.400.000', 'Rent', 'Expenses')
            s = _statement(uid, bank)
            _line(uid, s, rent, -1_000.0, IN_FY2025)
            _line(uid, s, rent, -200.0, IN_FY2026)
            db.session.commit()
            ctx = load_trial_balance(uid, as_at=FY2026)
        rows = _rows(ctx)
        assert rows['q.200.000'] == Decimal('1000.00'), entity
        assert _names(ctx)['q.200.000'] == retained.name, entity
        assert 'q.100.000' not in rows, entity
        assert sum(rows.values()) == 0, entity


# ---- a single-year client is not affected by any of this ------------------------------

def test_a_single_year_client_is_identical_for_every_entity(app):
    for entity in ('Sole Proprietor', 'NPO', 'Private Company', None):
        with app.app_context():
            uid = _client(entity)
            bank = _account(uid, 'ca.810.001', 'Bank', 'Assets')
            _account(uid, 'q.100.000', 'Capital', 'Equity')
            rent = _account(uid, 'e.400.000', 'Rent', 'Expenses')
            s = _statement(uid, bank)
            _line(uid, s, rent, -500.0, IN_FY2026)
            db.session.commit()
            ctx = load_trial_balance(uid, as_at=FY2026)
        rows = _rows(ctx)
        assert set(rows) == {'ca.810.001', 'e.400.000'}, (entity, rows)
        assert sum(rows.values()) == 0


# ---- the rule is tied to the seeded charts, not only to literals in these tests --------
#
# The choice of equity line reads the entity's display name. If a (locked) seed rename
# or a chart change ever broke the assumption behind the rule, these fail loudly instead
# of the rule quietly switching itself off (independent review of this work order).

def _seeded_links(entity_name):
    from services.chart_seed_data import ENTITY_SPECIFIC
    return {row[2] for row in ENTITY_SPECIFIC[entity_name]}


def test_the_named_entities_exist_in_the_seed_under_exactly_those_names():
    from reports.trial_balance_service import CAPITAL_IS_RETAINED_ENTITIES
    from services.chart_seed_data import ENTITY_NAMES
    assert CAPITAL_IS_RETAINED_ENTITIES <= {n.strip().lower() for n in ENTITY_NAMES}


def test_the_entities_that_use_q100_as_retained_have_it_and_no_q200_in_their_seeded_chart():
    from reports.trial_balance_service import (CAPITAL_IS_RETAINED_ENTITIES,
                                               CAPITAL_IS_RETAINED_LINK)
    from services.chart_seed_data import ENTITY_NAMES
    for name in ENTITY_NAMES:
        if name.strip().lower() in CAPITAL_IS_RETAINED_ENTITIES:
            links = _seeded_links(name)
            assert CAPITAL_IS_RETAINED_LINK in links, name
            assert 'q.200.000' not in links, name


def test_every_other_entity_is_not_on_the_q100_list_and_q100_is_not_its_retained_line():
    from reports.trial_balance_service import CAPITAL_IS_RETAINED_ENTITIES
    from services.chart_seed_data import ENTITY_NAMES
    for name in ENTITY_NAMES:
        if name.strip().lower() not in CAPITAL_IS_RETAINED_ENTITIES:
            # Private Company and Close Corporation carry share capital / members'
            # contribution on q.100.000 and a real Retained Earnings on q.200.000;
            # Partnership has q.200.000. None may be sent to q.100.000.
            assert 'q.200.000' in _seeded_links(name), name
