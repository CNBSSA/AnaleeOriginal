"""QA register #9–#11 — the trial-balance page must agree with itself.

All three are regressions from the 2026-09-07 scoped unfreeze that taught
``load_trial_balance`` to serve a closed financial year. That change was correct in
the service and incomplete everywhere around it:

* **#9** ``templates/reports/trial_balance.html`` re-derived every displayed row by
  summing ``account.transactions`` — the whole ORM relationship, no period bound —
  while the totals came from the service. Before the change both were all-time:
  wrong, but consistent. After it the page contradicted itself.
* **#10** the page's Excel, JSON and share-link actions built their URLs with no
  query string, so acting on a closed year silently served the current one.
* **#11** ``financial_year=YYYY`` was off by one for December year-ends, because
  ``get_financial_year`` keeps ``start_year`` one less than the calendar year it
  represents and only its date-derived path compensates.

The template is rendered from the real file here. A stub would not have caught #9,
because the defect was in the template rather than in anything Python.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from models import Account, CompanySettings, Transaction, User, db
from reports.routes import _fy_probe_date
from reports.trial_balance_service import load_trial_balance

TEMPLATE = 'reports/trial_balance.html'


def _user(app, *, fy_end: int) -> int:
    with app.app_context():
        user = User(username=f'tb{fy_end}', email=f'tb{fy_end}@example.com',
                    subscription_status='active')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(
            user_id=user.id, company_name='ACME Pty Ltd',
            registration_number='2020/123456/07', financial_year_end=fy_end))
        db.session.commit()
        return user.id


def _account(user_id: int, link: str, name: str, category: str) -> int:
    acc = Account(link=link, name=name, category=category,
                  sub_category=category, user_id=user_id)
    db.session.add(acc)
    db.session.flush()
    return acc.id


def _render(ctx, *, period_query=None):
    """Render the REAL template the way the route does."""
    # ChainableUndefined so base.html's own globals (current_user, …) resolve
    # quietly. The child template under test is the REAL file and every value it
    # uses is supplied below, so nothing it depends on is being stubbed away.
    env = Environment(loader=FileSystemLoader('templates'),
                      undefined=ChainableUndefined)
    env.globals['get_flashed_messages'] = lambda *a, **k: []
    env.globals['url_for'] = lambda endpoint, **kw: (
        '/' + endpoint + ('?' + '&'.join(f'{k}={v}' for k, v in sorted(kw.items())
                                         if k != '_external') if kw else ''))
    # The page extends base.html; render only the content block's table by
    # supplying a minimal parent-free template would change the artefact, so load
    # the real one and tolerate base.html's own requirements.
    template = env.get_template(TEMPLATE)
    return template.render(
        accounts=ctx.accounts,
        amounts_by_link={row.link: row.amount for row in ctx.rows},
        start_date=ctx.start_date,
        end_date=ctx.end_date,
        total_debits=ctx.total_debits,
        total_credits=ctx.total_credits,
        tb_balanced=ctx.total_debits == ctx.total_credits,
        period_options=[],
        period_query=period_query or {},
    )


def _rendered_amounts(html: str) -> list[tuple[str, str]]:
    """(debit, credit) text of each body row, in order."""
    import re
    body = html.split('<tbody>', 1)[1].split('</tbody>', 1)[0]
    rows = re.findall(r'<tr>(.*?)</tr>', body, re.S)
    out = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
        if len(cells) >= 4:
            out.append((cells[2].strip(), cells[3].strip()))
    return out


class TestTheRowsAgreeWithTheTotals:
    def test_a_transaction_after_the_period_end_is_in_neither(self, app):
        """The reproduction: a closed year's rows must not include next year's
        activity while the totals exclude it. Previously -60.00 against -30.00."""
        user_id = _user(app, fy_end=2)
        with app.app_context():
            sales = _account(user_id, 'i.100.000', 'Sales', 'Income')
            db.session.add_all([
                Transaction(date=datetime(2025, 6, 1), description='In period',
                            amount=-30.0, user_id=user_id, account_id=sales),
                Transaction(date=datetime(2026, 6, 1), description='Next year',
                            amount=-30.0, user_id=user_id, account_id=sales),
            ])
            db.session.commit()

            ctx = load_trial_balance(user_id, as_at=datetime(2025, 6, 30))
            html = _render(ctx)

        assert ctx.total_credits == Decimal('30.00')
        amounts = _rendered_amounts(html)
        assert amounts, 'no rows rendered — the assertion below would be vacuous'
        assert amounts == [('0.00', '30.00')], (
            f'the page shows {amounts} while the service totals say '
            f'credits={ctx.total_credits} — the page disagrees with itself')

    def test_the_rendered_rows_sum_to_the_rendered_totals(self, app):
        """A general invariant, not one example: whatever the data, the columns add
        up to the footer. This is the property that was broken."""
        user_id = _user(app, fy_end=2)
        with app.app_context():
            bank = _account(user_id, 'ca.810.001', 'Bank', 'Assets')
            sales = _account(user_id, 'i.100.000', 'Sales', 'Income')
            db.session.add_all([
                Transaction(date=datetime(2025, 6, 1), description='Receipt',
                            amount=140.0, user_id=user_id, account_id=bank),
                Transaction(date=datetime(2025, 7, 1), description='Sale',
                            amount=-140.0, user_id=user_id, account_id=sales),
                Transaction(date=datetime(2027, 1, 1), description='Future',
                            amount=999.0, user_id=user_id, account_id=bank),
            ])
            db.session.commit()

            ctx = load_trial_balance(user_id, as_at=datetime(2025, 6, 30))
            html = _render(ctx)

        amounts = _rendered_amounts(html)
        assert amounts
        debits = sum(Decimal(d) for d, _ in amounts)
        credits = sum(Decimal(c) for _, c in amounts)
        assert debits == ctx.total_debits, f'{debits} != {ctx.total_debits}'
        assert credits == ctx.total_credits, f'{credits} != {ctx.total_credits}'

    def test_a_zero_balance_account_is_still_listed(self, app):
        """Iron Rule: the row set must not shrink. An account whose entries net to
        zero was shown as 0.00 / 0.00 before and must still be."""
        user_id = _user(app, fy_end=2)
        with app.app_context():
            bank = _account(user_id, 'ca.810.001', 'Bank', 'Assets')
            db.session.add_all([
                Transaction(date=datetime(2025, 6, 1), description='In',
                            amount=50.0, user_id=user_id, account_id=bank),
                Transaction(date=datetime(2025, 6, 2), description='Out',
                            amount=-50.0, user_id=user_id, account_id=bank),
            ])
            db.session.commit()

            ctx = load_trial_balance(user_id, as_at=datetime(2025, 6, 30))
            html = _render(ctx)

        assert _rendered_amounts(html) == [('0.00', '0.00')]


class TestThePeriodSurvivesEveryAction:
    def test_the_three_actions_carry_the_period(self, app):
        user_id = _user(app, fy_end=2)
        with app.app_context():
            bank = _account(user_id, 'ca.810.001', 'Bank', 'Assets')
            db.session.add(Transaction(date=datetime(2025, 6, 1), description='R',
                                       amount=10.0, user_id=user_id, account_id=bank))
            db.session.commit()
            ctx = load_trial_balance(user_id, as_at=datetime(2025, 6, 30))
            html = _render(ctx, period_query={'financial_year': '2025'})

        for endpoint in ('reports.trial_balance_export', 'reports.trial_balance_api',
                         'reports.trial_balance_share_link'):
            assert f'/{endpoint}?financial_year=2025' in html, (
                f'{endpoint} lost the period — it would serve the current year')

    def test_without_a_period_the_urls_are_unchanged(self, app):
        """No selector given → the links look exactly as they always did."""
        user_id = _user(app, fy_end=2)
        with app.app_context():
            bank = _account(user_id, 'ca.810.001', 'Bank', 'Assets')
            db.session.add(Transaction(date=datetime(2025, 6, 1), description='R',
                                       amount=10.0, user_id=user_id, account_id=bank))
            db.session.commit()
            ctx = load_trial_balance(user_id, as_at=datetime(2025, 6, 30))
            html = _render(ctx)

        assert '/reports.trial_balance_export"' in html
        assert 'financial_year=' not in html.split('<tbody>')[0]


class TestTheFinancialYearSelectorIsNotOffByOne:
    def test_a_december_year_end_resolves_to_that_calendar_year(self, app):
        """#11. financial_year=2025 used to return 2026-01-01..2026-12-31."""
        user_id = _user(app, fy_end=12)
        with app.app_context():
            settings = CompanySettings.query.filter_by(user_id=user_id).first()
            dates = settings.get_financial_year(date=_fy_probe_date(2025, 12))
        assert dates['start_date'].date() == datetime(2025, 1, 1).date()
        assert dates['end_date'].date() == datetime(2025, 12, 31).date()

    @pytest.mark.parametrize('fy_end,expected_start,expected_end', [
        (2, (2025, 3, 1), (2026, 2, 28)),
        (6, (2025, 7, 1), (2026, 6, 30)),
        (11, (2025, 12, 1), (2026, 11, 30)),
    ])
    def test_non_december_year_ends_are_unchanged(self, app, fy_end,
                                                  expected_start, expected_end):
        """The twin: the probe must not disturb the year-ends that already worked."""
        user_id = _user(app, fy_end=fy_end)
        with app.app_context():
            settings = CompanySettings.query.filter_by(user_id=user_id).first()
            dates = settings.get_financial_year(date=_fy_probe_date(2025, fy_end))
        assert dates['start_date'].date() == datetime(*expected_start).date()
        assert dates['end_date'].date() == datetime(*expected_end).date()

    def test_the_probe_date_always_lands_inside_the_year_it_selects(self, app):
        """The property the fix rests on, across every possible year-end."""
        for fy_end in range(1, 13):
            user_id = _user(app, fy_end=fy_end)
            with app.app_context():
                settings = CompanySettings.query.filter_by(user_id=user_id).first()
                probe = _fy_probe_date(2025, fy_end)
                dates = settings.get_financial_year(date=probe)
            assert dates['start_date'] <= probe <= dates['end_date'], (
                f'year-end {fy_end}: probe {probe} is outside '
                f"{dates['start_date']}..{dates['end_date']}")
