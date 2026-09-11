"""Trial balance data + BooksXperts-compatible Excel export.

BooksXperts ``dataimports`` expects columns: Link, Account Name, Amount
(positive = debit, negative = credit). See booksxpert ``dataimports/sample_templates.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from typing import Sequence

from openpyxl import Workbook
from openpyxl.styles import Font
from sqlalchemy import and_

from models import Account, CompanySettings, Transaction

BOOKSXPERTS_TB_COLUMNS = ('Link', 'Account Name', 'Amount')


@dataclass(frozen=True)
class TrialBalanceRow:
    link: str
    account_name: str
    amount: Decimal  # signed: debit positive, credit negative


@dataclass(frozen=True)
class TrialBalanceContext:
    accounts: Sequence[Account]
    start_date: datetime
    end_date: datetime
    total_debits: Decimal
    total_credits: Decimal
    rows: tuple[TrialBalanceRow, ...]


def _quantize(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


# The chart's ``category`` vocabulary is exactly five values — Assets,
# Liabilities, Equity, Income, Expenses (services/chart_seed_data.py). The first
# three belong to the balance sheet and carry forward; the last two belong to the
# income statement and do not.
PROFIT_AND_LOSS_CATEGORIES = frozenset({'Income', 'Expenses'})


def _is_profit_or_loss(account: Account) -> bool:
    """True for an income-statement account.

    Anything unrecognised — a blank category, a value from some older chart — is
    treated as a balance-sheet account, i.e. it keeps the cumulative behaviour it
    has always had. An unknown account must not silently lose its prior years.
    """
    return (getattr(account, 'category', '') or '').strip() in PROFIT_AND_LOSS_CATEGORIES


def _account_balance(account: Account, start_date: datetime,
                     end_date: datetime) -> Decimal:
    """The account's trial-balance amount for the financial year (QA #13).

    **Balance-sheet accounts (Assets / Liabilities / Equity)** are cumulative up
    to and including ``end_date``. A bank balance is the sum of everything that
    ever happened to it, so it must carry forward.

    **Income-statement accounts (Income / Expenses)** carry the MOVEMENT inside
    the financial year only. Until now they were cumulative too, so a client with
    two years of data produced a 2025 trial balance containing 2024's income as
    well — and THE ACCOUNTANTS, which compiles the AFS from this payload, counted
    those prior years again. A trial balance for a year states that year's
    trading.

    Two things make this safe rather than a matter of taste:

    * the income statement in ``reports/routes.py`` already scopes P&L with
      ``Transaction.date.between(from_date, to_date)``, so this makes the two
      reports agree where they previously disagreed;
    * for a client whose data lies inside ONE financial year the two rules give
      the same number, so the common case is byte-identical (locked by test).

    Analee's transactions are single-legged — one row per bank line, one
    ``account_id`` — so there is no contra entry and no retained-earnings roll.
    Nothing here balances debits against credits and nothing here ever did; the
    page says as much ("a cash-basis summary from categorised bank activity — not
    an accrual GL"). That is why this change needs no closing entries.
    """
    if _is_profit_or_loss(account):
        total = sum((t.amount for t in account.transactions
                     if start_date <= t.date <= end_date), 0.0)
    else:
        total = sum((t.amount for t in account.transactions
                     if t.date <= end_date), 0.0)
    return _quantize(Decimal(str(total)))


def load_trial_balance(
    user_id: int,
    *,
    as_at: datetime | None = None,
    year: int | None = None,
) -> TrialBalanceContext:
    """Load FY-scoped trial balance for ``user_id`` (same logic as the HTML report).

    By default the balance is the client's CURRENT financial year — the one
    containing today. That is right for the accountant looking at the books,
    and wrong for the accountant compiling the year just closed: during AFS
    season the share link and the export carried the next year's year-to-date
    position, and nothing could ask for the year that had ended (QA register
    #6, 2026-09-07). Either keyword selects a different year and is passed
    straight through to ``CompanySettings.get_financial_year``:

    - ``as_at`` — any date inside the wanted year (a consumer that knows the
      year-end passes exactly that date, and the balance comes back as at it);
    - ``year`` — the year the financial year STARTS in, matching the
      ``financial_year`` selector the Analee reports already use.

    Both omitted → identical to the original behaviour.
    """
    company_settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if company_settings is None:
        raise ValueError('Company settings are not configured.')

    fy_dates = company_settings.get_financial_year(date=as_at, year=year)
    start_date, end_date = fy_dates['start_date'], fy_dates['end_date']

    # QA #13/#12: the candidate set is every account this client has ever posted
    # to on or before the period end. The previous query filtered the JOINED table
    # on `date >= start AND date <= end`, which turns an outer join into an
    # effective INNER one: an account was listed only if it had activity INSIDE
    # the year, so a balance-sheet account carrying a prior-year balance with no
    # current-year movement disappeared from the trial balance while its balance
    # was not zero. A trial balance must list every account with a balance.
    #
    # Selected by id through a distinct sub-select rather than a join, so the
    # result can never depend on how many transactions an account happens to
    # have. Accounts with no transactions at all are still excluded, exactly as
    # before — the page does not render the whole chart of accounts.
    posted_account_ids = {
        row[0]
        for row in Transaction.query.with_entities(Transaction.account_id)
        .filter(
            and_(
                Transaction.user_id == user_id,
                Transaction.account_id.isnot(None),
                Transaction.date <= end_date,
            )
        )
        .distinct()
        .all()
    }
    candidates = (
        Account.query.filter(
            and_(Account.user_id == user_id, Account.id.in_(posted_account_ids)))
        .order_by(Account.link)
        .all()
        if posted_account_ids else []
    )

    accounts: list[Account] = []
    total_debits = Decimal('0')
    total_credits = Decimal('0')
    export_rows: list[TrialBalanceRow] = []

    for account in candidates:
        balance = _account_balance(account, start_date, end_date)
        moved_in_period = any(
            start_date <= t.date <= end_date for t in account.transactions)
        # Shown when it traded this year (even if it nets to zero — that row was
        # always displayed as 0.00) OR when it carries a balance into the year.
        # Hidden only when there is genuinely nothing to report.
        if not moved_in_period and balance == 0:
            continue
        accounts.append(account)
        if balance > 0:
            total_debits += balance
        elif balance < 0:
            total_credits += abs(balance)
        if balance != 0:
            export_rows.append(
                TrialBalanceRow(
                    link=account.link,
                    account_name=account.name,
                    amount=balance,
                )
            )

    return TrialBalanceContext(
        accounts=tuple(accounts),
        start_date=start_date,
        end_date=end_date,
        total_debits=_quantize(total_debits),
        total_credits=_quantize(total_credits),
        rows=tuple(export_rows),
    )


def build_booksxperts_trial_balance_xlsx(
    rows: Sequence[TrialBalanceRow],
    *,
    company_name: str = '',
    period_end: datetime | None = None,
) -> bytes:
    """Build an ``.xlsx`` trial balance matching BooksXperts upload column headers."""
    wb = Workbook()
    ws = wb.active
    ws.title = 'Trial Balance'

    ws.append(list(BOOKSXPERTS_TB_COLUMNS))
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for row in rows:
        ws.append([row.link, row.account_name, float(row.amount)])

    if company_name or period_end:
        ws.append([])
        meta = 'Analee trial balance'
        if company_name:
            meta += f' — {company_name}'
        if period_end:
            meta += f' — as at {period_end.strftime("%Y-%m-%d")}'
        ws.append([meta])
        ws.append(['Import targets: BooksXperts (Data Imports → Upload Trial Balance) '
                   'or The Accountants (trial balance intake).'])
        ws.append(['Amount: positive = debit, negative = credit. Rows must sum to zero.'])
        ws.append(['Analee produces cash-basis balances from bank categorisation — '
                   'not an accrual GL or official AFS.'])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_filename(period_end: datetime) -> str:
    return f'analee-trial-balance-{period_end.strftime("%Y-%m-%d")}.xlsx'


def build_trial_balance_payload(
    ctx: TrialBalanceContext,
    *,
    user_id: int,
    company_name: str = '',
    registration_number: str | None = None,
) -> dict:
    """JSON-serialisable trial balance for API transmission (Phase 5).

    Contract for BooksXperts / The Accountants intake:
    ``{ company_id, as_at, rows: [{link, name, amount}] }``
    """
    rows = [
        {
            'link': row.link,
            'name': row.account_name,
            'amount': float(row.amount),
        }
        for row in ctx.rows
    ]
    return {
        'format_version': 1,
        'source': 'analee',
        'company_id': user_id,
        'company_name': company_name,
        'registration_number': registration_number or '',
        'as_at': ctx.end_date.strftime('%Y-%m-%d'),
        'period_start': ctx.start_date.strftime('%Y-%m-%d'),
        'period_end': ctx.end_date.strftime('%Y-%m-%d'),
        'rows': rows,
        'balanced': ctx.total_debits == ctx.total_credits,
        'total_debits': float(ctx.total_debits),
        'total_credits': float(ctx.total_credits),
    }
