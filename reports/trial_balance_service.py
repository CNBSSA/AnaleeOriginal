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


def _account_balance(account: Account, end_date: datetime) -> Decimal:
    """Cumulative balance up to and including ``end_date``.

    Until 2026-09-07 this summed EVERY transaction on the account, so only the
    choice of accounts was scoped to the financial year — the amounts were
    all-time. Invisible while a company held a single year of data; with two
    years, a balance "as at 28 Feb 2026" included April 2026's receipts. The
    ``<= end_date`` rule is the same one the financial-position report in this
    module already applies: cumulative to the period end, exactly as before for
    the current year (whose end lies in the future), honest for a closed one.
    """
    total = sum((t.amount for t in account.transactions if t.date <= end_date), 0.0)
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
    accounts = (
        Account.query.filter_by(user_id=user_id)
        .outerjoin(Account.transactions)
        .filter(
            and_(
                Transaction.date >= fy_dates['start_date'],
                Transaction.date <= fy_dates['end_date'],
            )
        )
        .order_by(Account.link)
        .all()
    )

    total_debits = Decimal('0')
    total_credits = Decimal('0')
    export_rows: list[TrialBalanceRow] = []

    for account in accounts:
        balance = _account_balance(account, fy_dates['end_date'])
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
        accounts=accounts,
        start_date=fy_dates['start_date'],
        end_date=fy_dates['end_date'],
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
