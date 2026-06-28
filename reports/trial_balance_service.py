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


def _account_balance(account: Account) -> Decimal:
    total = sum((t.amount for t in account.transactions), 0.0)
    return _quantize(Decimal(str(total)))


def load_trial_balance(user_id: int) -> TrialBalanceContext:
    """Load FY-scoped trial balance for ``user_id`` (same logic as the HTML report)."""
    company_settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if company_settings is None:
        raise ValueError('Company settings are not configured.')

    fy_dates = company_settings.get_financial_year()
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
        balance = _account_balance(account)
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
        ws.append(['Import in BooksXperts: Data Imports → Upload Trial Balance.'])
        ws.append(['Amount: positive = debit, negative = credit. Rows must sum to zero.'])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_filename(period_end: datetime) -> str:
    return f'analee-trial-balance-{period_end.strftime("%Y-%m-%d")}.xlsx'
