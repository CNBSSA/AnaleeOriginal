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


# --- Double entry from single-legged bank lines (Festus re-open, 2026-09-28) --
#
# Analee stores ONE row per bank line: a bank-signed amount (money in +, money
# out -) and the account the line was categorised to. Until 2026-09-28 the
# trial balance summed that bank-signed amount straight onto the category, so
# income read as a debit, expenses as a credit, the bank account was missing,
# and THE ACCOUNTANTS imported the profit with its sign inverted.
#
# Each bank line is now posted as the entry it is:
#   bank account          +amount   (money in debits the bank)
#   categorised account   -amount   (so income is a credit, an expense a debit)
# which balances by construction. Where the other side is not known yet the
# line waits in the Suspense Account, which is what an accountant would do.

BANK_LINK_PREFIXES = ('ca.810', 'ca.820', 'cl.810')
SUSPENSE_LINK = 'ca.900.000'
SUSPENSE_NAME = 'Suspense Account'
# A statement from before the bank account was recorded, whose bank cannot be
# inferred: its bank side still has to be posted somewhere, so it is posted to
# one clearly named row rather than guessed onto one of the client's banks.
UNRECORDED_BANK_LINK = 'ca.810.000'
UNRECORDED_BANK_NAME = 'Bank (statement account not recorded)'
RETAINED_LINKS = ('q.200.000', 'q.100.000')
RETAINED_NAME = 'Retained Earnings'


@dataclass(frozen=True)
class _SyntheticAccount:
    """A trial-balance row with no Account behind it (same shape the page reads)."""
    link: str
    name: str
    category: str


def _is_bank(account) -> bool:
    link = (getattr(account, 'link', '') or '').lower()
    return link.startswith(BANK_LINK_PREFIXES)


def _statement_banks(user_id: int, transactions) -> dict:
    """Map file_id -> the bank Account its lines belong to.

    Recorded on the statement since 2026-09-28. For an older statement it is
    inferred, in this order, and never guessed across banks:
      1. a line of that statement still sitting on a bank-type account (the
         import stamps the bank onto every line until it is categorised);
      2. the client's only bank-type account ever used for a statement.
    Anything left maps to None and is posted to UNRECORDED_BANK_LINK.
    """
    from models import BankStatementUpload, UploadedFile

    file_ids = {t.file_id for t in transactions if t.file_id is not None}
    accounts_by_id = {a.id: a for a in Account.query.filter_by(user_id=user_id).all()}
    banks: dict = {}
    if file_ids:
        for f in UploadedFile.query.filter(UploadedFile.id.in_(file_ids)).all():
            acct = accounts_by_id.get(getattr(f, 'bank_account_id', None))
            if acct is not None:
                banks[f.id] = acct

    stamped: dict = {}
    for t in transactions:
        if t.file_id is None or t.file_id in banks:
            continue
        acct = accounts_by_id.get(t.account_id)
        if acct is not None and _is_bank(acct):
            stamped.setdefault(t.file_id, {}).setdefault(acct.id, 0)
            stamped[t.file_id][acct.id] += 1
    for file_id, counts in stamped.items():
        if len(counts) == 1:
            banks[file_id] = accounts_by_id[next(iter(counts))]

    used = {a.id for a in banks.values()}
    used |= {row[0] for row in UploadedFile.query.with_entities(
        UploadedFile.bank_account_id).filter_by(user_id=user_id)}
    used |= {u.account_id for u in BankStatementUpload.query.filter_by(user_id=user_id)}
    used_banks = [accounts_by_id[i] for i in used if i in accounts_by_id
                  and _is_bank(accounts_by_id[i])]
    only_bank = used_banks[0] if len(used_banks) == 1 else None
    for file_id in file_ids:
        banks.setdefault(file_id, only_bank)
    banks[None] = only_bank
    return banks


def _first_by_link(accounts_by_link: dict, links, name: str, category: str):
    for link in links:
        if link in accounts_by_link:
            return accounts_by_link[link]
    return _SyntheticAccount(link=links[0], name=name, category=category)


def load_trial_balance(
    user_id: int,
    *,
    as_at: datetime | None = None,
    year: int | None = None,
) -> TrialBalanceContext:
    """Load the FY-scoped trial balance for ``user_id``.

    By default the balance is the client's CURRENT financial year — the one
    containing today. ``as_at`` (any date inside the wanted year) or ``year``
    (the year the financial year STARTS in) select another year and are passed
    straight to ``CompanySettings.get_financial_year`` (QA register #6).

    Semantics (QA #12/#13, kept): balance-sheet accounts are cumulative to the
    period end; income and expense accounts carry the year's movement only.
    Earlier years' income and expenses now roll into Retained Earnings, so
    the trial balance balances for a multi-year client too.

    Every row sums to zero: total debits equal total credits.
    """
    company_settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if company_settings is None:
        raise ValueError('Company settings are not configured.')

    fy_dates = company_settings.get_financial_year(date=as_at, year=year)
    start_date, end_date = fy_dates['start_date'], fy_dates['end_date']

    transactions = (
        Transaction.query.filter(
            and_(Transaction.user_id == user_id, Transaction.date <= end_date))
        .order_by(Transaction.id)
        .all()
    )
    accounts_by_id = {a.id: a for a in Account.query.filter_by(user_id=user_id).all()}
    accounts_by_link = {a.link: a for a in accounts_by_id.values()}
    banks = _statement_banks(user_id, transactions)
    suspense = _first_by_link(accounts_by_link, (SUSPENSE_LINK,), SUSPENSE_NAME, 'Assets')
    retained = _first_by_link(accounts_by_link, RETAINED_LINKS, RETAINED_NAME, 'Equity')
    unrecorded_bank = _SyntheticAccount(
        link=UNRECORDED_BANK_LINK, name=UNRECORDED_BANK_NAME, category='Assets')

    balances: dict = {}
    moved: set = set()
    holders: dict = {}

    def post(holder, amount: Decimal, in_period: bool) -> None:
        holders[holder.link] = holder
        balances[holder.link] = balances.get(holder.link, Decimal('0')) + amount
        if in_period:
            moved.add(holder.link)

    for t in transactions:
        amount = Decimal(str(t.amount))
        in_period = start_date <= t.date <= end_date
        category = accounts_by_id.get(t.account_id)
        bank = banks.get(t.file_id)
        if bank is None:
            # No statement bank known: a line still sitting on a bank-type
            # account is that bank's own, not yet categorised.
            bank = category if category is not None and _is_bank(category) \
                else unrecorded_bank
        if category is None or getattr(category, 'link', None) == bank.link:
            category = suspense
        post(bank, amount, in_period)
        if _is_profit_or_loss(category) and not in_period:
            # An earlier year's trading is already in the opening equity.
            post(retained, -amount, False)
        else:
            post(category, -amount, in_period)

    accounts: list = []
    total_debits = Decimal('0')
    total_credits = Decimal('0')
    export_rows: list[TrialBalanceRow] = []

    for link in sorted(balances):
        holder = holders[link]
        balance = _quantize(balances[link])
        # Shown when it moved this year (even at 0.00) or carries a balance in.
        if link not in moved and balance == 0:
            continue
        accounts.append(holder)
        if balance > 0:
            total_debits += balance
        elif balance < 0:
            total_credits += abs(balance)
        if balance != 0:
            export_rows.append(TrialBalanceRow(
                link=holder.link, account_name=holder.name, amount=balance))

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
