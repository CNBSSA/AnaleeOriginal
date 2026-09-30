"""Per-bank layout profiles for Tier-1 SA bank-statement text parsing.

Each profile knows how to detect its bank in statement text and how to parse
transaction lines for that bank's typical column layout. Unmatched text falls
back to the generic profile.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, List, Optional, Sequence

from .statement_integrity import (
    StatementLine, normalize_amount, normalize_date, resolve_date_with_period,
)

_DATE_TOKEN = (
    r'(?:\d{4}[/-]\d{2}[/-]\d{2}'
    r'|\d{2}[/-]\d{2}[/-]\d{2,4}'
    r'|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}'
    # Year-less "01 Feb" (FNB prints the year only in the header; 2026-09-30).
    # Three-letter month only and never followed by a year. Given its year by
    # _complete_leading_date before any parser sees the line.
    r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?![a-z])(?!\s+\d{4}\b))'
)

_DATE_RE = re.compile(rf'^\s*({_DATE_TOKEN})\s+(.*)$', re.IGNORECASE)
# One money token. The integer part is EITHER properly grouped thousands
# ("1 500", "12,345,678" — a 1-3 digit lead then exact 3-digit groups) OR an
# ungrouped run of digits ("1500"), always followed by cents.
#
# The previous pattern was ``[\d\s,]*\d\.\d{2}``, which let a token absorb ANY
# run of digits and spaces in front of the cents. So a figure standing next to
# other digits was read as one number: "POS 4521 150.00" became R4 521 150.00,
# and "1 500 8 650.25" (amount then balance) became R15 008 650.25 — the
# ~R10 000 000 rows seen live on 2026-09-30. A token now may not start straight
# after a digit, comma or point (so it cannot swallow a reference number in
# front of it) — except straight after another amount's cents, so abutting
# columns such as "1500.008650.25" still split into 1500.00 and 8650.25.
_AMOUNT_RE = re.compile(
    r'\(?-?(?:R\s?)?'
    r'(?:(?<![\d,.])|(?<=\.\d\d))'
    r'(?:\d{1,3}(?:[ ,]\d{3})+|\d+)\.\d{2}'
    r'-?\)?'
)

# Column-heading words. They mark a HEADER row only when the line does not
# start with a date: a real transaction such as "01 Feb Magtape Credit
# Medihelp 1,500.00" was thrown away as a header before 2026-09-30.
_HEADER_SUBSTRINGS = (
    'transaction date', 'posting date', 'description', 'money in', 'money out',
    'debit', 'credit',
)
# Never a transaction, dated or not.
_NON_TRANSACTION_SUBSTRINGS = (
    'balance', 'page ', 'continued', 'statement period',
    'account number', 'branch', 'brought forward', 'carried forward',
)
_SKIP_SUBSTRINGS = _HEADER_SUBSTRINGS + _NON_TRANSACTION_SUBSTRINGS


def _should_skip_line(low: str, dated: bool = False) -> bool:
    if not low.strip():
        return True
    words = _NON_TRANSACTION_SUBSTRINGS if dated else _SKIP_SUBSTRINGS
    return any(s in low for s in words if len(low) < 120 or s in low[:80])


# Dates printed WITH a year anywhere on the statement (period header,
# statement date) — the evidence for the year of a year-less line.
_DATED_RE = re.compile(
    r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})\b')
_YEARLESS_RE = re.compile(r'^\d{1,2}\s+[A-Za-z]{3}$')


def _statement_window(text: str):
    """(start, end) for completing year-less dates: the year up to the LATEST
    dated line on the statement. Anchoring on the latest date means an old date
    printed elsewhere (an account-opening date) cannot pull the year back."""
    found = []
    for m in _DATED_RE.finditer(text or ''):
        try:
            found.append(date.fromisoformat(normalize_date(m.group(1))))
        except ValueError:
            continue
    if not found:
        return None, None
    end = max(found)
    return (end - timedelta(days=364)).isoformat(), end.isoformat()


def _complete_leading_date(raw_line: str, start, end) -> str:
    """Rewrite a leading year-less date as ISO so every parser can read it."""
    m = _DATE_RE.match(raw_line)
    if not m or not _YEARLESS_RE.match(m.group(1).strip()):
        return raw_line
    try:
        iso = resolve_date_with_period(m.group(1), period_start=start, period_end=end)
    except ValueError:
        return raw_line
    return raw_line[:m.start(1)] + iso + raw_line[m.end(1):]


def _parse_generic_line(raw_line: str) -> Optional[StatementLine]:
    m = _DATE_RE.match(raw_line)
    if not m:
        return None
    date_token, remainder = m.group(1), m.group(2)
    try:
        iso_date = normalize_date(date_token)
    except ValueError:
        return None
    amounts = _AMOUNT_RE.findall(remainder)
    if not amounts:
        return None
    if len(amounts) >= 2:
        amount_token, balance_token = amounts[-2], amounts[-1]
    else:
        amount_token, balance_token = amounts[-1], None
    try:
        amount = normalize_amount(amount_token)
    except ValueError:
        return None
    balance = None
    if balance_token is not None:
        try:
            balance = normalize_amount(balance_token)
        except ValueError:
            balance = None
    first_amt_pos = remainder.find(amounts[0])
    description = remainder[:first_amt_pos].strip() or remainder.strip()
    if not description or _should_skip_line(description.lower(), dated=True):
        return None
    return StatementLine(date=iso_date, description=description, amount=amount, balance=balance)


def _parse_capitec_line(raw_line: str) -> Optional[StatementLine]:
    """Capitec: Date | Description | [Fee] | Money In | Money Out | Balance."""
    m = _DATE_RE.match(raw_line)
    if not m:
        return None
    date_token, remainder = m.group(1), m.group(2)
    try:
        iso_date = normalize_date(date_token)
    except ValueError:
        return None
    amounts = _AMOUNT_RE.findall(remainder)
    if not amounts:
        return None
    # Capitec often ends with Money In, Money Out, Balance (3 amounts) or Fee+In+Out+Bal (4)
    if len(amounts) >= 3:
        balance_token = amounts[-1]
        out_token = amounts[-2]
        in_token = amounts[-3]
        try:
            money_in = normalize_amount(in_token)
            money_out = normalize_amount(out_token)
            balance = normalize_amount(balance_token)
        except ValueError:
            return _parse_generic_line(raw_line)
        if money_in != 0 and money_out != 0:
            return _parse_generic_line(raw_line)
        amount = money_in if money_in != 0 else -abs(money_out)
        first_amt_pos = remainder.find(amounts[0])
        description = remainder[:first_amt_pos].strip()
    elif len(amounts) == 2:
        try:
            amount = normalize_amount(amounts[-2])
            balance = normalize_amount(amounts[-1])
        except ValueError:
            return None
        first_amt_pos = remainder.find(amounts[0])
        description = remainder[:first_amt_pos].strip()
    else:
        return _parse_generic_line(raw_line)
    if not description or _should_skip_line(description.lower(), dated=True):
        return None
    return StatementLine(date=iso_date, description=description, amount=amount, balance=balance)


def _parse_fnb_line(raw_line: str) -> Optional[StatementLine]:
    """FNB: Date | Description | Amount | [Balance] — amounts often trailing-minus."""
    return _parse_generic_line(raw_line)


def _parse_standard_bank_line(raw_line: str) -> Optional[StatementLine]:
    """Standard Bank: may label Payments/Deposits — use signed single amount."""
    low = raw_line.lower()
    if 'payment' in low and 'deposit' not in low[:20]:
        pass  # still one amount column in export
    return _parse_generic_line(raw_line)


def _parse_absa_line(raw_line: str) -> Optional[StatementLine]:
    return _parse_generic_line(raw_line)


def _parse_nedbank_line(raw_line: str) -> Optional[StatementLine]:
    return _parse_generic_line(raw_line)


@dataclass(frozen=True)
class BankProfile:
    profile_id: str
    display_name: str
    hints: Sequence[str]
    parse_line: Callable[[str], Optional[StatementLine]]
    opening_labels: Sequence[str] = field(default_factory=lambda: (
        'opening balance', 'balance brought forward', 'balance b/f', 'balance b/fwd',
        'opening bal', 'previous balance', 'bal b/f',
    ))
    closing_labels: Sequence[str] = field(default_factory=lambda: (
        'closing balance', 'balance carried forward', 'balance c/f', 'balance c/fwd',
        'closing bal', 'available balance', 'bal c/f',
    ))


GENERIC = BankProfile(
    profile_id='generic',
    display_name='Generic SA Bank',
    hints=(),
    parse_line=_parse_generic_line,
)

FNB = BankProfile(
    profile_id='fnb',
    display_name='FNB',
    hints=('fnb', 'first national bank', 'fnb.co.za'),
    parse_line=_parse_fnb_line,
)

CAPITEC = BankProfile(
    profile_id='capitec',
    display_name='Capitec',
    hints=('capitec', 'capitec bank'),
    parse_line=_parse_capitec_line,
)

STANDARD_BANK = BankProfile(
    profile_id='standard_bank',
    display_name='Standard Bank',
    hints=('standard bank', 'standardbank', 'sbsa'),
    parse_line=_parse_standard_bank_line,
)

ABSA = BankProfile(
    profile_id='absa',
    display_name='Absa',
    hints=('absa', 'absa bank'),
    parse_line=_parse_absa_line,
)

NEDBANK = BankProfile(
    profile_id='nedbank',
    display_name='Nedbank',
    hints=('nedbank',),
    parse_line=_parse_nedbank_line,
)

# Order matters: more specific hints before generic fallback in detection.
ALL_PROFILES: List[BankProfile] = [
    CAPITEC, FNB, STANDARD_BANK, ABSA, NEDBANK, GENERIC,
]


def detect_profile(text: str) -> BankProfile:
    """Pick the best-matching bank profile for statement text."""
    low = text.lower()
    for profile in ALL_PROFILES:
        if profile is GENERIC:
            continue
        if any(hint in low for hint in profile.hints):
            return profile
    return GENERIC


def parse_transaction_lines(text: str, profile: Optional[BankProfile] = None) -> list[StatementLine]:
    """Parse all transaction lines using the given (or auto-detected) profile."""
    profile = profile or detect_profile(text)
    start, end = _statement_window(text)
    lines: list[StatementLine] = []
    for raw_line in text.splitlines():
        raw_line = _complete_leading_date(raw_line, start, end)
        if _should_skip_line(raw_line.lower(), dated=bool(_DATE_RE.match(raw_line))):
            continue
        parsed = profile.parse_line(raw_line)
        if parsed is not None:
            lines.append(parsed)
    return lines
