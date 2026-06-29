"""Statement Integrity — mathematical gate for bank-statement capture.

Validates: opening_balance + Σ(transaction amounts) == closing_balance.
Pure module (no Flask/DB). Adapted from BooksXperts/trusteasygo OCR bank-feed plan.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

CENTS = Decimal("0.01")
DEFAULT_CONFIDENCE_FLOOR = Decimal("0.80")

_SA_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d %B %Y",
    "%d-%b-%Y", "%d-%B-%Y",
)


def normalize_amount(raw) -> Decimal:
    """Parse SA bank money tokens into a signed Decimal (2dp)."""
    if raw is None:
        raise ValueError("empty amount")
    if isinstance(raw, (int, float, Decimal)):
        return Decimal(str(raw)).quantize(CENTS)

    s = str(raw).strip().replace("\xa0", " ")
    if not s:
        raise ValueError("empty amount")

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    if s.endswith("-"):
        negative = True
        s = s[:-1].strip()
    elif s.startswith("-"):
        negative = True
        s = s[1:].strip()

    low = s.lower()
    if low.endswith("dr"):
        negative = True
        s = s[:-2].strip()
    elif low.endswith("cr"):
        s = s[:-2].strip()

    s = s.replace("R", "").replace("r", "").replace("$", "").replace(" ", "").replace(",", "")
    if s in ("", "."):
        raise ValueError(f"unparseable amount: {raw!r}")
    try:
        value = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"unparseable amount: {raw!r}") from exc
    if negative:
        value = -value
    return value.quantize(CENTS)


def normalize_date(raw) -> str:
    """Parse an SA bank date to ISO YYYY-MM-DD."""
    if raw is None:
        raise ValueError("empty date")
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    s = str(raw).strip()
    if not s:
        raise ValueError("empty date")
    for fmt in _SA_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date format: {raw!r}")


def _normalize_description(desc: str) -> str:
    return re.sub(r"\s+", " ", (desc or "").strip()).upper()


class StatementLine(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    date: str
    description: str
    amount: Decimal
    balance: Optional[Decimal] = None
    confidence: Decimal = Decimal("1.0")

    @field_validator("date", mode="before")
    @classmethod
    def _v_date(cls, v):
        return normalize_date(v)

    @field_validator("amount", "balance", mode="before")
    @classmethod
    def _v_money(cls, v):
        if v is None:
            return None
        return normalize_amount(v)

    def fingerprint(self) -> str:
        key = f"{self.date}|{_normalize_description(self.description)}|{self.amount}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class StatementHeader(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    bank: Optional[str] = None
    account_number: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    # Optional: a statement (or scan) may not expose machine-readable opening/
    # closing balances. When absent we still capture the lines; the math gate is
    # reported as "not verified" rather than failing the whole extraction.
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None

    @field_validator("opening_balance", "closing_balance", mode="before")
    @classmethod
    def _v_money(cls, v):
        return normalize_amount(v) if v not in (None, "") else None

    @field_validator("period_start", "period_end", mode="before")
    @classmethod
    def _v_date(cls, v):
        return normalize_date(v) if v not in (None, "") else None


class ExtractionResult(BaseModel):
    header: StatementHeader
    lines: list[StatementLine] = Field(default_factory=list)


class ReportCard(BaseModel):
    reconciled: bool
    status: str
    # declared_net/variance are None when opening or closing balance is unknown
    # (math gate could not be evaluated). computed_net is always the sum of lines.
    balances_provided: bool = True
    declared_net: Optional[Decimal] = None
    computed_net: Decimal
    variance: Optional[Decimal] = None
    line_count: int
    duplicate_count: int = 0
    duplicate_fingerprints: list[str] = Field(default_factory=list)
    running_balance_breaks: list[dict] = Field(default_factory=list)
    low_confidence_lines: list[dict] = Field(default_factory=list)
    continuity_ok: Optional[bool] = None
    errors: list[dict] = Field(default_factory=list)


def self_audit(
    result: ExtractionResult,
    prior_closing: Optional[Decimal] = None,
    confidence_floor: Decimal = DEFAULT_CONFIDENCE_FLOOR,
) -> ReportCard:
    """Run capture-integrity checks; return a ReportCard verdict."""
    header = result.header
    lines = result.lines

    computed_net = sum((ln.amount for ln in lines), Decimal("0.00")).quantize(CENTS)

    # The math gate (opening + Σ == closing) can only run when BOTH header
    # balances are known. If either is missing, capture the lines anyway and
    # report the gate as "not verified" instead of failing extraction.
    balances_provided = (
        header.opening_balance is not None and header.closing_balance is not None
    )
    declared_net: Optional[Decimal] = None
    variance: Optional[Decimal] = None

    errors: list[dict] = []
    if balances_provided:
        declared_net = (header.closing_balance - header.opening_balance).quantize(CENTS)
        variance = (declared_net - computed_net).quantize(CENTS)
        if variance != Decimal("0.00"):
            errors.append({
                "code": "MATH_GATE_FAILED",
                "message": (
                    "Statement does not balance: opening + sum(transactions) "
                    "does not equal closing."
                ),
                "declared_net": str(declared_net),
                "computed_net": str(computed_net),
                "variance": str(variance),
            })

    running_breaks: list[dict] = []
    prev_balance: Optional[Decimal] = header.opening_balance
    for idx, ln in enumerate(lines):
        if ln.balance is None:
            prev_balance = None
            continue
        if prev_balance is not None:
            expected = (prev_balance + ln.amount).quantize(CENTS)
            if expected != ln.balance:
                running_breaks.append({
                    "line_index": idx,
                    "date": ln.date,
                    "description": ln.description,
                    "expected_balance": str(expected),
                    "stated_balance": str(ln.balance),
                    "difference": str((ln.balance - expected).quantize(CENTS)),
                })
        prev_balance = ln.balance
    if running_breaks:
        errors.append({
            "code": "RUNNING_BALANCE_BREAK",
            "message": f"{len(running_breaks)} line(s) break the running balance.",
            "count": len(running_breaks),
        })

    continuity_ok: Optional[bool] = None
    if prior_closing is not None and header.opening_balance is not None:
        prior_closing = Decimal(prior_closing).quantize(CENTS)
        continuity_ok = (header.opening_balance == prior_closing)
        if not continuity_ok:
            errors.append({
                "code": "CONTINUITY_GAP",
                "message": (
                    "Opening balance does not match the previous statement's "
                    "closing balance."
                ),
                "prior_closing": str(prior_closing),
                "this_opening": str(header.opening_balance),
                "gap": str((header.opening_balance - prior_closing).quantize(CENTS)),
            })

    seen: dict[str, int] = {}
    dup_prints: list[str] = []
    for ln in lines:
        fp = ln.fingerprint()
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] == 2:
            dup_prints.append(fp)
    duplicate_count = sum(c - 1 for c in seen.values() if c > 1)
    if duplicate_count:
        errors.append({
            "code": "DUPLICATE_LINES",
            "message": f"{duplicate_count} duplicate transaction line(s) detected.",
            "count": duplicate_count,
        })

    low_conf: list[dict] = []
    for idx, ln in enumerate(lines):
        if ln.confidence < confidence_floor:
            low_conf.append({
                "line_index": idx,
                "date": ln.date,
                "description": ln.description,
                "confidence": str(ln.confidence),
            })
    if low_conf:
        errors.append({
            "code": "LOW_CONFIDENCE",
            "message": (
                f"{len(low_conf)} line(s) below the {confidence_floor} confidence "
                "floor — human review required."
            ),
            "count": len(low_conf),
        })

    if not balances_provided:
        # Lines captured, but the opening/closing math gate could not be run.
        reconciled = False
        status = "unverified"
    else:
        reconciled = (variance == Decimal("0.00")) and not running_breaks and not low_conf
        status = "reconciled" if reconciled else "unreconciled"
    return ReportCard(
        reconciled=reconciled,
        status=status,
        balances_provided=balances_provided,
        declared_net=declared_net,
        computed_net=computed_net,
        variance=variance,
        line_count=len(lines),
        duplicate_count=duplicate_count,
        duplicate_fingerprints=dup_prints,
        running_balance_breaks=running_breaks,
        low_confidence_lines=low_conf,
        continuity_ok=continuity_ok,
        errors=errors,
    )
