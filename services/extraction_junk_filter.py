"""Filter non-transaction rows pulled in during statement import/extraction.

SA bank PDFs and exports often include zero-amount notice lines (scheduled-
transaction reminders, available-balance summaries, etc.). Those are not money
movement. This module classifies extracted rows conservatively:

- Non-zero amounts are always kept.
- Zero amounts matching definite notice patterns are dropped.
- Other zero amounts are flagged for review (never silently dropped).
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

_ZERO_EPS = Decimal("0.005")

# High-confidence notice / summary lines (zero amount only).
_DEFINITE_JUNK_RE = re.compile(
    r"|".join(
        (
            r"schd(?:uled)?\s+trxn",
            r"scheduled\s+trxn",
            r"scheduled\s+trans(?:action)?",
            r"\bno\s+av(?:ail)?\.?\s*bal\b",
            r"\bavail(?:able)?\s+bal(?:ance)?\b",
            r"\bavailable\s+to\s+spend\b",
            r"\bbalance\s+summary\b",
            r"\bstatement\s+summary\b",
            r"\bpage\s+\d+\s+of\s+\d+\b",
            r"\bbrought\s+forward\b",
            r"\bcarried\s+forward\b",
            r"\bopening\s+bal(?:ance)?\b",
            r"\bclosing\s+bal(?:ance)?\b",
        )
    ),
    re.IGNORECASE,
)


class JunkVerdict(str, Enum):
    KEEP = "keep"
    DROP = "drop"
    FLAG = "flag"


def _normalize_description(description: str | None) -> str:
    text = " ".join(str(description or "").split()).lower()
    return text.strip()


def amount_is_zero(amount: Any) -> bool:
    """True when the amount is missing, unreadable, or effectively zero."""
    if amount is None:
        return True
    try:
        value = amount if isinstance(amount, Decimal) else Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        return True
    return abs(value) < _ZERO_EPS


def classify_extraction_row(description: str | None, amount: Any) -> JunkVerdict:
    """Classify one extracted/import row."""
    if not amount_is_zero(amount):
        return JunkVerdict.KEEP

    desc = _normalize_description(description)
    if not desc:
        return JunkVerdict.FLAG

    if _DEFINITE_JUNK_RE.search(desc):
        return JunkVerdict.DROP

    return JunkVerdict.FLAG


def filter_review_rows(rows: List[MutableMapping[str, Any]]) -> tuple[List[MutableMapping[str, Any]], int]:
    """Drop definite junk; flag other zero-amount rows for the OCR review screen."""
    kept: list[MutableMapping[str, Any]] = []
    dropped = 0
    for row in rows:
        verdict = classify_extraction_row(row.get("description"), row.get("amount"))
        if verdict is JunkVerdict.DROP:
            dropped += 1
            continue
        row.setdefault("duplicate", False)
        row["statement_notice"] = verdict is JunkVerdict.FLAG
        kept.append(row)
    return kept, dropped


def filter_normalized_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Filter normalized CSV/Excel rows before persisting transactions."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    flagged = 0
    for row in rows:
        verdict = classify_extraction_row(row.get("Description"), row.get("Amount"))
        if verdict is JunkVerdict.DROP:
            dropped += 1
            continue
        if verdict is JunkVerdict.FLAG:
            flagged += 1
        kept.append(dict(row))
    return kept, dropped, flagged
