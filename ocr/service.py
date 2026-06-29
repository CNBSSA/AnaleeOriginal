"""Bank-statement OCR service.

Analee is strictly a cash-basis tool, so this module covers ONLY PDF bank
statements (receipt/image OCR was removed). It exposes:

- a thin ``extract_and_normalize_statement`` wrapper over the full SA pipeline in
  :mod:`ocr.statement_extractor` (Tier-1 digital PDF + Tier-2 Claude Vision); and
- ``mark_duplicates``, a pure helper that flags rows already present in the user's
  transactions, so the review screen can pre-exclude them.
"""
import logging
from difflib import SequenceMatcher
from typing import List, Dict, Optional

from .statement_extractor import extract_bank_statement, BankStatementExtraction

logger = logging.getLogger(__name__)

# Accepted document extensions -> Anthropic media types (PDF bank statements).
ALLOWED_DOCUMENT_TYPES = {
    'pdf': 'application/pdf',
}

MAX_PDF_BYTES = 32 * 1024 * 1024    # 32 MB (Anthropic PDF request limit)


def extract_and_normalize_statement(
    pdf_bytes: bytes,
    client=None,
    opening_balance: Optional[str] = None,
    closing_balance: Optional[str] = None,
) -> List[Dict]:
    """Extract a PDF bank statement via the full SA pipeline.

    Returns review-ready row dicts, or [] on failure (legacy contract for callers
    that only check emptiness). Prefer :func:`extract_bank_statement` when you
    need error detail and the integrity report card.
    """
    outcome = extract_bank_statement(
        pdf_bytes,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        client=client,
    )
    if not outcome.ok:
        if outcome.error:
            logger.error("OCR statement extraction: %s", outcome.error)
        return []
    return outcome.rows


# --- duplicate flagging ----------------------------------------------------

_DUPLICATE_SIMILARITY = 0.85


def _descriptions_match(a: str, b: str) -> bool:
    """True if two descriptions are close enough to be the same transaction."""
    a = (a or '').lower().strip()
    b = (b or '').lower().strip()
    if not a or not b:
        # Same date+amount with a missing description on either side -> treat as a
        # likely duplicate (conservative; the user can still re-include it).
        return True
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= _DUPLICATE_SIMILARITY


def mark_duplicates(rows: List[Dict], existing: List) -> List[Dict]:
    """Set ``row['duplicate']`` on each row that likely matches an existing row.

    ``existing`` is an iterable of (date_str 'YYYY-MM-DD', amount float,
    description str) for the user's already-stored transactions. A row is flagged
    when an existing entry shares the same date and amount (to the cent) and a
    matching description. Pure function — no DB access — so it is easily tested.
    """
    index: Dict = {}
    for date_str, amount, description in existing:
        try:
            key = (date_str, round(float(amount), 2))
        except (TypeError, ValueError):
            continue
        index.setdefault(key, []).append(description or '')

    for row in rows:
        row['duplicate'] = False
        date_str = row.get('date')
        amount = row.get('amount')
        if not date_str or amount is None:
            continue
        candidates = index.get((date_str, round(float(amount), 2)))
        if not candidates:
            continue
        description = row.get('description') or ''
        if any(_descriptions_match(description, candidate) for candidate in candidates):
            row['duplicate'] = True
    return rows
