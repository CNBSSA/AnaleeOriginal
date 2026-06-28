"""
Receipt OCR extraction service (Phase 1).

Sends a receipt image to Claude (vision) and returns normalized line items in the
same {date, description, amount} shape the existing import path expects. All
functions are defensive: on any failure they return an empty list rather than
raising, mirroring the rest of the AI stack's fallback discipline.

The Claude call (``extract_receipt``) is isolated from the parsing/normalization
helpers so the latter can be unit-tested without any network access.
"""
import base64
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Optional

from config import CLAUDE_MODEL, OCR_MODEL
from nlp_utils import get_claude_client
from .statement_extractor import extract_bank_statement, BankStatementExtraction

logger = logging.getLogger(__name__)

# Accepted image extensions -> Anthropic media types.
ALLOWED_IMAGE_TYPES = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
    'gif': 'image/gif',
}

# Accepted document extensions -> Anthropic media types (Phase 2: PDF statements).
ALLOWED_DOCUMENT_TYPES = {
    'pdf': 'application/pdf',
}

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PDF_BYTES = 32 * 1024 * 1024    # 32 MB (Anthropic PDF request limit)

_EXTRACTION_PROMPT = (
    "You are a precise financial document reader. Extract the purchase line items "
    "from this receipt image. Respond with ONLY a JSON array and nothing else.\n"
    "Each element must be an object: "
    '{"date": "YYYY-MM-DD" or null, "description": "<item or merchant>", '
    '"amount": <positive number>, "confidence": <number 0..1>}.\n'
    "Use the receipt's transaction date for every row. If individual line items are "
    "unclear, return a single row for the receipt total. Amounts are positive numbers "
    "without currency symbols. Do not invent data; set confidence lower when unsure."
)


def extract_receipt(image_bytes: bytes, media_type: str, client=None) -> List[Dict]:
    """Call Claude vision and return the raw parsed rows (list of dicts).

    Never raises. ``client`` may be injected for testing; otherwise the shared
    Anthropic client is used. Returns [] if the client or response is unusable.
    """
    client = client or get_claude_client()
    if not client:
        logger.error("OCR: AI client unavailable (is ANTHROPIC_API_KEY set?)")
        return []
    try:
        encoded = base64.standard_b64encode(image_bytes).decode('ascii')
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': encoded,
                        },
                    },
                    {'type': 'text', 'text': _EXTRACTION_PROMPT},
                ],
            }],
        )
        text = response.content[0].text.strip()
        return parse_rows(text)
    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        return []


def parse_rows(text: str) -> List[Dict]:
    """Pull the JSON array out of a model response and parse it. Never raises."""
    if not text:
        return []
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end < start:
        logger.error("OCR: no JSON array found in model response")
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.error(f"OCR: JSON parse error: {str(e)}")
        return []
    return data if isinstance(data, list) else []


def normalize_rows(raw_rows: List[Dict], signed: bool = False) -> List[Dict]:
    """Coerce raw extracted rows into a clean, render-ready shape.

    Output rows: {'date': 'YYYY-MM-DD' or '', 'description': str,
    'amount': float, 'confidence': float in [0,1]}. Rows with neither a
    description nor a parseable amount are dropped.

    When ``signed`` is True the amount sign is preserved (bank statements have
    debits and credits); otherwise amounts are made positive (receipts).
    """
    normalized = []
    for row in raw_rows or []:
        if not isinstance(row, dict):
            continue
        description = str(row.get('description') or '').strip()
        amount = parse_amount(row.get('amount'), signed=signed)
        date_str = parse_date(row.get('date'))
        if not description and amount is None:
            continue
        try:
            confidence = max(0.0, min(1.0, float(row.get('confidence'))))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append({
            'date': date_str,
            'description': description,
            'amount': amount if amount is not None else 0.0,
            'confidence': confidence,
        })
    return normalized


def extract_and_normalize(image_bytes: bytes, media_type: str, client=None) -> List[Dict]:
    """Convenience: extract a receipt image then normalize (positive amounts)."""
    return normalize_rows(extract_receipt(image_bytes, media_type, client=client))


def parse_amount(value, signed: bool = False) -> Optional[float]:
    """Parse a currency-ish value to a float, or None if unparseable.

    Handles thousands separators, currency symbols and accounting-style
    parentheses for negatives. When ``signed`` is False the result is made
    positive (receipts); when True the sign is preserved (bank statements).
    """
    if value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.replace(',', '').replace('$', '').strip()
            negative = text.startswith('(') and text.endswith(')')
            text = text.strip('()').strip()
            if not text:
                return None
            number = float(text)
            if negative:
                number = -number
        else:
            number = float(value)
        if not signed:
            number = abs(number)
        return round(number, 2)
    except (TypeError, ValueError):
        return None


def parse_date(value) -> str:
    """Parse a date in several common formats to 'YYYY-MM-DD', or '' if unknown."""
    if not value:
        return ''
    text = str(value).strip()
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%d %b %Y', '%d %B %Y'):
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return ''


# --- Phase 2+: SA bank statements (Tier-1 digital PDF + Tier-2 Claude Vision) ---


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


# --- Phase 2.1: duplicate flagging ----------------------------------------

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
