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
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import List, Dict, Optional

from config import CLAUDE_MODEL
from nlp_utils import get_claude_client

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

CONFIDENCE_FLOOR = 0.80  # rows below this are flagged even when math balances

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


# --- Phase 2: PDF bank statements -----------------------------------------

_STATEMENT_PROMPT = (
    "You are a precise bank-statement reader. Extract EVERY transaction line from "
    "this bank statement. Respond with ONLY a JSON array and nothing else.\n"
    "Each element: {\"date\": \"YYYY-MM-DD\", \"description\": \"<narration or merchant>\", "
    "\"amount\": <number>, \"confidence\": <number 0..1>}.\n"
    "Sign convention: amount is NEGATIVE for money leaving the account (debits, "
    "withdrawals, payments) and POSITIVE for money entering (credits, deposits).\n"
    "Use each transaction's own date. Do NOT include opening/closing balances or "
    "summary totals. Do not invent rows; lower the confidence when unsure."
)


def extract_statement(pdf_bytes: bytes, client=None) -> List[Dict]:
    """Send a PDF bank statement to Claude (document block) and return raw rows.

    Never raises. ``client`` may be injected for testing.
    """
    client = client or get_claude_client()
    if not client:
        logger.error("OCR: AI client unavailable for statement extraction")
        return []
    try:
        encoded = base64.standard_b64encode(pdf_bytes).decode('ascii')
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'document',
                        'source': {
                            'type': 'base64',
                            'media_type': 'application/pdf',
                            'data': encoded,
                        },
                    },
                    {'type': 'text', 'text': _STATEMENT_PROMPT},
                ],
            }],
        )
        text = response.content[0].text.strip()
        return parse_rows(text)
    except Exception as e:
        logger.error(f"OCR statement extraction error: {str(e)}")
        return []


def extract_and_normalize_statement(pdf_bytes: bytes, client=None) -> List[Dict]:
    """Convenience: extract a PDF statement then normalize with signed amounts."""
    return normalize_rows(extract_statement(pdf_bytes, client=client), signed=True)


# --- Phase 2.2: header extraction + Math Integrity Gate -------------------

_STATEMENT_WITH_HEADER_PROMPT = (
    "You are a precise bank-statement reader. Extract the opening balance, closing balance, "
    "and EVERY transaction from this bank statement. "
    "Respond with ONLY a JSON object and nothing else, in this exact shape:\n"
    '{"opening_balance": <number or null>, "closing_balance": <number or null>, '
    '"transactions": [{"date": "YYYY-MM-DD", "description": "<narration>", '
    '"amount": <number>, "confidence": <number 0..1>}]}\n'
    "Sign convention for transactions: NEGATIVE for money leaving the account "
    "(debits, withdrawals, payments), POSITIVE for money entering (credits, deposits).\n"
    "opening_balance and closing_balance are the figures stated on the statement itself "
    "(positive for credit balance). Set to null if not clearly stated.\n"
    "Use each transaction's own date. Do not invent rows; lower confidence when unsure."
)


def _parse_balance(value) -> Optional[float]:
    """Parse a balance figure (string or number) to a rounded float, or None."""
    if value is None:
        return None
    try:
        cleaned = str(value).replace(',', '').replace('R', '').replace(' ', '').strip()
        return round(float(cleaned), 2)
    except (TypeError, ValueError):
        return None


def extract_statement_with_header(pdf_bytes: bytes, client=None):
    """Extract a PDF statement returning (opening_balance, closing_balance, rows).

    Returns a 3-tuple (float|None, float|None, List[Dict]).
    rows carries signed amounts, normalised like extract_and_normalize_statement.
    Never raises — returns (None, None, []) on any failure.
    """
    client = client or get_claude_client()
    if not client:
        logger.error("OCR: AI client unavailable for statement-with-header extraction")
        return None, None, []
    try:
        encoded = base64.standard_b64encode(pdf_bytes).decode('ascii')
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'document',
                        'source': {
                            'type': 'base64',
                            'media_type': 'application/pdf',
                            'data': encoded,
                        },
                    },
                    {'type': 'text', 'text': _STATEMENT_WITH_HEADER_PROMPT},
                ],
            }],
        )
        text = response.content[0].text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            logger.error("OCR: no JSON object in statement-with-header response")
            return None, None, []
        data = json.loads(text[start:end + 1])
        opening = _parse_balance(data.get('opening_balance'))
        closing = _parse_balance(data.get('closing_balance'))
        raw_rows = data.get('transactions', [])
        rows = normalize_rows(raw_rows if isinstance(raw_rows, list) else [], signed=True)
        return opening, closing, rows
    except Exception as e:
        logger.error(f"OCR statement-with-header error: {str(e)}")
        return None, None, []


def audit_statement(rows, opening_balance, closing_balance) -> dict:
    """Run the Math Integrity Gate against extracted statement rows.

    Returns a report_card dict with keys:
      has_balances, reconciled, opening_balance, closing_balance,
      declared_net, computed_net, variance, low_confidence_count,
      low_confidence_indexes, flags.

    IFRS for SMEs §2.5 (faithful representation): captured data must equal the source.
    """
    CENTS = Decimal('0.01')
    flags = []
    has_balances = opening_balance is not None and closing_balance is not None

    try:
        computed = sum(
            Decimal(str(r.get('amount') or 0)).quantize(CENTS)
            for r in rows
        )
    except (TypeError, ValueError, InvalidOperation):
        computed = Decimal('0.00')

    computed_net = float(computed)
    declared_net = None
    variance = None
    reconciled = False

    if has_balances:
        try:
            ob = Decimal(str(opening_balance)).quantize(CENTS)
            cb = Decimal(str(closing_balance)).quantize(CENTS)
            declared_net_d = (cb - ob).quantize(CENTS)
            variance_d = (declared_net_d - computed).quantize(CENTS)
            declared_net = float(declared_net_d)
            variance = float(variance_d)
            reconciled = (variance_d == Decimal('0.00'))
            if not reconciled:
                flags.append(
                    f"Math gate failed: opening + transactions ≠ closing "
                    f"(variance R{variance:+.2f})."
                )
        except (TypeError, ValueError, InvalidOperation):
            flags.append("Could not verify math gate (balance parse error).")

    low_confidence_indexes = [
        i for i, r in enumerate(rows)
        if (r.get('confidence') or 0.0) < CONFIDENCE_FLOOR
    ]
    low_confidence_count = len(low_confidence_indexes)
    if low_confidence_count:
        flags.append(
            f"{low_confidence_count} row(s) below {int(CONFIDENCE_FLOOR * 100)}% confidence "
            "— review before importing."
        )
        reconciled = False

    return {
        'has_balances': has_balances,
        'reconciled': reconciled,
        'opening_balance': opening_balance,
        'closing_balance': closing_balance,
        'declared_net': declared_net,
        'computed_net': computed_net,
        'variance': variance,
        'low_confidence_count': low_confidence_count,
        'low_confidence_indexes': low_confidence_indexes,
        'flags': flags,
    }


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
