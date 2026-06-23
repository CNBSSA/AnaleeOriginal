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

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

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


def normalize_rows(raw_rows: List[Dict]) -> List[Dict]:
    """Coerce raw extracted rows into a clean, render-ready shape.

    Output rows: {'date': 'YYYY-MM-DD' or '', 'description': str,
    'amount': float, 'confidence': float in [0,1]}. Rows with neither a
    description nor a parseable amount are dropped.
    """
    normalized = []
    for row in raw_rows or []:
        if not isinstance(row, dict):
            continue
        description = str(row.get('description') or '').strip()
        amount = parse_amount(row.get('amount'))
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
    """Convenience: extract then normalize."""
    return normalize_rows(extract_receipt(image_bytes, media_type, client=client))


def parse_amount(value) -> Optional[float]:
    """Parse a currency-ish value to a positive float, or None if unparseable."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(',', '').replace('$', '').strip()
            if not value:
                return None
        return round(abs(float(value)), 2)
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
