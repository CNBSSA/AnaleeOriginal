"""Batched account + explanation suggestions — one AI call for many rows.

Why this exists
---------------
``PredictiveFeatures.suggest_account`` makes **one Claude call per transaction**
and puts the user's ENTIRE chart of accounts in every prompt. A seeded SA chart
carries ~1 000 accounts, so a 160-row statement sent roughly 160 copies of that
chart. This module sends one call per *batch* of rows instead, which is the
same work at a fraction of the cost and latency, and asks for the explanation
in the same response so a row comes back complete rather than categorised but
blank.

Rules this module keeps (all of them load-bearing)
-------------------------------------------------
* **Never invent an account.** Every suggested name is matched against the
  user's real chart; anything else is discarded, not "closest-matched".
* **Never guess when the AI is offline.** With no client we return nothing, so
  the caller applies nothing — the same rule as the ASF guard (57ee9a9),
  which the batch path never had.
* **Survive an imperfect reply.** A truncated or fenced response is salvaged
  row by row; one unreadable row never discards the rest.
* **Confidence is the model's own claim**, so it is clamped to [0, 1] and left
  for the caller to gate. This module applies nothing itself.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from config import CLAUDE_MODEL

logger = logging.getLogger(__name__)

#: Rows per AI call. Chosen so the reply stays well inside ``MAX_REPLY_TOKENS``
#: (~60 tokens/row) while still amortising the chart-of-accounts prompt.
BULK_SUGGESTION_BATCH = 25

MAX_REPLY_TOKENS = 4000

#: Hard cap on how much chart we will put in a prompt. A full SA chart is
#: ~1 000 accounts; sending all of them is what made the per-row design so
#: expensive. Batching already amortises it, but the cap bounds the worst case.
MAX_ACCOUNTS_IN_PROMPT = 400

_PROMPT = """You are a South African bookkeeper working on a cash-basis ledger.

For EACH numbered transaction below, choose the most appropriate account from
the chart, and write a short plain-English explanation of what the transaction
is (one sentence, no more than 200 characters).

Chart of accounts (choose ONLY from these exact names):
{chart}

Transactions:
{rows}

Respond with JSON ONLY — no markdown fences, no commentary — exactly this shape:
[
  {{"i": <the transaction number>,
    "account": "<exact account name from the chart, or null if genuinely unsure>",
    "confidence": <0.0 to 1.0 — your certainty in the ACCOUNT choice>,
    "explanation": "<short plain-English explanation>"}}
]

Rules:
- Use the exact account name as printed in the chart. Never invent one.
- A negative amount is money OUT, a positive amount is money IN. Never file a
  payment to an income account or a receipt to an expense account.
- If you are genuinely unsure of the account, set "account" to null and give a
  low confidence — do not guess. Still write the explanation.
- Return one object for EVERY transaction number listed.
"""


@dataclass
class RowSuggestion:
    """One row's suggestion. ``account_id`` is set only for a validated match."""
    index: int
    account_id: Optional[int] = None
    account_name: Optional[str] = None
    confidence: float = 0.0
    explanation: str = ''


def _format_chart(accounts: Sequence) -> str:
    lines = []
    for account in accounts[:MAX_ACCOUNTS_IN_PROMPT]:
        lines.append(f"- {account.name} ({account.category})")
    return "\n".join(lines)


def _format_rows(rows: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        date = row.get('date') or ''
        amount = row.get('amount')
        lines.append(
            f"{row['index']}. {date} | {row.get('description', '')} | {amount}"
        )
    return "\n".join(lines)


def _salvage_objects(text: str) -> List[dict]:
    """Recover every complete ``{...}`` object from a possibly truncated reply.

    Same lesson as the statement extractor: a reply cut off by the token
    ceiling must not throw away the rows that arrived intact.
    """
    objects: List[dict] = []
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(text[start:i + 1]))
                except ValueError:
                    pass
                start = None
    return objects


def _parse_reply(text: str) -> List[dict]:
    text = (text or '').strip()
    if text.startswith('```'):
        parts = text.split('```')
        text = parts[1].lstrip('json').strip() if len(parts) > 1 else text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except ValueError:
        pass
    salvaged = _salvage_objects(text)
    if salvaged:
        logger.warning(
            "Bulk suggestion reply was not valid JSON (likely truncated) — "
            "salvaged %d row(s)", len(salvaged),
        )
    return salvaged


def _clamp_confidence(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def suggest_for_rows(
    rows: Sequence[Dict[str, Any]],
    accounts: Sequence,
    client=None,
) -> Dict[int, RowSuggestion]:
    """Suggest an account + explanation for each row in ONE AI call.

    ``rows`` are dicts with ``index``, ``description``, ``amount`` and
    optionally ``date``. ``accounts`` are this user's Account rows — the caller
    is responsible for scoping them, and only names present here can ever be
    returned.

    Returns a mapping of row index -> :class:`RowSuggestion`. An empty mapping
    means "no suggestions" — never a guess. The caller decides what confidence
    is high enough to act on.
    """
    if not rows or not accounts:
        return {}

    if client is None:
        from nlp_utils import get_claude_client
        client = get_claude_client()
    if client is None:
        # Offline: decline rather than fall back to text similarity. The batch
        # path used to consume PredictiveFeatures' SequenceMatcher fallback,
        # whose ratio could clear the auto-apply gate and silently post a
        # guessed account.
        logger.info("Bulk suggestions skipped: no AI client configured")
        return {}

    by_name = {account.name.lower(): account for account in accounts}

    prompt = _PROMPT.format(chart=_format_chart(accounts), rows=_format_rows(rows))

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_REPLY_TOKENS,
            system="You are a meticulous South African bookkeeper.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
    except Exception as exc:
        logger.error("Bulk suggestion call failed (%s: %s)", type(exc).__name__, exc)
        return {}

    valid_indexes = {row['index'] for row in rows}
    suggestions: Dict[int, RowSuggestion] = {}

    for item in _parse_reply(text):
        try:
            index = int(item.get('i'))
        except (TypeError, ValueError):
            continue
        if index not in valid_indexes:
            continue

        explanation = str(item.get('explanation') or '').strip()[:500]
        confidence = _clamp_confidence(item.get('confidence'))

        account_id = None
        account_name = None
        raw_name = item.get('account')
        if raw_name not in (None, '', 'null'):
            matched = by_name.get(str(raw_name).strip().lower())
            if matched is not None:
                account_id = matched.id
                account_name = matched.name
            else:
                # Hallucinated or renamed account: keep the explanation, drop
                # the account rather than fuzzy-matching onto something else.
                logger.info(
                    "Bulk suggestion named an account not in the chart: %r", raw_name)
                confidence = 0.0

        suggestions[index] = RowSuggestion(
            index=index,
            account_id=account_id,
            account_name=account_name,
            confidence=confidence,
            explanation=explanation,
        )

    return suggestions
