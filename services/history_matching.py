"""History-first matching — reuse what this practice already decided.

For a bookkeeping practice the strongest predictor of how a line should be
filed is not a model: it is *what the same payee was filed under last month*.
That signal is free, instant, deterministic, and it compounds — every row the
accountant completes makes the next statement more automatic.

This module builds an index of the user's already-completed rows and matches
new rows against it **before** any AI call is made, so a recurring payee costs
nothing and cannot be re-interpreted differently each month.

How a description is matched
----------------------------
SA bank narrations carry a stable payee plus per-transaction noise::

    "Magtape Credit Medihelp Smh0363383 20210204"
    "Card Purchase Checkers Sandton 4021"

:func:`normalize_description` drops every token containing a digit (references,
card fragments, dates, branch codes) and lowercases the rest, leaving
``"magtape credit medihelp"`` — the part that actually identifies the payee.
Exact match on that key is an O(1) dict lookup, not the O(n) SequenceMatcher
scan the old Recall did on every keystroke.

Ambiguity is refused, not averaged
----------------------------------
If the same payee has historically been filed to more than one account, the
match is only trusted when one account clearly dominates. A genuinely split
payee (a card used for two purposes) returns a low confidence so the caller
leaves it for the accountant instead of silently picking the more common one.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from models import Transaction, db

logger = logging.getLogger(__name__)

#: Rows whose payee has been seen this many times, always filed the same way,
#: are treated as settled practice.
_REPEAT_CONFIDENCE = 0.97
#: Seen once before, unambiguous.
_SINGLE_CONFIDENCE = 0.90
#: Seen several times but not always the same account; one account dominates.
_DOMINANT_CONFIDENCE = 0.86
#: Share of occurrences one account needs before it counts as dominant.
_DOMINANCE_RATIO = 0.8
#: Genuinely split payee — surfaced, never auto-applied.
_AMBIGUOUS_CONFIDENCE = 0.40

#: A key shorter than this is too generic to identify a payee ("payment", "fee").
_MIN_KEY_LENGTH = 4

_NOISE_TOKENS = frozenset({
    'the', 'and', 'for', 'ref', 'reference', 'payment', 'pmt', 'trf', 'transfer',
})


def normalize_description(text: Optional[str]) -> str:
    """Reduce a bank narration to the part that identifies the payee.

    Drops tokens containing digits (reference numbers, dates, card fragments)
    and generic noise words, lowercases, and collapses whitespace. Returns ''
    when nothing identifying survives, which callers must treat as "no key".
    """
    if not text:
        return ''
    cleaned = re.sub(r'[^\w\s]', ' ', str(text).lower())
    tokens = []
    for token in cleaned.split():
        if any(ch.isdigit() for ch in token):
            continue
        if token in _NOISE_TOKENS:
            continue
        if len(token) < 2:
            continue
        tokens.append(token)
    key = ' '.join(tokens).strip()
    return key if len(key) >= _MIN_KEY_LENGTH else ''


@dataclass
class HistoryEntry:
    """What the practice has previously done with one payee."""
    key: str
    account_counts: Dict[int, int] = field(default_factory=dict)
    account_names: Dict[int, str] = field(default_factory=dict)
    explanation: str = ''
    total: int = 0

    def verdict(self) -> tuple[Optional[int], Optional[str], float]:
        """(account_id, account_name, confidence) for this payee."""
        if not self.account_counts:
            return None, None, 0.0
        best_id, best_count = max(self.account_counts.items(), key=lambda kv: kv[1])
        name = self.account_names.get(best_id)

        if len(self.account_counts) == 1:
            confidence = _REPEAT_CONFIDENCE if best_count > 1 else _SINGLE_CONFIDENCE
            return best_id, name, confidence

        if best_count / self.total >= _DOMINANCE_RATIO:
            return best_id, name, _DOMINANT_CONFIDENCE

        # Genuinely split: surface the most common one but well below any
        # sane auto-apply gate, so a human decides.
        return best_id, name, _AMBIGUOUS_CONFIDENCE


def build_history_index(
    user_id: int,
    exclude_file_id: Optional[int] = None,
) -> Dict[str, HistoryEntry]:
    """Index this user's completed rows by normalised payee.

    Only rows carrying BOTH an account and an explanation count as settled
    practice. One query builds the whole index, so a batch pays for it once.
    ``exclude_file_id`` keeps the statement being processed from teaching
    itself off its own half-finished rows.
    """
    query = (
        db.session.query(
            Transaction.description,
            Transaction.account_id,
            Transaction.explanation,
        )
        .filter(
            Transaction.user_id == user_id,
            Transaction.account_id.isnot(None),
            Transaction.explanation.isnot(None),
            Transaction.explanation != '',
        )
    )
    if exclude_file_id is not None:
        query = query.filter(
            (Transaction.file_id.is_(None)) | (Transaction.file_id != exclude_file_id))

    index: Dict[str, HistoryEntry] = {}
    names = _account_names(user_id)

    for description, account_id, explanation in query.all():
        key = normalize_description(description)
        if not key:
            continue
        entry = index.get(key)
        if entry is None:
            entry = HistoryEntry(key=key)
            index[key] = entry
        entry.total += 1
        entry.account_counts[account_id] = entry.account_counts.get(account_id, 0) + 1
        if account_id in names:
            entry.account_names[account_id] = names[account_id]
        if not entry.explanation and explanation:
            entry.explanation = explanation.strip()[:500]

    logger.info("History index for user %s: %d distinct payee(s)", user_id, len(index))
    return index


def _account_names(user_id: int) -> Dict[int, str]:
    from models import Account
    return {
        account.id: account.name
        for account in Account.query.filter_by(user_id=user_id).all()
    }


def match_rows(
    rows: Sequence[Dict[str, Any]],
    index: Dict[str, HistoryEntry],
):
    """Match rows against the history index.

    Returns ``{row index: RowSuggestion}`` using the same shape as the AI
    suggester, so the caller can merge the two without special-casing either.
    Rows with no history match are simply absent.
    """
    from services.bulk_suggestions import RowSuggestion

    matches = {}
    if not index:
        return matches

    for row in rows:
        key = normalize_description(row.get('description'))
        if not key:
            continue
        entry = index.get(key)
        if entry is None:
            continue
        account_id, account_name, confidence = entry.verdict()
        if account_id is None:
            continue
        matches[row['index']] = RowSuggestion(
            index=row['index'],
            account_id=account_id,
            account_name=account_name,
            confidence=confidence,
            explanation=entry.explanation,
        )
    return matches
