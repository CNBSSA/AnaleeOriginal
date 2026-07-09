"""Guard: Analee's chart stays reconciled with BooksXperts' for the TB handoff.

Analee exports a trial balance that BooksXperts imports BY LINK; the imported
row's subcategory (from BooksXperts' chart) drives AFS classification. If Analee
ever emits a link BooksXperts doesn't have, that row lands in SUSPENSE; if it
emits a link whose subcategory differs, the row is MISCLASSIFIED. This guard fails
on either kind of drift.

When BooksXperts' chart legitimately changes (a Festus-approved act), regenerate
the reference and commit it:
    <booksxpert>/.venv/bin/python scripts/refresh_booksxperts_chart_reference.py \
        --booksxpert <booksxpert checkout>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chart_reconciliation import (  # noqa: E402
    analee_chart, booksxperts_reference, reconcile,
)


def test_every_analee_link_exists_in_booksxperts():
    result = reconcile()
    assert result['missing_in_booksxperts'] == [], (
        'Analee can emit trial-balance links that BooksXperts has no account for — '
        'those rows would land in SUSPENSE on import:\n  '
        + '\n  '.join(result['missing_in_booksxperts'])
        + '\n\nAdd the accounts to BooksXperts (Festus-approved), then regenerate '
          'reconciliation/booksxperts_chart_reference.json.'
    )


def test_no_subcategory_mismatch_on_shared_links():
    result = reconcile()
    lines = [f'{link}: Analee[{a}] vs BooksXperts[{b}]'
             for link, a, b in result['subcategory_mismatch']]
    assert result['subcategory_mismatch'] == [], (
        'Analee and BooksXperts classify the same account link under DIFFERENT '
        'subcategories — a transmitted trial balance would be MISCLASSIFIED:\n  '
        + '\n  '.join(lines)
        + '\n\nReconcile the charts (Festus-approved), then regenerate the reference.'
    )


def test_reference_and_analee_chart_are_non_trivial():
    # Sanity: both sides actually loaded (guards against an empty reference silently
    # passing the checks above).
    assert len(booksxperts_reference()) > 100
    assert len(analee_chart()) > 100
