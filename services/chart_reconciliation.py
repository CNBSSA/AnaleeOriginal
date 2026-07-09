"""Keep Analee's chart reconciled with BooksXperts' — the trial-balance handoff.

Analee exports a trial balance that BooksXperts (and The Accountants) import via
`dataimports`, keyed on each account's LINK (see reports/trial_balance_service.py:
columns Link, Account Name, Amount). On import, a row's link maps to a BooksXperts
CompanyAccount whose SUBCATEGORY drives AFS classification. So for the handoff to
be correct, every link Analee can emit must:
  • exist in BooksXperts' chart (otherwise the row lands in SUSPENSE), and
  • carry the SAME subcategory there (otherwise the row is MISCLASSIFIED).

`reconciliation/booksxperts_chart_reference.json` is a committed snapshot of
BooksXperts' transmission contract (link -> subcategory). This module compares
Analee's own chart against it; the guard test fails on any drift. Regenerate the
reference with `scripts/refresh_booksxperts_chart_reference.py` when BooksXperts'
chart legitimately changes (a Festus-approved act) and commit the new JSON.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(_ROOT, 'reconciliation', 'booksxperts_chart_reference.json')


def analee_chart():
    """Analee's own chart as {link: subcategory}, from services/chart_seed_data.py.

    Tuple shape: (account_number, account_name, account_link, category, sub_category).
    Covers the common accounts plus every entity-specific account (a link should
    classify identically across entities; the last wins if ever duplicated)."""
    from services import chart_seed_data as seed
    out = {}
    for t in seed.COMMON_ACCOUNTS:
        out[t[2]] = t[4]
    for _entity, rows in seed.ENTITY_SPECIFIC.items():
        for t in rows:
            out[t[2]] = t[4]
    return out


def booksxperts_reference():
    """BooksXperts' transmission contract as {link: subcategory}."""
    with open(REFERENCE_PATH) as fh:
        ref = json.load(fh)
    return {link: meta['subcategory'] for link, meta in ref['accounts'].items()}


def reconcile():
    """Return the drift that would break the TB handoff.

    {
      'missing_in_booksxperts': [link, ...],   # Analee emits it, BX has no such link -> suspense
      'subcategory_mismatch':   [(link, analee_subcat, bx_subcat), ...],  # -> misclassified
    }
    An empty result on both keys == fully reconciled.
    """
    an = analee_chart()
    bx = booksxperts_reference()
    missing = sorted(link for link in an if link not in bx)
    mismatch = sorted(
        (link, an[link], bx[link]) for link in an if link in bx and an[link] != bx[link]
    )
    return {'missing_in_booksxperts': missing, 'subcategory_mismatch': mismatch}
