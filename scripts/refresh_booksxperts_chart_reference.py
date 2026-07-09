"""Refresh the committed BooksXperts chart reference (link -> subcategory).

Run this ONLY when BooksXperts' chart of accounts has legitimately changed (a
Festus-approved act) and Analee's trial-balance handoff must be reconciled to the
new contract. It reads BooksXperts' master seed and rewrites
`reconciliation/booksxperts_chart_reference.json`; commit the result.

Because it reads a file from a *separate* repo, point it at a local BooksXperts
checkout and run it with an interpreter that can import Django (the seed migration
imports `django.db`). Example:

    /path/to/booksxpert/.venv/bin/python scripts/refresh_booksxperts_chart_reference.py \
        --booksxpert /path/to/booksxpert

The default --booksxpert is a sibling checkout at ../booksxpert.
"""
import argparse
import importlib.util
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(_ROOT, 'reconciliation', 'booksxperts_chart_reference.json')
SEED_REL = os.path.join('app', 'migrations', '0154_seed_chart_of_accounts.py')


def _load_module(path):
    spec = importlib.util.spec_from_file_location('bx_seed_0154', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# BooksXperts chart changes made AFTER migration 0154 (which stays historical).
# Each entry mirrors a later, Festus-approved BooksXperts migration. Keep this
# list in step whenever the BooksXperts chart changes — the guard test will
# catch a miss.
POST_SEED_ADJUSTMENTS = {
    # 0232 (Wave 2B): rental split — rename commercial, add residential.
    'i.060.000': {'name': 'Rental Income - Commercial'},
    'i.061.000': {'name': 'Rental Income - Residential', 'subcategory': 'Sales', 'add': True},
    # 0235 (Wave 3): non-supplies re-filed under Other income ('Income').
    'is.700.000': {'subcategory': 'Income'},
    'is.710.000': {'subcategory': 'Income'},
    'i.090.000': {'subcategory': 'Income'},
}


def build_reference(booksxpert_root):
    seed_path = os.path.join(booksxpert_root, SEED_REL)
    if not os.path.exists(seed_path):
        sys.exit(f'BooksXperts seed not found at {seed_path} — pass --booksxpert <checkout>.')
    seed = _load_module(seed_path)  # needs Django importable
    accounts = {}

    def add(rows):
        for t in rows:  # (number, name, link, subcat, main, default_for)
            accounts[t[2]] = {'subcategory': t[3], 'name': t[1]}

    add(seed.COMMON_ACCOUNTS)
    for _entity, rows in seed.ENTITY_SPECIFIC.items():
        add(rows)

    for link, change in POST_SEED_ADJUSTMENTS.items():
        if change.get('add'):
            accounts[link] = {'subcategory': change['subcategory'], 'name': change['name']}
        elif link in accounts:
            accounts[link].update(
                {k: v for k, v in change.items() if k in ('subcategory', 'name')})

    return {
        'source': 'CNBSSA/booksxpert — app/migrations/0154_seed_chart_of_accounts.py',
        'note': ('Transmission contract for the Analee -> BooksXperts trial-balance handoff. '
                 'dataimports maps each TB row to a CompanyAccount BY LINK; the subcategory '
                 'drives AFS classification. Analee must not emit a link absent here, nor one '
                 'whose subcategory differs. Regenerate with '
                 'scripts/refresh_booksxperts_chart_reference.py (a Festus-approved act).'),
        'accounts': dict(sorted(accounts.items())),
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--booksxpert', default=os.path.join(os.path.dirname(_ROOT), 'booksxpert'),
                    help='Path to a local CNBSSA/booksxpert checkout (default: ../booksxpert).')
    args = ap.parse_args(argv)

    ref = build_reference(args.booksxpert)
    with open(REFERENCE_PATH, 'w') as fh:
        json.dump(ref, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(f'Wrote {len(ref["accounts"])} accounts to {REFERENCE_PATH}. '
          'Review and commit the change.')


if __name__ == '__main__':
    main(sys.argv[1:])
