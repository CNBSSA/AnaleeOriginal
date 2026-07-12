"""Refresh the committed BooksXperts chart reference (link -> subcategory).

Run this ONLY when BooksXperts' chart of accounts has legitimately changed (a
Festus-approved act) and Analee's trial-balance handoff must be reconciled to
the new contract. It reads BooksXperts' RUNNABLE master seed —
`app/management/commands/seed_chart_of_accounts.py`, the same command Railway
preDeploy executes, i.e. the live source of truth — and rewrites
`reconciliation/booksxperts_chart_reference.json`; commit the result.

P1 redesign (Festus, 2026-07-10; adversarial audit #2): the previous version
read the HISTORICAL migration 0154 and patched it with a hand-maintained
POST_SEED_ADJUSTMENTS overlay. That overlay was a manual drift channel — every
BooksXperts chart change had to be mirrored here by hand, and the reference had
already fallen 28 links behind. Reading the runnable seed removes the overlay
and the drift channel entirely.

The seed file is parsed with `ast` — no Django import, no BooksXperts venv
needed; any Python 3 interpreter works:

    python scripts/refresh_booksxperts_chart_reference.py \
        --booksxpert /path/to/booksxpert

The default --booksxpert is a sibling checkout at ../booksxpert.
"""
import argparse
import ast
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(_ROOT, 'reconciliation', 'booksxperts_chart_reference.json')
SEED_REL = os.path.join('app', 'management', 'commands', 'seed_chart_of_accounts.py')

# Canonical BooksXperts subcategory taxonomy — PKs 1..10 are load-bearing and
# machine-pinned in BooksXperts (create_subcategories + migration
# 0185_pin_subcategory_pks + core/subcategories.py). The seed rows carry the
# integer PK; the reference (and Analee's own chart) speak the NAME.
SUBCATEGORY_NAMES = {
    1: 'Current Asset',
    2: 'Current Liability',
    3: 'Cost of Sales',
    4: 'Expenses',
    5: 'Sales',
    # BooksXperts' canonical display name for pk 6 is 'Fixed Assets &
    # Non-Current Assets'; the transmission contract (and Analee's own frozen
    # chart vocabulary) has always called the same bucket 'Non-Current Asset'.
    # Same pk, same semantics — keep the contract vocabulary stable here.
    6: 'Non-Current Asset',
    7: 'Non-Current Liabilities',
    8: 'Equity',
    9: 'Tax',
    10: 'Income',
}


def _parse_seed_tables(seed_path):
    """Extract COMMON_ACCOUNTS and ENTITY_SPECIFIC from the seed command via
    `ast` — the module imports Django models at module level, so exec'ing it
    would need a configured BooksXperts environment; parsing does not.

    Module-level simple constants (SALES = 5, MA_REVENUE = 'Revenue', …) are
    resolved so the row tuples evaluate to plain literals.
    """
    with open(seed_path) as fh:
        tree = ast.parse(fh.read(), filename=seed_path)

    constants = {}
    tables = {}

    class _Resolver(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in constants:
                return ast.copy_location(ast.Constant(constants[node.id]), node)
            return node

    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        value = node.value
        if isinstance(value, ast.Constant):
            constants[name] = value.value
        elif name in ('COMMON_ACCOUNTS', 'ENTITY_SPECIFIC'):
            resolved = _Resolver().visit(value)
            ast.fix_missing_locations(resolved)
            tables[name] = ast.literal_eval(resolved)

    missing = {'COMMON_ACCOUNTS', 'ENTITY_SPECIFIC'} - set(tables)
    if missing:
        sys.exit(f'Seed tables not found in {seed_path}: {sorted(missing)} — '
                 'has the seed command been restructured?')
    return tables['COMMON_ACCOUNTS'], tables['ENTITY_SPECIFIC']


def build_reference(booksxpert_root):
    seed_path = os.path.join(booksxpert_root, SEED_REL)
    if not os.path.exists(seed_path):
        sys.exit(f'BooksXperts seed not found at {seed_path} — pass --booksxpert <checkout>.')
    common, entity_specific = _parse_seed_tables(seed_path)

    accounts = {}

    def add(rows):
        for t in rows:  # (number, name, link, subcategory_pk, main, default_for)
            subcat_pk = t[3]
            if subcat_pk not in SUBCATEGORY_NAMES:
                sys.exit(f'Unknown subcategory pk {subcat_pk!r} on link {t[2]!r} — '
                         'update SUBCATEGORY_NAMES only if BooksXperts has '
                         'legitimately extended its pinned taxonomy.')
            accounts[t[2]] = {'subcategory': SUBCATEGORY_NAMES[subcat_pk], 'name': t[1]}

    add(common)
    for _entity, rows in entity_specific.items():
        add(rows)

    return {
        'source': ('CNBSSA/booksxpert — app/management/commands/'
                   'seed_chart_of_accounts.py (the runnable master seed)'),
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
