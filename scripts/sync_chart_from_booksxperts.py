"""Sync Analee's chart of accounts (services/chart_seed_data.py) from
BooksXperts' live seed — additive-only, conflict-safe (Festus, 2026-07-17).

Default is a dry-run report. Nothing is written to chart_seed_data.py (a
frozen asset) without --apply, and even then only clean, additive ADD rows
are written — CONFLICT and STALE entries are always report-only, for a human
to resolve.

    python scripts/sync_chart_from_booksxperts.py
    python scripts/sync_chart_from_booksxperts.py --apply
    python scripts/sync_chart_from_booksxperts.py --booksxpert /path/to/booksxpert

After --apply: run the test suite, then land the frozen-asset change via
    python protected_assets.py --authorized-by "Festus: <reason>"
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from services.chart_reconciliation import (  # noqa: E402
    apply_sync_diff,
    compute_sync_diff,
    default_booksxpert_root,
)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--booksxpert', default=None,
                     help='Path to a local CNBSSA/booksxpert checkout (default: ../booksxpert).')
    ap.add_argument('--apply', action='store_true',
                     help='Write ADD rows into services/chart_seed_data.py. Default is dry-run.')
    args = ap.parse_args(argv)

    root = args.booksxpert or default_booksxpert_root()
    diff = compute_sync_diff(root)

    if diff.is_clean:
        print('Fully reconciled — nothing to add, no conflicts, no stale links.')
        return

    if diff.adds_common:
        print(f'ADD — common ({len(diff.adds_common)}):')
        for number, name, link, category, sub_category in diff.adds_common:
            print(f'  {link}  {name}  (number={number}, {category}/{sub_category})')

    for entity_name, rows in sorted(diff.adds_by_entity.items()):
        print(f'ADD — entity {entity_name!r} ({len(rows)}):')
        for number, name, link, category, sub_category in rows:
            print(f'  {link}  {name}  (number={number}, {category}/{sub_category})')

    if diff.conflicts:
        print(f'CONFLICT — not touched, needs human review ({len(diff.conflicts)}):')
        for link, analee_subcat, bx_subcat in diff.conflicts:
            print(f'  {link}  analee={analee_subcat}  booksxperts={bx_subcat}')

    if diff.stale:
        print(f'STALE — Analee has it, BooksXperts no longer does, not removed ({len(diff.stale)}):')
        for link in diff.stale:
            print(f'  {link}')

    if diff.out_of_scope:
        print(f'OUT OF SCOPE — BooksXperts entity has no Analee counterpart ({len(diff.out_of_scope)}):')
        for entity_name, link, name in diff.out_of_scope:
            print(f'  [{entity_name}] {link}  {name}')

    if not args.apply:
        print(f'\nDry-run only — {diff.add_count} ADD row(s) would be written. Re-run with --apply to write them.')
        return

    target_path = os.path.join(_ROOT, 'services', 'chart_seed_data.py')
    written = apply_sync_diff(diff, target_path)
    print(f'\nWrote {written} ADD row(s) to {target_path}. '
          'Review the diff, run tests, then land via '
          'python protected_assets.py --authorized-by "Festus: <reason>".')


if __name__ == '__main__':
    main(sys.argv[1:])
