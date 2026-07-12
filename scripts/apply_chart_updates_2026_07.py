#!/usr/bin/env python3
"""Deliver the 2026-07 chart updates to EXISTING company charts (idempotent).

Audit #2 P2 (Festus-approved, 2026-07-10): `services/chart_seed_data.py` only
shapes NEWLY seeded charts — the deployed instance's existing per-user Account
rows never received the Wave 2B/3 changes, so their trial-balance handoff to
BooksXperts drifts from the reconciliation contract. This script applies exactly
those changes to existing rows, only-where-still-old so a user's deliberate
customisation is never overridden:

  1. i.060.000 named exactly 'Rental Income' -> 'Rental Income - Commercial'
     (Wave 2B rental split; §12(c) is why the two rentals must be separate).
  2. Add i.061.000 'Rental Income - Residential' (Income / Sales) for every
     user who has a chart (has i.060.000) but lacks the residential account.
  3. is.700.000, is.710.000, i.090.000: sub_category 'Sales' -> 'Income'
     (Wave 3 — non-supplies out of the Sales bucket).

DRY-RUN by default — prints what would change. Pass --apply to write.
Run on the deployment (Railway shell) with the app's normal env:

    python scripts/apply_chart_updates_2026_07.py            # preview
    python scripts/apply_chart_updates_2026_07.py --apply    # write

Safe to run repeatedly. Data-only: no frozen chart file is touched.
"""
import sys

from app import create_app
from models import db, Account

RENAME_LINK = 'i.060.000'
RENAME_OLD = 'Rental Income'
RENAME_NEW = 'Rental Income - Commercial'

RESIDENTIAL = dict(link='i.061.000', name='Rental Income - Residential',
                   category='Income', sub_category='Sales', account_code='4111')

REFILE_LINKS = ('is.700.000', 'is.710.000', 'i.090.000')


def run(apply_changes):
    renamed = refiled = added = 0

    # 1. Rental rename — only the untouched seed name is renamed.
    for acc in Account.query.filter_by(link=RENAME_LINK, name=RENAME_OLD):
        print(f'  rename user={acc.user_id}: "{acc.name}" -> "{RENAME_NEW}"')
        acc.name = RENAME_NEW
        renamed += 1

    # 2. Residential rental — add where the user has a chart but lacks it.
    users_with_chart = {a.user_id for a in
                        Account.query.filter_by(link=RENAME_LINK).all()}
    users_with_residential = {a.user_id for a in
                              Account.query.filter_by(link=RESIDENTIAL['link']).all()}
    for user_id in sorted(users_with_chart - users_with_residential):
        print(f'  add user={user_id}: {RESIDENTIAL["link"]} "{RESIDENTIAL["name"]}"')
        db.session.add(Account(user_id=user_id, is_active=True, **RESIDENTIAL))
        added += 1

    # 3. Non-supplies re-filed under Income — only rows still at 'Sales'.
    for acc in Account.query.filter(Account.link.in_(REFILE_LINKS),
                                    Account.sub_category == 'Sales'):
        print(f'  re-file user={acc.user_id}: {acc.link} "{acc.name}" Sales -> Income')
        acc.sub_category = 'Income'
        refiled += 1

    if apply_changes:
        db.session.commit()
        verdict = 'APPLIED'
    else:
        db.session.rollback()
        verdict = 'DRY-RUN (nothing written — pass --apply to write)'

    print(f'\n{verdict}: {renamed} renamed, {added} residential account(s) added, '
          f'{refiled} re-filed.')


def main(argv):
    apply_changes = '--apply' in argv
    app = create_app()
    with app.app_context():
        run(apply_changes)


if __name__ == '__main__':
    main(sys.argv[1:])
