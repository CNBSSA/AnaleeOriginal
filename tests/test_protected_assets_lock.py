"""The Protected Assets Lock guard — Analee's chart of accounts is FROZEN.

If this fails, a protected asset (the chart of accounts / trial-balance core) was
changed. That is only allowed as a deliberate, Festus-approved act: get his
approval, then run
    python protected_assets.py --authorized-by "Festus: <reason>"
and commit the changed protected_assets.lock.json. Do NOT edit the lock by hand,
and do NOT change a protected asset to make an unrelated task pass.

This repo is under a maintenance freeze; this guard is a read-only safety net (it
only hashes files) and changes no product behaviour.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import protected_assets  # noqa: E402


def test_all_protected_assets_match_the_lock():
    changed, removed, added = protected_assets.diff_against_current()
    detail = []
    for k, (old, new) in changed.items():
        detail.append(f'CHANGED {k}: {old} -> {new}')
    detail += [f'REMOVED {k}' for k in removed]
    detail += [f'NEW {k} (not yet locked)' for k in added]
    assert (changed, removed, added) == ({}, [], []), (
        'PROTECTED ASSETS CHANGED — the chart of accounts is FROZEN.\n'
        + '\n'.join(detail)
        + '\n\nAllowed only with Festus\'s explicit approval; then run '
          '`python protected_assets.py --authorized-by "Festus: <reason>"` and commit the lock.'
    )


def test_lock_records_who_authorised_it():
    lock = protected_assets.load_lock()
    assert lock.get('authorized_by'), 'the lock must record who authorised the freeze.'
    assert lock.get('assets'), 'the lock must fingerprint at least one asset.'


def test_every_declared_asset_is_fingerprinted():
    fps = protected_assets.build_fingerprints()
    for rel in protected_assets.PROTECTED_FILES:
        assert f'file:{rel}' in fps
