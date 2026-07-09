"""Protected Assets Lock — a machine-enforced FREEZE on critical assets (Flask edition).

This repo is under a maintenance freeze, and its **chart of accounts** is a
critical asset Festus has lost before. This guard freezes the chart (and the
trial-balance core) byte-for-byte: nobody — human or AI agent — changes them as a
side effect of another task. A change is allowed only as a deliberate, Festus-
approved act.

Mirrors the Feature Manifest guard already in this repo, but stricter: the Feature
Manifest enforces a FLOOR (nothing removed); this lock freezes exact content.

  • Guard (CI / pre-deploy):   python protected_assets.py --check   (exit 1 on drift)
  • Re-freeze (approved change): python protected_assets.py --authorized-by "Festus: <reason>"
The pytest guard is `tests/test_protected_assets_lock.py`.

Read-only except the explicit re-freeze; it only hashes files.
"""
import hashlib
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(_ROOT, 'protected_assets.lock.json')

# Whole-file assets, frozen byte-for-byte (repo-root-relative).
PROTECTED_FILES = [
    'services/chart_of_accounts.py',     # chart-of-accounts logic
    'services/chart_seed_data.py',       # the chart seed definition
    'services/entity_chart_rules.py',    # per-entity chart rules
    'services/entity_chart_schema.py',   # chart schema
    'utils/chart_of_accounts.py',        # chart helpers
    'reports/trial_balance_service.py',  # trial-balance (financial-reporting core)
]


def fingerprint_file(rel_path):
    with open(os.path.join(_ROOT, rel_path), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_fingerprints():
    return {f'file:{rel}': fingerprint_file(rel) for rel in PROTECTED_FILES}


def load_lock():
    with open(LOCK_PATH) as fh:
        return json.load(fh)


def save_lock(fingerprints, authorized_by):
    payload = {
        'authorized_by': authorized_by,
        'reason': authorized_by,
        'assets': dict(sorted(fingerprints.items())),
    }
    with open(LOCK_PATH, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')


def diff_against_current():
    current = build_fingerprints()
    try:
        locked = load_lock().get('assets', {})
    except FileNotFoundError:
        locked = {}
    changed = {k: (locked.get(k), current.get(k))
               for k in set(locked) & set(current) if locked.get(k) != current.get(k)}
    removed = sorted(set(locked) - set(current))
    added = sorted(set(current) - set(locked))
    return changed, removed, added


def _main(argv):
    check = '--check' in argv
    authorized_by = None
    if '--authorized-by' in argv:
        i = argv.index('--authorized-by')
        if i + 1 < len(argv):
            authorized_by = argv[i + 1]

    changed, removed, added = diff_against_current()
    for k, (old, new) in changed.items():
        print(f'[~] CHANGED {k}: {old} -> {new}')
    for k in removed:
        print(f'[-] REMOVED {k}')
    for k in added:
        print(f'[+] NEW     {k} (not yet locked)')

    if check:
        if changed or removed or added:
            print('PROTECTED ASSETS CHECK FAILED — the chart of accounts / trial-balance core is '
                  'FROZEN. A change is only allowed with Festus\'s explicit approval; then run '
                  '`python protected_assets.py --authorized-by "Festus: <reason>"` and commit the '
                  'new protected_assets.lock.json.')
            return 1
        print('Protected Assets OK — everything frozen and matching the lock.')
        return 0

    if not authorized_by:
        print('Refusing to write: pass --authorized-by "Festus: <reason>" to record who approved '
              're-freezing the protected assets. Use --check to guard only.')
        return 2
    save_lock(build_fingerprints(), authorized_by)
    print(f'Protected Assets Lock re-frozen ({len(PROTECTED_FILES)} assets). '
          f'Authorised by: {authorized_by}. Commit protected_assets.lock.json.')
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
