# Protected Assets — FROZEN (do not change without Festus's explicit approval)

This repo is under a maintenance freeze, and its **chart of accounts** is a
critical asset Festus has lost before. The chart-of-accounts files and the
trial-balance core are now **frozen** byte-for-byte. Nobody — human or AI agent —
changes them as a side effect of another task. A change is allowed **only** as a
deliberate act Festus has explicitly approved.

## Frozen assets
- `services/chart_of_accounts.py`
- `services/chart_seed_data.py` (the chart seed definition)
- `services/entity_chart_rules.py`
- `services/entity_chart_schema.py`
- `utils/chart_of_accounts.py`
- `reports/trial_balance_service.py` (financial-reporting core)

## Enforcement
- `protected_assets.py` fingerprints each frozen asset (SHA-256);
  `protected_assets.lock.json` is the committed baseline (records `authorized_by`).
- `tests/test_protected_assets_lock.py` fails the build the instant any asset
  drifts. `python protected_assets.py --check` is the CLI / pre-deploy gate
  (exit 1 on drift). Verified: tampering turns the guard red; restoring is green.

## The ONLY sanctioned way to change a frozen asset
1. Festus explicitly approves the change.
2. Make it on a feature branch.
3. Re-freeze, recording who approved it:
   ```
   python protected_assets.py --authorized-by "Festus: <why approved>"
   ```
4. Commit the changed `protected_assets.lock.json` with the change.

The lock's `authorized_by` / `reason` fields are the audit trail. No approval →
guard stays red → nothing merges.
