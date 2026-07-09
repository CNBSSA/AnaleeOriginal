# Chart reconciliation — Analee → BooksXperts trial-balance handoff

Analee exports a trial balance that BooksXperts (and The Accountants) import via
`dataimports`. The intake maps each TB row to a chart account **by its LINK**
(columns: `Link, Account Name, Amount` — see `reports/trial_balance_service.py`),
and the matched account's **subcategory** drives AFS classification.

So for the handoff to be correct, every account link Analee can emit must:
- **exist** in BooksXperts' chart — otherwise the row lands in **suspense**;
- carry the **same subcategory** there — otherwise the row is **misclassified**.

## The guard

- `reconciliation/booksxperts_chart_reference.json` — a committed snapshot of
  BooksXperts' transmission contract (`link → subcategory`, generated from
  `booksxpert/app/migrations/0154_seed_chart_of_accounts.py`).
- `services/chart_reconciliation.py` — compares Analee's own chart
  (`services/chart_seed_data.py`) against that reference.
- `tests/test_chart_reconciliation.py` — fails the build if Analee emits a link
  BooksXperts lacks, or classifies a shared link under a different subcategory.
  Verified: tampering the reference (moving a subcategory) trips the guard.

Current status: **204/204 links reconciled, zero subcategory mismatches** — a
transmitted trial balance maps cleanly into BooksXperts with correct AFS
classification.

## When BooksXperts' chart changes

BooksXperts' chart is a frozen asset; changing it is a Festus-approved act. After
it changes, regenerate this reference so the two stay reconciled:

```
<booksxpert>/.venv/bin/python scripts/refresh_booksxperts_chart_reference.py \
    --booksxpert <path to booksxpert checkout>
```

Review and commit the updated `reconciliation/booksxperts_chart_reference.json`.
The guard then re-verifies Analee's chart against the new contract.
