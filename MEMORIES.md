# Bug-Finding Memory

Tracks bugs reported across runs. One line each: location + root cause, PR URL, status, date recorded.
Only open or rejected PRs are kept here.

## Open

- `reports/routes.py::trial_balance` — date filter applied only in the join/WHERE while the balance sums the lazy `account.transactions` relationship, so out-of-period transactions leak into trial-balance totals; fixed with `contains_eager`. PR: https://github.com/CNBSSA/AnaleeOriginal/pull/4 — status: open — recorded: 2026-06-23
