# Analee — subscriber deletion and record retention

**Ruling: 2026-09-04. Delegated by Festus ("make your choice").**
Supersedes nothing; there was no prior policy, which is how the defect below
survived.

> **Deleting a subscriber soft-deletes the PERSON and RETAINS their accounting
> records.** No path in this application may cascade-delete a user's financial
> data.

---

## Why this, and not a hard delete

It is not a preference. Every financial relationship on `User` in `models.py` is
declared `cascade='all, delete-orphan'` with `ondelete='CASCADE'` —
`transactions`, `accounts`, `bank_statement_uploads`, `company_settings`,
`historical_data` and the rest. So `db.session.delete(user)` does not merely
remove a person: it destroys that person's **books**.

Those books are accounting records, and South African law requires they be kept:

| Authority | Requirement |
|---|---|
| **Companies Act 71 of 2008, s24(3)** | Accounting records retained **7 years** |
| **Tax Administration Act 28 of 2011, ss29–30** | Records supporting a return kept **5 years** from submission — longer while an audit, objection or appeal is live |
| **VAT Act 89 of 1991, s55** | VAT records kept **5 years** |
| **POPIA 4 of 2013, s14(1)(a)** | Personal information may be retained where **another law requires it** — so this retention is lawful, not an exception to POPIA |
| **POPIA, s14(4)** | Once retention is no longer authorised, the information must be destroyed or de-identified |

There is a second reason beyond compliance. These are the customer's **own
evidence for their own filings**. If SARS queries a return two years after an
accountant closes their Analee account, purging the underlying transactions
leaves that person unable to answer. Destroying it on request would be a
disservice dressed up as a favour.

## What the three methods do

Defined on `User` in `models.py`. **No schema change** — no migration, no new
column, consistent with how this repo has handled every prior change.

| Method | Effect |
|---|---|
| `soft_delete()` | `is_deleted=True`, `subscription_status='deleted'`. `is_active` already excludes both, so Flask-Login refuses the session. The password hash is deliberately **not** cleared, so restore is lossless. **Touches no financial record.** |
| `restore_account()` | Reverses it to `'deactivated'` — **not** straight back to active. The prior status is not stored, so an admin re-grants access deliberately rather than this method guessing at an entitlement. Fail-closed. |
| `release_identifiers_for_reregistration()` | Moves the email/username aside to `deleted+<id>@analee.invalid` (`.invalid` is reserved by RFC 2606 and can never be routed or registered) so the address is free again. Raises `ValueError` on a live account. **The row and every record under it stay put.** |

## What changed at the call sites

Both callers existed and both were broken — they called methods that were
defined nowhere, so each raised `AttributeError`.

**`admin/routes.py` — the dangerous one.** After the undefined `soft_delete()`,
the handler purged transactions, accounts, statement uploads, company settings
and more, then flashed *"permanently deleted"*. It never ran, because the
`AttributeError` dropped it into the error handler every time. **Implementing
`soft_delete()` without changing this would have armed it.** The purge is gone;
the handler soft-deletes and says plainly that records are retained.

*This is a deliberate behaviour change to an admin action, made under Festus's
delegation.* The admin can still remove a subscriber's access; what they can no
longer do is destroy the books as a side effect.

**`forms/auth.py` — a destructive write inside a validator.** Typing an address
into the signup form called `restore_account()` and then
`db.session.delete(user)`, committed — cascading the previous account's books
away merely because someone typed. A validator now only validates. Freeing the
address happens deliberately in `auth.register`, after every validator passes,
via `release_identifiers_for_reregistration()`.

## Guarded by

`tests/test_deletion_lifecycle.py` (6). The tests that matter assert records
**survive**. Proven toothed: restoring the old `db.session.delete(existing_user)`
fails with *"the new signup reused the retired row"*.

**If one of those assertions starts failing, the product is destroying customer
books. Do not fix it by changing the assertion.**

## Still open

**Purging after the retention period genuinely expires.** POPIA s14(4) requires
destruction or de-identification once retention is no longer authorised. No such
operation exists, and one was not built here — a tool whose job is to delete
accounting records deserves its own task, its own audit and Festus's explicit
word, not the tail end of a repair. Tracked in `docs/QA_AUDIT_2026-09-04.md`.

**A data-subject deletion request under POPIA s24** would need this policy
explained to the requester: the personal identifiers can be released, the
statutory accounting records cannot be destroyed inside the retention window.
Worth a line in the privacy notice.
