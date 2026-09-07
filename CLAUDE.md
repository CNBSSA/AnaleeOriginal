# Project Working Agreement

## ⛔ STANDING RULE — MAIN-BRANCH PROMOTION WINDOW ONLY (Festus, 2026-08-06)

We now have live paying clients on the SA products. **Promote `develop → main`
(production) ONLY inside the low-usage window: 11:00 PM – 5:00 AM SAST** (South
Africa overnight; the SAST anchor is fixed across US DST). Always after sufficient
testing **and** a boot-verify (`manage.py check` + the suite). `develop` work and PR
merges into `develop` continue anytime — only `develop → main` is gated. A genuine
live-incident hotfix (active data/security harm) may promote outside the window
**only with Festus's explicit say-so on that specific fix.** Corporation-wide
authoritative copy + the US-Eastern equivalent (5–11 PM ET during EDT; 4–10 PM EST) — the SAME SA-anchored window, NOT a separate US clock (corrected Festus 2026-08-08) — and Nigeria (≈10 PM–4 AM WAT) +
rationale: `autonomusFV/agents/infra_facts.md`.

## ⚠️ OPEN DEFECTS — READ BEFORE CALLING THIS REPO HEALTHY (2026-09-04)

A green suite and a green deploy are **not** evidence that this product is
ready. As of 2026-09-04 there are verified open defects, including one that may
be a live incident. The standing register is **`docs/QA_AUDIT_2026-09-04.md`** —
read it before reporting on Analee's readiness, and update it when a finding is
fixed or a new one is verified.

GitHub **Issues are disabled on this repo**, so that file IS the issue tracker.
Do not let a finding live only in a chat transcript.

Two lessons from that audit worth carrying:
- **A security fix can exist in history and be on no branch.** `71eef5a` removed
  the hard-coded admin credential in June and is an ancestor of neither `main`
  nor `develop`. Checking recent CI would never have shown it.
- **"Not configured" that exits 0 reads as healthy.** Green automation is not
  evidence of availability.

## CNBSSA agent system (corp)

- **Corp agent system (context layers, memory, learning):** `autonomusFV/agents/CNBSSA_AGENT_SYSTEM.md` (workspace clone path); org conventions: `autonomusFV/org-conventions/`.

## ❄️ THE FREEZE — REDEFINED BY FESTUS, 2026-07-18: A CAPABILITY FREEZE, NOT A WHOLE-REPO FREEZE

Festus (voice note, 2026-07-18): *"I'm not freezing the entire application. I'm
just freezing those capacities because I lost it before, and I do not want to
lose it again. That is the analysis feature that I'm locking — the capacity to
analyze things and learn from the past and suggest things. I'm not locking the
upload, download, and all that."*

Festus (2026-07-19, with a screenshot of the **Analyze Data** page): *"This is
the feature I am freezing — here and everywhere it rears its head. And the rest
of AI agents, the data processing features must be preserved. It is the engine —
not how data get to the processing features or after the processing features."*

**What IS frozen (do not touch without Festus's explicit approval):**
1. **THE ENGINE — the Analyze Data processing pipeline**, concretely (as shown
   on the Financial Analysis → Analyze Data screen): the transaction work loop
   (assign accounts + explanations, **10 at a time**), **Auto-process next 10**
   (AI account suggestions), the **iCountant Assistant** guided one-by-one
   review, and the machinery beneath them — categorisation, the
   learn-from-past-transactions capability (Recall/similarity), and the
   suggestion logic. Any module whose job is to *analyze, learn, or suggest* is
   inside the freeze. **The freeze follows the engine everywhere it rears its
   head** — including the embedded `analee/` module in `CNBSSA/booksxpert` —
   and the same principle protects the data-processing features of every AI
   agent across the corporation's products (corporation-wide copy:
   `autonomusFV/CLAUDE.md` → "DATA-PROCESSING ENGINE FREEZE").
2. **The chart of accounts + trial-balance core** — already machine-enforced
   (see PROTECTED ASSETS below): `services/chart_of_accounts.py`,
   `services/chart_seed_data.py`, `services/entity_chart_rules.py`,
   `services/entity_chart_schema.py`, `utils/chart_of_accounts.py`,
   `reports/trial_balance_service.py`.

**What is NOT frozen (normal working surface — standard workflow applies, no
"scoped re-open" ceremony needed):** how data GETS TO the engine (upload/
download flows, ingress) and what happens AFTER it (trial-balance export,
transmission, reporting — reporting lives in BooksXperts/THE ACCOUNTANTS, never
here), pages/templates/navigation, auth/login/session, the `club_sso/` module,
`entitlement.py`, `provisioning.py` (the seam), and new accountant-companion
surfaces (e.g. the One-Login Practice Layer planned in
`autonomusFV/docs/analee_accountants_companion_redesign.md`). Every change
still runs the full Company Development Workflow below — audits are never
waived — but Festus's per-scope unfreeze approval is only required when a
change would touch the frozen capabilities above.

**Standing engineering rule:** any task that modifies the non-frozen surface
must *call* the frozen engine, never *change* it — and its Change-Impact Audit
must explicitly state "frozen analysis engine + chart/TB core untouched".

**TODO (next Analee coding task):** enumerate the engine's files (the Analyze
Data pipeline, Auto-process, iCountant Assistant, prediction/suggestion
services) and add them to `protected_assets.lock.json` so capability-freeze #1
is machine-enforced exactly like the chart freeze (#2), not just
policy-enforced. `?` — file list to be verified in-repo at that time.

### Historical note — the 2026-06-26 whole-repo freeze framing (superseded 2026-07-18)

The section below recorded a whole-repo "maintenance only" freeze on 2026-06-26,
when the embedded `analee/` module inside `CNBSSA/booksxpert` was designated the
canonical Analee. That framing is **superseded** by the capability freeze above:
Festus's 2026-07-15 "companion of the accountants" direction and 2026-07-18
Practice Layer decision place the accountant-facing Analee surface back in
active development **in this repo**. The text is retained as history; the
"scoped re-open + re-freeze" records that follow it remain accurate history of
work done while the whole-repo framing stood.

> **[Historical, 2026-06-26]** This standalone Analee repo is frozen. It is no
> longer the place for new feature work. The canonical Analee is the **embedded
> `analee/` module inside `CNBSSA/booksxpert`** (Django, multi-tenant, real
> GL/journal posting, plus the BooksXperts Assistant). Standalone Analee and the
> embedded one were investigated side by side (2026-06-26): the embedded version
> is equal-or-better on every capability, and the **one** thing standalone had
> that it lacked — **Recall** (similarity-based "find my similar past
> transactions", formerly *ERF*) — has been **harvested into
> `booksxpert/analee/services/recall.py`** with an agent tool and tests.
>
> Rules while frozen: **Allowed:** security fixes, dependency CVEs, and keeping
> the deployed instance running. **Not allowed without Festus re-opening the
> repo:** new features, refactors, or cosmetic churn — including the
> previously-planned internal ERF/ASF/ESF symbol rename (#3), formally DROPPED
> (see `docs/PERSONA_AND_TOOL_NAMING.md`). New ideas for Analee go to
> **booksxpert** (the embedded module), not here.

### Scoped re-open + re-freeze record (Festus, 2026-07-10)

Festus re-opened this repo for **ONE scope only**: The Practice Club SSO
consumer, so Analee joins the Club suite ("it's very important for accountants…
we have to just be very careful to work on it and move it into the group, then
we can freeze it back"). Delivered as the **sealed `club_sso/` module**: one
registration call in `app.py`, dark behind `CLUB_ENABLED`, fail-soft at boot,
per-user workspace mapping, env-guarded walkthrough seed
(`CLUB_WALKTHROUGH_SEED=1`), 9 tests in `tests/test_club_sso.py`.

**The repo is now RE-FROZEN with `club_sso/` inside the freeze.** The frozen
rules above apply to `club_sso/` exactly as to everything else. Festus's future
Analee ideas ("I'm going to build upon it") each require his explicit re-open,
scoped like this one. *(2026-07-18: under the redefined capability freeze,
`club_sso/` is now normal working surface.)*

### Scoped re-open + re-freeze record (Festus, 2026-07-13)

Festus re-opened this repo for **ONE scope only**: the **Analee bundle access
model** — "Analee is only available for users who is either a subscriber of The
Accountants or a member of the Practice Club" (the access rule that makes Club /
Accountants membership meaningful for Analee, completing the "move it into the
group" intent). Delivered, all **dark** and **additive** (no schema change):
- `entitlement.py` — single source of truth `analee_entitled(user)` = Club
  member (session `club_session`/`club_member_id`, set by `club_sso`) OR
  subscriber (`subscription_status in active/pending`). Binary — full access, no
  tier.
- `app.py` — `before_request` gate behind `ANALEE_ENTITLEMENT_ENFORCED` (default
  off); friendly `/entitlement-required` notice (`routes.py` +
  `templates/entitlement_required.html`).
- `provisioning.py` — sealed S2S endpoint `POST /api/provisioning/analee`, dark
  behind `ANALEE_PROVISIONING_ENABLED` (404 while off) + bearer secret
  `ANALEE_PROVISIONING_SECRET`; activates/deactivates a user by email via
  `subscription_status` (no schema change; `is_deleted` untouched); CSRF-exempt.
- 17 tests (`tests/test_entitlement_gate.py`, `tests/test_provisioning.py`);
  full suite 136 OK. Also fixed a latent bug carried in older app.py revisions
  (`session` used but not imported).

**The repo is RE-FROZEN with the entitlement gate + provisioning inside the
freeze.** The frozen rules apply to them exactly as to everything else. Any
further Analee work requires Festus's explicit, scoped re-open. *(2026-07-18:
under the redefined capability freeze, these are now normal working surface.)*

### Scoped re-open + re-freeze record (Festus, 2026-07-15)

Festus re-opened this repo for **ONE scope only**: the **client-workspace
provisioning seam** — Analee becomes "a companion of the accountants": THE
ACCOUNTANTS orchestrates one Analee workspace per firm client, so bank
statements are analysed per client and the trial balance flows back to the
right client over the existing share-URL import. Delivered entirely **inside
the sealed `provisioning.py` module** (same dark flag
`ANALEE_PROVISIONING_ENABLED`, same fail-closed bearer
`ANALEE_PROVISIONING_SECRET`), all **additive, no schema change**:
- `POST /api/provisioning/analee/workspace` — idempotent ensure: dedicated
  `User` under a deterministic alias email
  (`client+<ref>@ws.theaccountants.local`, random password nobody learns) +
  `CompanySettings` named after the client + entity-correct chart via the
  **frozen** chart service (called, never changed). Re-ensure renames /
  reactivates.
- `POST /api/provisioning/analee/workspace/login-link` — short-TTL
  (`ANALEE_WORKSPACE_LINK_TTL`, default 90 s) `itsdangerous`-signed login
  path, purpose-salted off the same secret; **refuses non-workspace
  accounts**, so it can never authenticate as a human user.
- `GET /workspace/enter?token=…` — verifies and logs the accountant into the
  client workspace (`session['workspace_session']`); expired/tampered/revoked
  → friendly redirect to login.
- 12 tests in `tests/test_workspace_provisioning.py`; full suite 149 OK;
  `protected_assets.py --check` clean.

**The repo is RE-FROZEN with the workspace seam inside the freeze.** The
frozen rules apply to it exactly as to everything else. The heavy lifting
(client mapping, buttons, orchestration) lives in `CNBSSA/accountants`, which
is not frozen — future workspace ideas land there, not here. *(2026-07-18:
under the redefined capability freeze, the seam is now normal working surface;
the Practice Layer plan builds on it.)*

### Scoped re-open + re-freeze record (Festus, 2026-07-17)

Festus re-opened this repo for **ONE scope only**: extending the existing
detect-only `services/chart_reconciliation.py` guard into an additive-only,
conflict-safe **propose + apply** sync from BooksXperts' live chart seed —
so a BooksXperts chart change can flow into Analee's own chart without
losing the existing safety net. Delivered:
- `services/chart_reconciliation.py` — `compute_sync_diff()` (reads
  BooksXperts' `seed_chart_of_accounts.py` via `ast`, no BooksXperts changes)
  classifies every link as **ADD** (safe, auto-proposed), **CONFLICT** (same
  link, different subcategory — never auto-applied, surfaced), **STALE**
  (Analee has it, BooksXperts no longer does — surfaced, never removed), or
  **out-of-scope** (a BooksXperts entity — Personal Liability Company, Trust —
  with no Analee `ENTITY_NAMES` counterpart). `apply_sync_diff()` writes
  ADD-only rows into `services/chart_seed_data.py`, insertion points located
  via `ast` (never guessed by hand); new account numbers auto-allocated
  per-subcategory, collision-checked.
- `scripts/sync_chart_from_booksxperts.py` — dry-run report by default;
  `--apply` writes.
- Ran `--apply` for real: **34 ADD rows, 0 conflicts, 0 stale** (the chart
  was already well-reconciled), 24 out-of-scope entries. No duplicate links
  in any entity's combined chart after the merge.
- 8 tests in `tests/test_chart_sync.py` (fixture-based diff classification +
  number-allocator collision safety + apply-writer correctness + a
  real-checkout smoke test); full suite 157 OK.

**The repo is RE-FROZEN with the sync machinery inside the freeze.** The
frozen rules apply to it exactly as to everything else. The equivalent
full-parity sync for THE ACCOUNTANTS' chart templates lives in
`CNBSSA/accountants` (`chart/booksxperts_sync.py`), which is not frozen the
same way — future chart-sync ideas for that repo land there, not here.
*(2026-07-18: chart-sync machinery touches the chart seed — it REMAINS inside
the capability freeze, per frozen item #2.)*

### Scoped re-open + re-freeze record (Festus, 2026-09-06)

Festus re-opened this repo for **ONE scope only**: **advancing automation** —
"more automation and more automation until nothing to automate". A review of
the whole Import → Categorize → Explain → Reconcile → Report pipeline found the
ceiling was not the AI but four defects plus a per-row call design. Delivered
in three stages, each through the full workflow:

**Tier 0 — unblock (maintenance, no re-open needed).** A row counted as
finished if it had EITHER an account OR an explanation, and both import paths
stamp the chosen bank account onto every row — so a statement was reported
"All processed" the instant it landed and its rows were invisible to
auto-processing (465 live transactions on Festus's own account). A row is now
done only with **both** halves. Also: the batch omitted `user_id`, sending every
tenant's chart to the model; auto-applied rows leaving the result set made the
offset window skip unassigned rows permanently; `save-transaction` and
`replicate-explanation` raised `TypeError` on every call
(`dict.get(key, type=int)` on a plain dict) so dropdown edits were lost
silently; and reconciliation reported "removed N duplicates, fixed N invalid
dates" while deleting and rewriting nothing. Removal stays un-automated by
design — the duplicate rule groups on date+amount+description, so two genuine
identical charges in a day are indistinguishable from a double capture.

**Honesty pass (maintenance).** Two places presented invented figures as
analysis, the rule already set by `57ee9a9`: `/api/icountant/<id>/insights`
returned the first three accounts in the chart at a fabricated `confidence: 0.5`
whenever the AI category did not map — which is nearly always, since the
category vocabulary is `nlp_utils`' personal-finance list while
`Account.category` only holds Assets/Liabilities/Equity/Income/Expenses — and
`icountant.html` auto-selected suggestion[0] into the dropdown, steering the
accountant into posting to Bank Cheque Account 1. The expense forecast carried
`overall_confidence: 0.85` / `reliability_score: 0.80` as literals nothing ever
computed, rendered as percentages on the page and in the client-facing PDF.

**Tier 1 — batched processing (this scope).** `services/bulk_suggestions.py`
replaces one Claude call per transaction (each carrying the user's entire
~1 000-account chart) with **one call per batch of 25** that returns the account
**and** the explanation, so a processed row comes back complete instead of
categorised but blank. Rules held: never invent an account (names are matched
against the real chart, never fuzzy-matched); never guess when the AI is offline
(returns nothing, so nothing is applied — the batch path previously consumed
`PredictiveFeatures`' SequenceMatcher fallback, whose ratio could clear the 0.85
gate); salvage a truncated reply row by row; write an account only into an empty
slot; write an explanation only into an empty slot, tagged with the new
`SOURCE_AI` so the books always show a machine wrote it, and
`save_explanation` refuses to let it overwrite anything a person wrote.

**Tier 2 — history first (this scope).** `services/history_matching.py` makes
the practice's own past treatment the FIRST signal, ahead of any AI call: what
this firm filed a payee under last month is free, instant, deterministic, and
consistent month to month. `normalize_description()` reduces an SA narration to
its payee by dropping every token containing a digit (references, dates, card
fragments), so "Magtape Credit Medihelp Smh0363383 20210204" and next month's
equivalent share one key; matching is an O(1) dict lookup against an index
built once per batch, replacing the old Recall's O(n) SequenceMatcher scan per
keystroke. Only rows with no precedent are sent to the model, so a recurring
payee costs nothing and the system gets more automatic the more the accountant
works. Ambiguity is refused rather than averaged: a payee historically split
across accounts returns a confidence below any auto-apply gate so a human
decides, and only rows carrying BOTH an account and an explanation count as
settled practice. A statement never teaches itself (`exclude_file_id`), so one
early mistake cannot propagate through the rest of the file.

**Tier 3 — no button (this scope).** `services/auto_process.py` starts the
categorise/explain pass the moment a statement is imported, from BOTH import
paths, so the accountant opens Analyze Data to find only the exceptions waiting
rather than a button to press. It runs **off-request on a daemon thread**: a
465-row statement is ~19 batched AI calls, and doing that inside the upload
request is exactly the mistake that made PDF extraction unusable for weeks —
gunicorn kills a slow worker from *outside* Flask, so no error handler runs and
the user sees a bare 500. The deploy runs `gthread` (3 workers × 4 threads), so
there are threads for it. Properties: never blocks the request; never crashes
the worker (every exception caught and logged); **idempotent**, since it only
fills empty slots, so a run cut short by a deploy can simply be re-run or
finished with the existing button; bounded by `MAX_BATCHES`; stops early when
no progress is possible (AI offline and no history) instead of spinning; and
switchable with `ANALEE_AUTOPROCESS_ON_IMPORT=0` for importing without spending
API credit.

**Tier 4 — leverage on what is left (this scope).**
`services/accountant_fanout.py` + "Apply to similar" on each analyze row: one
accountant decision is applied to every matching row on the statement.
Exceptions arrive in *families* (forty card purchases at one supermarket,
twelve identical debit orders), and deciding one then retyping it thirty-nine
times was the largest remaining piece of manual work. The client wizard already
had a fan-out (`services/client_erf.py`); the accountant — who does the
professional bulk of the work — had none, and the client version carries only
the explanation, never the account. Matching uses the same payee key as history
matching, so what the accountant fans out today is exactly what the system
recognises on its own next month. Refusals: a row **a person already decided**
(accountant or client) is never offered and never written, even if its id is
posted directly; ids are re-validated against file and user rather than
trusted; a foreign account is ignored rather than guessed around; a description
with no identifying payee fans out to nothing; bounded by `MAX_FANOUT`. It is
preview-then-confirm — the count is shown before anything is written.

**Tier 5 — the machine's work is visible (this scope).** Automation now writes
explanations at scale and every write was already tagged (`SOURCE_AI` for the
machine, accountant/client for people) — but the tag was **recorded and
displayed nowhere**, so an accountant had no way to tell a machine-written line
from their own. In a professional ledger that is not acceptable, and it makes
review impossible to target. Each row now carries an author badge (Analee / You
/ Client), the page summarises who explained what, and `?view=ai` shows **only**
what the machine wrote so its output can be spot-checked without re-reading the
statement. Corrections made there are kept and become next month's precedent
via history matching.

**The repo is RE-FROZEN with the automation work inside the freeze.** The
frozen rules apply to `services/bulk_suggestions.py`,
`services/history_matching.py`, `services/auto_process.py`,
`services/accountant_fanout.py` and the batch path exactly as to everything
else. The remaining ideas (wiring the dormant
keyword/rule engine — note its seeded rules are English personal-finance
categories that do not fit an SA business chart; a durable job queue that
survives a worker restart; bank feeds or email ingestion) each require Festus's
explicit, scoped re-open — or belong in the embedded `analee/` module in
`booksxpert`, which has real double-entry GL posting and so does not carry
standalone Analee's single-`account_id` limitation.

### Scoped re-open + re-freeze record (Festus, 2026-09-06) — THE AUTOMATION TIERS

Festus merged SEVEN PRs himself on 2026-09-06 (#106–#111 at 21:04, and #113
at 21:11): Tier 0
"unblock automation", "Refuse to show invented numbers as analysis", Tier 1
"one AI call per batch", Tier 2 "the practice's own history decides before the
model does", Tier 3 "importing a statement starts the work", Tier 4 "one
accountant decision, applied to the whole family", and Tier 5 "show who
decided each row, and let the machine's work be reviewed". Reviewed and
verified 2026-09-07.

**This work is inside the CAPABILITY freeze** — `services/analyze_processing.py`
IS the Analyze Data pipeline, `history_matching.py` is learn-from-the-past, and
`bulk_suggestions.py` is suggestion logic; the freeze covers "any module whose
job is to analyze, learn, or suggest". Festus merging the PRs is the approval;
this entry is the record that was missing.

**What was CHANGED (the unfreeze scope):** `services/analyze_processing.py`,
`routes.py`, `templates/analyze.html`, `static/js/analyze/*`, plus five NEW
services — `bulk_suggestions.py`, `history_matching.py`, `auto_process.py`,
`accountant_fanout.py`, `client_explanation.py`.

**What stayed FROZEN and is untouched — machine-verified:**
`predictive_features.py`, `ai_utils.py`, `services/chart_of_accounts.py`,
`chart_seed_data.py`, `entity_chart_rules.py`, `entity_chart_schema.py`,
`utils/chart_of_accounts.py`, `reports/trial_balance_service.py`.
`protected_assets.py --check` passes; nothing in the lock drifted.

**Verified good (the reason this is worth having):**
- Two sources of FABRICATED CONFIDENCE were deleted — a hardcoded
  `overall_confidence: 0.85 / reliability_score: 0.80`, and a fallback that
  padded the suggestion list with arbitrary accounts at an invented
  `confidence: 0.5`. Same disease as the 2026-09-03 "Bank charges → Patents and
  Trademarks 0.42" report: never show an invented number as analysis.
- A real crash was fixed: `data.get('account_id', type=int)` — `dict.get()`
  takes no `type=` kwarg, so edits were not saving. That is Tier 0.
- Nothing removed: no route, no view function, no nav item, no menu.
- Full suite 331 passed / 0 failed (was 252 before the tiers).
- Tier 5 (#113) is PURE ADDITION: +218 lines, zero deletions, its own tests;
  protected_assets --check re-verified clean after it landed.

**⚠️ OPERATIVE WARNING — THESE TIERS ARE LIVE, NOT DARK.** Unlike every other
recent Analee change, most of this has no feature flag:
- `history_matching`, `bulk_suggestions`, `accountant_fanout` are called
  unconditionally — they change how Analee analyses the moment `main` deploys.
- `ANALEE_AUTOPROCESS_ON_IMPORT` DOES exist but **defaults to `'1'` (ON)**, so
  importing a statement now auto-starts AI processing without the user pressing
  anything, and spends Anthropic tokens per import. Set it to `0` to stop that.

**On "Auto-process next 10" (checked, and it is HONEST):** the button label is
correct. `batchProcessor.js` still sends `batch_size: 10` and the route honours
it. `ANALYZE_BATCH_SIZE = 25` is only the server-side default for callers that
pass no size — in practice the auto-process-on-import path
(`auto_process.py` calls `process_transaction_batch` without a size), matching
`BULK_SUGGESTION_BATCH = 25`, one AI call per 25 rows. The frozen "10 at a time"
work loop is preserved as `ANALYZE_PAGE_SIZE = 10`.

**THE ENGINE IS RE-FROZEN** with these tiers inside the freeze. Any further
change to the analysis/learning/suggestion capability needs Festus's explicit,
scoped re-open, recorded here as this entry is.

---

## PROTECTED ASSETS — FROZEN (do not touch without Festus's explicit approval)

The **chart of accounts** (and the trial-balance core) is frozen and machine-
enforced — a critical asset Festus has lost before. Do NOT change these as a side
effect of another task; only a deliberate, Festus-approved re-freeze lands a
change.

Frozen: `services/chart_of_accounts.py`, `services/chart_seed_data.py`,
`services/entity_chart_rules.py`, `services/entity_chart_schema.py`,
`utils/chart_of_accounts.py`, `reports/trial_balance_service.py`.

Enforcement: `protected_assets.py` + `protected_assets.lock.json` +
`tests/test_protected_assets_lock.py` fail the build if any frozen asset drifts
(`python protected_assets.py --check` is the CLI gate). To land an APPROVED
change: `python protected_assets.py --authorized-by "Festus: <reason>"` then
commit the new lock. Full policy: `docs/PROTECTED_ASSETS.md`.

---

## Company development workflow (standing — Festus, 2026-05-17)

**Mandatory for every coding task. No step skipped.** The corporation-wide
authoritative copy lives in `autonomusFV/CLAUDE.md`.

| Step | Name | What the agent does |
|------|------|---------------------|
| 1 | **Planning Audit** | Inspect current state, dependencies, callers, and contracts that must be preserved. No guesses — file paths and shapes must be verified in the repo. |
| 2 | **Change-Impact Audit** | After drafting the plan, assess blast radius: routes/menus/features affected, migrations, runtime implications. Revise the plan until side effects are understood and controlled. |
| 3 | **Present Plan** | Present the plan to Festus with both audits stated explicitly. **Wait for approval** — no implementation begins without it. |
| 4 | **Implement** | Execute on a feature branch off **`develop`** only. One logical change per commit; avoid broad refactors inside a fix. |
| 5 | **Test** | Run targeted tests and checks. No success claims without evidence. Mark uncertain claims with `?`. |
| 6 | **Post-Engagement Audit** | Verify the task goal was met and nothing existing was damaged (imports, routes, menus, contracts). Report what was validated and what could not be validated here. |
| 7 | **Merge Control** | Open PR → **`develop`** only. **Festus decides** when to merge and when `develop` → `main` promotion happens — never by the agent without his explicit approval. |
| 8 | **Final Status** | Close every task with one of: **READY** / **READY WITH RISKS** / **NOT READY** — with brief justification. |

The three audits in steps 1, 2, and 6 are non-negotiable even under an
autonomous directive (see below).

**Honesty is non-negotiable.** Do not describe work as complete before
verification finishes.

## Git branches & pull requests (standing — Festus, 2026-06-28)

**Agents must NEVER open a PR against `main`.** All agent work targets
**`develop` only**.

Promotion flow:
1. Agent opens PR → **`develop`** (draft or ready, per instruction).
2. Festus reviews, merges, and tests on **`develop`** (e.g. Railway staging).
3. Only after Festus is satisfied does **`develop` → `main`** promotion happen —
   never by the agent without Festus's explicit approval.

Rules:
- **Do not** set `base_branch: main` on PRs.
- **Do not** push feature branches intended to merge straight into `main`.
- **Do** branch off `develop` using `cursor/<descriptive-name>-<suffix>`.
- If work was accidentally merged to `main` first, **stop** and report it to
  Festus; land the same changes on `develop` via a proper PR — do not repeat
  direct-to-main merges.

## Autonomous Directive (standing — non-negotiable)

When Festus issues an **autonomous directive** ("go on autonomously", "just do
it", "only ask what you cannot decide", etc.), that autonomy is **always
contingent on the Company Development Workflow above** — autonomy **never**
waives any step, especially the Planning Audit (1), Change-Impact Audit (2), and
Post-Engagement Audit (6).

**Each time Festus issues an autonomous directive, the agent MUST restate this
contingency in the reply** — this restatement is a rule, not a courtesy.

Operating under an autonomous directive means:
- **Do** proceed and decide what can be decided from the code or sensible defaults.
- **Still always** run the full workflow (audits in steps 1, 2, and 6 on every
  change) so existing application wins are not damaged.
- **Still** branch off `develop`, PR to `develop` only, and **never** promote
  `develop` → `main` without Festus's explicit approval.
- **Stop and ask only** for questions/decisions that genuinely cannot be made
  without Festus.

## Chart reconciliation with BooksXperts (trial-balance handoff)

Analee exports a trial balance that BooksXperts imports **by account link**; the
matched account's subcategory drives AFS classification. A guard keeps the two
charts reconciled: `services/chart_reconciliation.py` +
`reconciliation/booksxperts_chart_reference.json` +
`tests/test_chart_reconciliation.py` fail the build if Analee emits a link
BooksXperts lacks (→ suspense) or classifies a shared link differently
(→ misclassified). When BooksXperts' chart changes (Festus-approved), regenerate
the reference with `scripts/refresh_booksxperts_chart_reference.py`. Full detail:
`docs/CHART_RECONCILIATION.md`.
