# Project Working Agreement

## ❄️ STATUS: FROZEN — MAINTENANCE ONLY (Festus, 2026-06-26)

**This standalone Analee repo is frozen.** It is no longer the place for new
feature work. The canonical Analee is the **embedded `analee/` module inside
`CNBSSA/booksxpert`** (Django, multi-tenant, real GL/journal posting, plus the
BooksXperts Assistant). Standalone Analee and the embedded one were investigated
side by side (2026-06-26): the embedded version is equal-or-better on every
capability, and the **one** thing standalone had that it lacked — **Recall**
(similarity-based "find my similar past transactions", formerly *ERF*) — has been
**harvested into `booksxpert/analee/services/recall.py`** with an agent tool and
tests. With that extracted, this repo's reason to grow is gone.

**Rules while frozen:**
- **Allowed:** security fixes, dependency CVEs, and keeping the deployed instance
  running. Each still goes through the company development workflow below.
- **Not allowed without Festus re-opening the repo:** new features, refactors, or
  cosmetic churn — **including the previously-planned internal ERF/ASF/ESF symbol
  rename (#3), which is now formally DROPPED** (see
  `docs/PERSONA_AND_TOOL_NAMING.md`). Polishing the internals of a frozen app is
  wasted effort.
- New ideas for Analee go to **booksxpert** (the embedded module), not here.

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
