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
  running. Each still goes through the three audits below.
- **Not allowed without Festus re-opening the repo:** new features, refactors, or
  cosmetic churn — **including the previously-planned internal ERF/ASF/ESF symbol
  rename (#3), which is now formally DROPPED** (see
  `docs/PERSONA_AND_TOOL_NAMING.md`). Polishing the internals of a frozen app is
  wasted effort.
- New ideas for Analee go to **booksxpert** (the embedded module), not here.

---

## Workflow & Audits Directive (standing)

Every change goes through three audits, and nothing is committed/pushed that
skips them:

1. **Pre-planning audit** — before proposing a plan, inspect the current state,
   dependencies, callers, and downstream consumers of the code in scope.
   Establish the exact contracts that must be preserved.
2. **Pre-implementation audit** — before editing, re-confirm the blast radius of
   the specific edit (who imports it, what shape callers expect, migration/runtime
   implications) so the change cannot silently break a consumer.
3. **Post-implementation impact assessment** — after editing, verify nothing was
   damaged (compile/import checks, targeted tests, contract checks) and report
   what was validated and what could not be validated here.

Changes are kept small and separate (one logical change per commit), and broad
refactors are avoided inside a fix.

## Autonomous Directive (standing — non-negotiable)

When Festus issues an **autonomous directive** ("go on autonomously", "just do
it", "only ask what you cannot decide", etc.), that autonomy is **always
contingent on the Workflow & Audits Directive above** — autonomy **never**
waives the three audits (pre-planning, pre-implementation, post-implementation).

**Each time Festus issues an autonomous directive, the agent MUST restate this
contingency in the reply** — this restatement is a rule, not a courtesy.

Operating under an autonomous directive means:
- **Do** proceed and decide what can be decided from the code or sensible defaults.
- **Still always** run all three audits on every change so existing application
  wins are not damaged.
- **Stop and ask only** for questions/decisions that genuinely cannot be made
  without Festus.
