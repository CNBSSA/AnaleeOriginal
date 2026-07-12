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

## ⚖️ COMPANY DEVELOPMENT WORKFLOW — MANDATORY FOR EVERY APPLICATION CHANGE

**Read the canonical workflow document first:** `autonomusFV/docs/DEVELOPMENT_WORKFLOW_WITH_MEMORY.md`

This is the authoritative, permanent policy — effective 2026-07-10 — for every agent across all 12 CNBSSA repositories. All development (code, templates, scripts, configuration, documentation) follows this single workflow:

**Core structure (9 steps):**
1. **Memory Checkpoint** — consult prior learnings before starting
2. **Planning Audit** — map current state, review assumptions
3. **Change Impact Audit** — assess blast radius and risks (contextualized against history)
4. **Present the Plan** — await Festus approval before coding
5. **Implementation** — develop on designated branch only (never `main`)
6. **Testing** — verify new functionality and prior failure modes
7. **Learning Checkpoint** — capture what was learned for next time
8. **Post-Engagement Audit** — confirm outcome vs. expectations
9. **Merge Control** — Festus decides when to promote to `main`

**Key rule:** All development happens on `develop` (or feature branch off `develop`). Promotion to `main` only with Festus's explicit approval after testing.

**Autonomous directives are contingent on this workflow.** Autonomy never waives the audits or branch/merge control — it accelerates execution on things the agent can decide, but the workflow and testing remain mandatory.

**Do not skip audits. Do not claim success without testing. Do not develop in `main`.**

For detailed rationale, examples, and the within-loop memory mechanisms, read the canonical document: `autonomusFV/docs/DEVELOPMENT_WORKFLOW_WITH_MEMORY.md` (permanent policy, locked by Festus, 2026-07-10).
