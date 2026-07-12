# Project Working Agreement

## STANDING DEV WORKFLOW — comply with the Enhanced Development Workflow (authoritative: `autonomusFV/CLAUDE.md`)

Every code change in this repo follows the corporation-wide **Enhanced Development
Workflow with Within-Loop Memory** and its four refinements. The authoritative copy
lives in `autonomusFV/CLAUDE.md`; the product-specific rules in this file are its
specifics and never override it. In brief:
- **Memory Checkpoint → Planning Audit → Change-Impact Audit → (present) → implement
  on the dev branch → test → Learning Checkpoint → Post-Engagement Audit → merge
  control.** Never merge to `main` without Festus's explicit word.
- **A — tier it:** FULL ceremony for money/ledger/AFS/fiduciary/auth/schema/
  protected-assets/`main`-promotion/hard-to-reverse changes; LITE (checkpoint → one
  audit → test → one-paragraph result) for trivial reversible ones. Less ceremony,
  never less safety.
- **B — memory informs, git/tests decide:** verify any remembered fact against the
  live repo before acting on it.
- **C — autonomous directives:** under "go", present audits+plan and proceed the same
  turn; still never merge without Festus.
- **D — contradiction gate:** a change that breaks a locked/fiduciary principle STOPS
  and returns to Festus — not audit-and-proceed.

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

## Autonomous Directive (standing)

When the user issues an **autonomous directive** ("go on autonomously", "just do
it", etc.), the autonomous directive is **contingent on the Workflow & Audits
Directive above** — autonomy never waives the three audits. Each time the user
issues an autonomous directive, explicitly note that the autonomous directive is
contingent on our workflow and audits directive, then proceed, asking only for
decisions that cannot be made from the code or sensible defaults.
