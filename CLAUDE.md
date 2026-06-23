# Project Working Agreement

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
