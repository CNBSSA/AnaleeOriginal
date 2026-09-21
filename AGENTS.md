# AGENTS.md — AnaleeOriginal (Analee)

## ⛔ CURRENT SCOPE — SA PRODUCTS ARE FIX-ONLY (Festus, 2026-09-21, until he changes it)

> *"For now, until I change it, no new development in all South African products.
> We only fix existing features and functionalities. No removal of features and
> menus, but fix all that are not working well."*

**This binds EVERY agent working in this repo — internal or external, Claude,
Cursor, Copilot, Codex, or any other model.** It is not Claude-specific and it is
not advisory.

- **Do:** fix what exists and does not work, or does not work well — defects,
  correctness, reliability, performance, usability. **Finishing a feature that
  shipped half-working IS a fix.**
- **Do NOT:** add new features, modules or surfaces, however small or however good
  the idea. Capture it in the PR description or the backlog, say so plainly, and
  go back to fixing.
- **Remove nothing.** The owner restated this in the same breath — see rule 2
  below. Not as a side effect, not as a "simplification", not by hiding it behind
  a flag or a permission.
- **The mandate is ACTIVE.** *"Fix all that are not working well"* means go and
  find what is broken. Audit findings are to be **fixed**, not merely logged.
- **Grey edge, ruled in advance:** if you cannot tell whether something is a fix
  or a new feature, it is a **new feature** — stop and ask the owner.

Full text: this repo's `CLAUDE.md` → STANDING DIRECTIVE. Corporation-wide
authoritative copy: `autonomusFV/CLAUDE.md`.


Start with `CLAUDE.md` for product context, the app map, and the full
development workflow. This file restates the **non-negotiable guardrails** every
agent (Cursor, Claude, or otherwise) must follow. The same rules live in
`.cursor/rules/cnbssa-guardrails.mdc` (always applied).

## Non-negotiable rules

1. **Branch policy — `develop` only.** Branch from `develop`; open every PR with
   **base = `develop`**. **NEVER** open, retarget, or push a PR against `main`.
   `main` is production; `develop → main` is the owner's decision alone. If your
   work is based on `main`, re-base it onto `develop` before opening the PR.
2. **Additive-first.** No existing route, menu/nav item, page, button, or feature
   may disappear, be hidden, or be replaced as a side effect. Add new screens;
   do not redirect or remove existing ones without the owner's prior approval.
3. **Minimal scope.** Keep each PR to the task described. No unrelated refactors,
   no scope creep. Significant changes (billing/pricing behaviour, auth, schema
   migrations, access control, anything hard to reverse) must be called out and
   approved by the owner before they land.
4. **No invented facts.** Never publish prices, contacts, dates, or claims you
   have not verified in-repo or been given by the owner. If unverified, leave it
   out — do not guess.
5. **Honesty.** No success claims without running the tests. Mark uncertain
   claims with `?`.

## Running tests

See `CLAUDE.md` for how to run this repo's test suite. Run the full suite (and
any boot/check step) before declaring work done.
