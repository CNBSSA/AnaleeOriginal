# AGENTS.md — AnaleeOriginal (Analee)

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
