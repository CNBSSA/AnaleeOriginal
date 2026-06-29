# Analee — persona & tool naming (decision + remaining-work plan)

**Decision (Festus, 2026-06-26): one face, three tools.**
The user meets **one persona — Analee** ("Analee, your AI accountant").
The three internal mechanisms (formerly the branded features **ERF / ASF / ESF**)
are **never marketed separately** again. They survive as **tools** behind Analee,
renamed for clarity. **Nothing is deleted — the muscles stay; only the brand names retire.**

## The naming map

| Old brand / code | New user-facing name | What it does | Mechanism |
|---|---|---|---|
| ERF — Explanation Recognition Feature | **Recall** | finds *your* similar past, already-explained transactions to reuse | pure algorithm (text similarity ≥0.70 / ≥0.95); no AI call |
| ASF — Account Suggestion Feature | **Account Match** | suggests the right ledger account, top-3 + confidence | AI (`claude-opus-4-7`) + keyword/rule fallback |
| ESF — Explanation Suggestion Feature | **Auto-Explain** | drafts the explanation/narrative for an entry | AI + history fallback |
| iCountant (assistant persona) | **Analee** | the one assistant the user talks to; runs the three tools + builds the journal | orchestrator |

## Why we keep three tools (not collapse into one)

iCountant/Analee is the *face*; the three are *muscles*. They cannot be deleted:
- **Reused beyond the assistant** — they feed the 5-tier `PredictiveEngine`
  (`predictive_utils.py`) and the batch/bank-statement flows, not only the chat.
- **Different costs** — Recall is free/instant (no API); Account Match & Auto-Explain
  cost a Claude call. The engine tries the free one first; merging loses that.
- **Independent failure & fallback** — each degrades on its own.
- **Agent-native standard** — these are exactly the discrete tools an agent calls
  (`recall_similar`, `match_account`, `write_explanation`).

## DONE — user-facing rename (branch `claude/analee-retire-feature-acronyms`)

Inspection-verified, zero logic touched (could not run the suite in-sandbox — no venv):
- `templates/analyze.html` — tutorial headings: ERF/ASF/ESF → Recall / Account Match / Auto-Explain.
- `templates/icountant.html` — title → "Analee — your AI accountant".
- `templates/base.html` — nav link → "Ask Analee".
- `routes.py` — in-progress flash text → "Analee is processing…" (string only; `py_compile` clean).

## ~~REMAINING — internal rename~~ — DROPPED (repo frozen 2026-06-26)

**This work (#3) is no longer planned.** AnaleeOriginal was frozen to maintenance-only
on 2026-06-26 once **Recall** was harvested into `booksxpert/analee` (see the FREEZE
banner in `CLAUDE.md`). Renaming the internals of a frozen app is wasted effort, so the
tiers below are kept only as a historical record — **do not action them** unless Festus
re-opens the repo.

~~Do this only where the Flask app boots and the pytest suite runs (local venv or PC).~~

### Tier 1 — zero-logic (safe; cosmetic clarity)
Docstrings, comments, and **log-message prefixes** still say ERF/ASF/ESF in:
`predictive_features.py`, `ai_utils.py` (e.g. `logger.info(f"ASF: …")`), `routes.py`,
`icountant.py`, `predictive_utils.py`. Rename text → Recall / Account Match / Auto-Explain.
No behaviour change. **Note:** the public functions are *already* well-named
(`predict_account`, `find_similar_transactions`, `suggest_explanation`,
`suggest_account`) — **do not rename functions**; only the acronym text in their docs/logs.

### Tier 2 — logic-bearing (needs the suite green before + after)
- **`AI_FEATURES_CONFIG`** (`predictive_utils.py:504`) keys `'ERF' / 'ASF' / 'ESF'`
  are read by live logic (`['ERF']['text_threshold']`, `['ASF']['enabled']`, …) and by
  `tests/rollback_verification.py` (12 refs, treats them as *protected core features*).
  **Recommendation: KEEP these keys as stable internal identifiers** (add a one-line
  comment mapping key→tool name). Clarity gain is marginal; the blast radius (every
  reader + the protected-core test) is not worth the risk. If renamed, change keys +
  all readers + the rollback test **atomically** in one commit, suite green either side.

### Tier 3 — data / route identifiers (defer; back-compat sensitive)
- Route name `main.icountant_interface` and URL — purely internal; rename only with a
  redirect from the old path so no bookmark/link breaks. Low priority.
- `ICountant` class (`routes.py:1088`, a placeholder) and `icountant.py` class — internal.
- **`'processed_by': 'iCountant'`** (`icountant.py:273`) is **stored on transaction
  records**. Existing rows carry `'iCountant'`. Do **not** blind-rename: either keep the
  historical tag, or write new rows as `'Analee'` while readers accept both. Treat as a
  data-migration decision, not a string swap.

## Acceptance for the remaining pass
1. App boots (`flask`/app import clean) and **full pytest suite green** before and after.
2. `grep -rniE '\b(ERF|ASF|ESF)\b'` returns nothing outside an intentional, commented
   back-compat shim (e.g. kept `AI_FEATURES_CONFIG` keys).
3. `tests/rollback_verification.py` still passes (its protected-core checks updated in lockstep if Tier 2 keys are renamed).
4. One logical change per commit; user-facing behaviour identical.
