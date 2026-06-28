# Analee Invoices

**Study: surgically removing invoice and accrual features from Analee**

| Field | Value |
|-------|-------|
| Status | Draft study (no code changes yet) |
| Repo | `AnaleeOriginal` (standalone Analee) |
| Date | 2026-06-28 |
| Audience | Product, engineering, Festus |

---

## 1. Purpose

This document studies how to **surgically remove all invoice-related capability** from **standalone Analee** and align the product with a single, narrow mission:

> **Analee processes bank statements for cash-basis users and produces a trial balance that can be uploaded or transmitted to BooksXperts or The Accountants.**

Analee is **not** a general ledger, **not** an accrual accounting system, and **not** an invoice platform. It is a **bank-statement analysis and categorisation tool** whose primary deliverable is a **BooksXperts-compatible trial balance** (see PR #22 — export on `/reports/trial-balance`).

---

## 2. Product boundary (target state)

### 2.1 What Analee **is**

| Capability | Description |
|------------|-------------|
| Bank upload | SA bank CSV/XLSX, format auto-detection |
| Analyse | Assign chart account + explanation per line (AI-assisted) |
| Chart of accounts | Entity-scoped master chart (BooksXperts parity links) |
| Trial balance | FY-scoped balances derived from categorised bank lines |
| **Export / transmit TB** | `Link`, `Account Name`, `Amount` → BooksXperts or The Accountants |

### 2.2 What Analee **is not**

| Excluded | Reason |
|----------|--------|
| Customer / supplier **invoices** | Accrual workflow; belongs in BooksXperts |
| Accounts receivable / payable **control** | No debtors/creditors ledger in Analee |
| Receipts / payments **documents** | Cash settlement is implied by bank categorisation |
| Invoice ↔ bank **matching** | BooksXperts `analee/` module only |
| Official **AFS / VAT201** | Downstream systems compile from real GL |
| Double-entry **journal engine** | Analee holds single-sided categorised lines |

### 2.3 Cash basis vs accrual

For the target user, **revenue and expenses are recognised when cash moves** (bank line categorised to income/expense). There is:

- No separate “invoice raised” event
- No open debtor/creditor balance to reconcile
- No stock/COGS split unless the user manually categorises inventory movements (unusual for pure cash-basis micro businesses)

The **trial balance** is therefore a **summary of categorised bank activity** for a period, not an audit-grade accrual GL.

---

## 3. Repository map — where “invoice” lives today

### 3.1 Standalone `AnaleeOriginal` — **almost no invoice features**

A full-code search (`invoice`, `Invoice`, `receivable`, `payable`, `accrual`) shows **no invoice module, routes, models, or UI** in production code.

| Location | Finding | Invoice? |
|----------|---------|----------|
| `routes.py`, `models.py` | Bank upload, analyse, settings | No |
| `bank_statements/` | Statement import only | No |
| `reports/routes.py` | Cashbook, GL, TB, balance sheet, income statement | No invoice logic |
| `historical_data/ai_suggestions.py` | Fallback keyword `'invoice': 'Invoice payment'` | **Hint only** |
| `services/chart_seed_data.py` | Master chart includes Trade Receivables, Accrued Income, Trade Payables | **Chart accounts**, not features |
| `ignore/Pasted-*.txt` | Early prototype notes (invoice OCR, PDF export) | **Discarded ideas**, not shipped |

**Conclusion:** There is **nothing large to “rip out”** for invoices in standalone Analee — the work is **prevention, clarification, and trimming accrual-shaped surfaces**, not deleting an invoice app.

### 3.2 BooksXperts `booksxpert/analee/` — **not in scope for this repo**

BooksXperts embeds a **different** Analee that **does** integrate with invoicing:

- `link_invoice`, `link_receipt`, `InvoiceReceiptLink`
- Shadow invoice queue (legacy / reduced)
- URLs such as `uncategorised-invoices` (bank analyse screen naming)
- Full double-entry bank posting

That code **must stay in BooksXperts** for accrual clients. It should **not** be ported back into standalone Analee. This study applies to **`AnaleeOriginal` only**.

### 3.3 The Accountants — **downstream consumer, not source of invoices**

[The Accountants](https://github.com/CNBSSA/accountants) ingests **trial balances** (Excel/CSV), maps accounts, posts adjusting journals, and compiles AFS. It does not expect Analee to send invoices — only **balanced TB lines**.

---

## 4. Inventory — features to treat as “invoice/accrual-adjacent”

Even without an invoice app, some surfaces **imply accrual accounting** or **confuse** cash-basis users.

### 4.1 Reports menu (`templates/base.html`)

| Report | Accrual signal | Recommendation |
|--------|----------------|----------------|
| **Cashbook** | Low — bank-centric | **Keep** (primary working view) |
| **Trial Balance** | Neutral — handoff artefact | **Keep** + export (done) |
| **General Ledger** | Medium — sounds like full GL | **Rename or hide** → “Account activity” / cash-basis summary |
| **Financial Position** | High — balance sheet | **Hide or mark “preview only”** — misleading without accrual |
| **Income Statement** | Medium — P&L from bank only | **Keep with label** “Cash-based summary (not accrual AFS)” |

### 4.2 Chart of accounts (`chart_seed_data.py`)

Master chart includes accrual-native accounts, for example:

- `ca.300.000` Trade Receivables  
- `ca.330.000` Accrued Income  
- `cl.500.000` Trade Payables  

For **cash-basis Analee users**, these accounts should **rarely appear** in the trial balance. Options:

| Option | Effort | Effect |
|--------|--------|--------|
| **A. No change** | None | User can still pick AR/AP in analyse — **avoid** |
| **B. Cash-basis chart profile** | Medium | Copy master chart minus AR/AP/accrued; default for new users |
| **C. UI guardrails** | Low | Warn when categorising bank line to AR/AP: “Cash-basis: use income/expense, not receivables” |

**Recommendation:** **C** first, **B** when entity onboarding is refined.

### 4.3 AI / NLP remnants

| File | Item | Action |
|------|------|--------|
| `historical_data/ai_suggestions.py` | `'invoice': 'Invoice payment'` | Replace with `'payment': 'Customer payment'` or remove |
| `ignore/` prototype notes | Invoice OCR, receipt parsing | **Do not implement**; archive or delete `ignore/` from product docs |
| Chat / iCountant copy | Any “upload invoices” language | Audit templates; bank-only wording |

### 4.4 Duplicate upload entry points

Navigation has both **Bank Statements** and **Upload Data** (`main.upload`). Consolidating on **bank-only** upload reduces the impression that Analee accepts invoice files.

---

## 5. Surgical removal plan (phased)

Each phase is independently shippable. **No phase adds invoice logic.**

### Phase 0 — Document and freeze scope (this document)

- [x] Name the boundary: bank → TB → BooksXperts / Accountants  
- [ ] Festus sign-off on “no invoices in standalone Analee”  
- [ ] Add one-line mission to `README` / landing copy  

### Phase 1 — UX truthfulness (low risk)

| Task | Files | Notes |
|------|-------|-------|
| Relabel reports | `templates/base.html`, report templates | Cash-basis disclaimers |
| Hide Financial Position from default nav | `base.html` | Or move under “Advanced preview” |
| Remove / merge duplicate upload | `routes.py`, `base.html` | Single bank upload path |
| Export TB for Accountants | `trial_balance_service.py` | Same columns as BooksXperts; footer text names both targets |

**Acceptance:** New user cannot find “invoice” in nav or help.

### Phase 2 — NLP and copy cleanup

| Task | Files |
|------|-------|
| Remove invoice keyword fallback | `historical_data/ai_suggestions.py` |
| Scan `templates/`, `chat/`, `icountant.html` for invoice wording | Various |
| Reject invoice file types if any generic upload accepts PDF “invoices” | `bank_statements/upload_validator.py` |

**Acceptance:** `rg -i invoice` on `*.py` + `templates/` returns zero user-facing strings (except this doc).

### Phase 3 — Cash-basis chart guardrails

| Task | Files |
|------|-------|
| Define `ACCRUAL_CONTROL_LINKS` constant | `services/chart_of_accounts.py` or new `cash_basis.py` |
| Warn on analyse when user picks AR/AP/accrued | `routes.py` / analyse JS |
| Optional: filter AR/AP out of analyse dropdown | `analyze.html`, API |

**Acceptance:** TB export for a typical user contains only bank, cash, income, expense, equity — no AR/AP balances.

### Phase 4 — Report rationalisation

| Task | Rationale |
|------|-----------|
| Drop or stub **General Ledger** double-entry language | Analee has no journals |
| Keep **Cashbook** as chronological bank view | Matches user mental model |
| **Trial balance** = only formal “output” report | Aligns with handoff |
| Deprecate **Financial Position** / **Income Statement** OR generate from TB with watermark “Draft — not for filing” | Avoid competing with BooksXperts AFS |

### Phase 5 — Transmission (post manual export)

| Step | Mechanism |
|------|-----------|
| Now | Download `.xlsx` → manual upload in BooksXperts `dataimports` or Accountants portal |
| Next | Signed export URL / API payload: `{ company_id, as_at, rows: [{link, name, amount}] }` |
| Later | OAuth handoff Analee → BooksXperts company (same `account_link` map) |

**BooksXperts import contract** (already live):

```
Link | Account Name | Amount
```

Positive = debit, negative = credit, rows must sum to zero.

**The Accountants** accepts similar staged TB (account name + amount / debit-credit columns); Analee export should remain compatible with both via the same three-column file.

### Phase 6 — Explicit non-goals (do not build in Analee)

- Invoice PDF upload / OCR  
- Customer master + invoicing  
- VAT tax invoices (§16(2) input VAT needs valid tax invoice)  
- Bank ↔ invoice matching (BooksXperts only)  
- Stock / inventory COGS automation (BooksXperts inventory module)  

---

## 6. What we keep unchanged

| Area | Why |
|------|-----|
| Bank statement pipeline | Core product |
| Analyse + batch AI | Core product |
| Entity-scoped chart + lock rules | BooksXperts parity for `account_link` |
| Trial balance + Excel export | Primary handoff |
| Historical data / recall | Improves categorisation quality |
| iCountant / chat (bank-focused) | UX for categorisation |

---

## 7. Risk register

| Risk | Mitigation |
|------|------------|
| User categorises sales twice (bank + manual TB line) | Export checklist: “one line per economic event”; education |
| TB does not balance | Show imbalance on TB page (already partially done); block export optional |
| User expects AFS from Analee | Hide accrual reports; point to BooksXperts / Accountants |
| Confusion with BooksXperts “Analee” module | Rename in docs: **“Analee Bank”** (standalone) vs **“BooksXperts Bank Reconciliation”** (embedded) |
| AR/AP on chart tempts wrong categorisation | Phase 3 guardrails |

---

## 8. Verification checklist (post-implementation)

```bash
# No user-facing invoice strings in app code
rg -i 'invoice' --glob '*.py' --glob '*.html' --glob '*.js' \
  --glob '!docs/**' --glob '!ignore/**'

# Trial balance export still matches BooksXperts template headers
pytest tests/test_trial_balance_export.py -q

# Full regression
pytest tests/ -q
```

Manual:

1. Upload bank statement → analyse all lines to income/expense/bank  
2. Open Trial Balance → Export for BooksXperts  
3. Upload file in BooksXperts Data Imports → map links → post  
4. Repeat upload in The Accountants portal (if available)  

---

## 9. Summary

| Question | Answer |
|----------|--------|
| Does standalone Analee have invoices today? | **No** — only NLP hint + chart account names |
| What is the main “removal” work? | **UX, chart guardrails, accrual report hiding**, not deleting modules |
| What is the primary output? | **Trial balance Excel** → BooksXperts or The Accountants |
| Where do invoices belong? | **BooksXperts** (full GL + bank match) |
| Smallest next code PR? | **Phase 1** — nav labels, hide Financial Position, dual-target export footer |

---

## 10. References

| Resource | Path |
|----------|------|
| Trial balance export | `reports/trial_balance_service.py`, PR #22 |
| BooksXperts TB import | `booksxpert/dataimports/views.py`, `sample_templates.py` |
| BooksXperts bank + invoice reconciliation | `booksxpert/docs/RECONCILIATION_GOALS.md` |
| Accountants TB intake | `accountants/dataimports/`, `portal/views.py` |
| Analee master chart links | `services/chart_seed_data.py` |

---

*End of study — awaiting product sign-off before Phase 1 implementation.*
