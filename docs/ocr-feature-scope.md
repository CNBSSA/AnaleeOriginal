# OCR Ingestion — Feature Scope (Planning Only)

> Status: **scoping document, not implemented.** This describes how OCR-based
> ingestion (scanned receipts and PDF bank statements) would slot into the
> existing transaction pipeline, the options considered, the recommended
> approach, risks, and a phased plan. No code is changed by this document.

## 1. Goal

Let a user upload a **scanned receipt (image)** or a **PDF bank statement** and
have the system extract line items (date, description, amount) and turn them into
`Transaction` rows — reusing the existing import path rather than building a
parallel one.

## 2. Where it plugs in (existing pipeline)

The current ingestion contract is already well defined:

- `routes.py::process_transaction_rows(df, uploaded_file, user)` (around
  `routes.py:1140`) iterates a pandas DataFrame and builds `Transaction` objects
  from **`row['Date']`, `row['Description']`, `row['Amount']`**.
- `bank_statements/excel_reader.py` / `bank_statements/services.py` already read
  structured **Excel/CSV** into that shape.
- `UploadedFile` (models.py) tracks an upload and owns its `Transaction` rows
  (`file_id`), scoped per `user_id`.

**Design rule:** OCR is only a *new front-end* that produces the **same
normalized `{Date, Description, Amount}` rows**. Everything downstream
(persistence, trial balance, categorization, anomaly detection) stays unchanged.

```
upload (img/pdf) → validate → OCR/extract → normalize to {Date,Description,Amount}
   → REVIEW (human confirm) → process_transaction_rows() → existing app
```

The **review/confirm step is mandatory** (see Risks) — OCR output must never be
auto-committed without the user confirming, because misread amounts corrupt the
books.

## 3. Options considered

| Option | Pros | Cons |
|---|---|---|
| **A. Tesseract (pytesseract, local)** | Free, offline, no data egress | Raw text only; heavy custom layout/table parsing; weak on receipts/tables; needs the `tesseract-ocr` system binary |
| **B. Cloud document AI** (AWS Textract / Google Document AI / Azure) | Strong table & line-item extraction | Financial data leaves the environment (privacy); per-page cost; another vendor + keys |
| **C. Claude vision (Anthropic multimodal)** | **Reuses what we already have** — `anthropic` client, `ANTHROPIC_API_KEY`, and the new `config.CLAUDE_MODEL` indirection; one call can both extract *and* categorize; strong semantic parsing of messy receipts | Per-call cost; needs a vision-capable model id; still an external API (same trust boundary the app already uses for categorization) |

## 4. Recommendation

**Primary: Option C (Claude vision).** It fits the current architecture with the
least new surface area — the app already sends transaction text to Anthropic, so
the trust boundary is unchanged, and `config.CLAUDE_MODEL` gives a single place
to point at a vision-capable model. Extraction returns structured JSON
(`[{date, description, amount, confidence}]`) that normalizes directly into the
existing DataFrame contract.

**Optional later: Option A (Tesseract)** as an offline/no-egress fallback for
users who cannot send documents to an external API.

This keeps OCR consistent with the rest of the AI stack (centralized model id,
same client, same fallback philosophy).

## 5. Proposed architecture

1. **Upload endpoint** — `POST /upload/receipt` (and/or extend the bank-statement
   upload). Accepts `image/png|jpeg|webp` and `application/pdf`.
2. **Validation** — `bank_statements/upload_validator.py`-style checks: MIME/type
   allowlist, max file size, page-count cap for PDFs, per-user ownership.
3. **Preprocessing** —
   - Images: pass through (optionally downscale via Pillow, already a dependency).
   - PDFs: either send pages to Claude directly, or rasterize with `pdf2image`
     (needs `poppler`) / read text-based PDFs with `pdfplumber`/`pypdf`.
4. **Extraction (OCR)** — Claude vision call using `config.CLAUDE_MODEL`,
   prompted to return **strict JSON** line items with per-row confidence. Reuse
   the existing fallback discipline (return empty/partial on parse failure).
5. **Normalization** — map extracted rows to `{Date, Description, Amount}`,
   parse currency/locale, coerce dates; attach `confidence`.
6. **Review UI** — show extracted rows, flag low-confidence cells, let the user
   edit/approve. Nothing is written to `Transaction` until confirmation.
7. **Commit** — feed approved rows through `process_transaction_rows(...)` under a
   new `UploadedFile`, so they behave identically to spreadsheet imports.
8. **(Optional) Auto-categorize** — on commit, run the existing
   `categorize_transaction` / `predict_account` so OCR rows arrive pre-classified.

## 6. Data model

- Reuse **`UploadedFile`** for the upload record and `Transaction.file_id` linkage.
- Add (optional) a lightweight **`DocumentExtraction`** table (or JSON column) to
  hold raw extracted rows + confidence between extraction and confirmation, so a
  user can resume a review. Avoid persisting `Transaction` rows pre-confirmation.

## 7. New dependencies

- Option C: none new for images beyond `anthropic` (present) + `pillow` (present).
  PDFs: add `pypdf`/`pdfplumber` (text PDFs) and/or `pdf2image` + `poppler`
  (scanned PDFs).
- Option A (if added): `pytesseract` + `tesseract-ocr` system binary.

## 8. Security / privacy (financial documents are sensitive)

- **Data egress:** Options B/C send documents to a third party. Confirm this is
  acceptable for the user's compliance posture; document it. Option A avoids it.
- **Isolation & limits:** enforce `user_id` scoping, file type/size limits, PDF
  page caps, and rate/cost controls on the extraction call.
- **Retention:** decide whether to store the original upload or discard after
  extraction; store under access-controlled storage if kept.
- **Production protection:** respect existing `PROTECT_PRODUCTION` /
  `PREVENT_MODIFICATIONS` flags.

## 9. Accuracy & correctness safeguards

- Human-in-the-loop review before persistence (non-negotiable).
- Per-row confidence; visually flag anything below a threshold.
- Reconcile extracted line-item sum vs. any stated total on the document.
- De-duplicate against existing `Transaction` rows (date+amount+description) to
  avoid double-importing a statement already loaded via spreadsheet.
- Robust handling of: multi-page PDFs, rotated/low-quality scans, non-receipt
  images, multiple currencies, and varied date formats.

## 10. Phased delivery

- **Phase 1 (MVP):** single receipt **image** → Claude vision → JSON line items →
  review screen → commit via `process_transaction_rows`. No PDF, no auto-categorize.
- **Phase 2:** **PDF bank statements** (multi-page, tables) + de-duplication.
- **Phase 3:** auto-categorization integration (batch) and an optional offline
  Tesseract path.

## 11. Testing strategy

- Unit: normalization (currency/date parsing), JSON-extraction parsing with
  malformed model output, confidence thresholding, de-dup logic — all without
  live API calls (mock the client), mirroring the existing AI fallbacks.
- Integration: a handful of fixture receipts/statements asserting the normalized
  DataFrame matches the `{Date, Description, Amount}` contract.
- Regression: confirm the spreadsheet import path is untouched.

## 12. Open questions for the user (decisions needed before build)

1. **Privacy:** is sending financial documents to an external API (Claude/cloud)
   acceptable, or is an **offline-only** (Tesseract) path required?
2. **Documents:** primarily **receipt images**, **PDF statements**, or both first?
3. **Retention:** store original uploads, or discard after extraction?
4. **Volume:** expected documents/month (drives cost + rate-limit design)?
5. **Locale:** which currencies/date formats/languages must be supported?

_No implementation will start until these are answered and a normal
plan → pre-implementation audit → implementation → impact-assessment cycle is run._
