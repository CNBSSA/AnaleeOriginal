# Analee — Bank & Invoice Upload Guide

These templates import into Analee's transaction importer. Both the **bank
statement** and **invoices** templates use the **same 3 required columns** —
Analee treats every line as a dated, signed money movement.

## The format (required)

| Column | Required | Notes |
|---|---|---|
| **Date** | yes | The transaction / invoice date |
| **Description** | yes | Narration, payee, or invoice number |
| **Amount** | yes | Signed number (see sign rule below) |

- Column **names** must be `Date`, `Description`, `Amount` (case-insensitive, any
  order). **Keep the header row.**
- For **.xlsx**, put data on the **first sheet**. (Our templates include a second
  "Instructions" sheet — that's fine, the importer only reads the first.)
- **Extra columns are ignored**, so you can keep your own reference columns.

## Which file to use on which page

| Page | .xlsx | .csv |
|---|---|---|
| **Upload Data** | ✅ | ✅ |
| **Bank Statements** | ✅ | ✅ |
| **Historical Data** | ✅ | ✅ |

**Both `.xlsx` and `.csv` work on every upload page.** Pick whichever is easier
for you.

## The sign rule (important)

Analee reads the **sign** of `Amount`:

- **Positive = money IN** — deposits, customer payments, **sales invoices** (income / receivable).
- **Negative = money OUT** — withdrawals, bank charges, **bills you pay** (expense / payable).

So in the **invoice** template: a sales invoice you raise is **positive**; a
supplier bill you owe is **negative**; a credit note / refund flips the sign.

## Amount — what's accepted vs skipped

| You type | Result |
|---|---|
| `1500.00`, `-200.00` | ✅ imported as-is |
| `$1,999.95` | ✅ `$` and thousands `,` are stripped → `1999.95` |
| ` 50.00 ` (spaces) | ✅ trimmed |
| `0` | ✅ imported as zero |
| `(200.00)` (parentheses) | ❌ **row skipped** — use a minus sign: `-200.00` |
| `1.234,56` (European) | ❌ misread — use US format `1234.56` |
| `abc`, blank | ❌ **row skipped** |

## Date — what's accepted vs skipped

| You type | Result |
|---|---|
| `2026-01-31` (recommended) | ✅ |
| `31/01/2026`, `01/31/2026` | ✅ (prefer `YYYY-MM-DD` to avoid day/month ambiguity) |
| `31-Jan-2026`, `Jan 31 2026` | ✅ |
| `not-a-date`, blank | ❌ **row skipped** |

## Good to know (every-scenario checklist)

- **Bad rows are skipped individually** — the rest of the file still imports, so a
  couple of typos won't fail the whole upload.
- **One account per upload** — you choose the bank/account on the upload page; all
  rows in the file are imported against it.
- **CSV encoding**: UTF-8 is best; Latin-1 is also accepted automatically.
- **Max file size: 40 MB.** Allowed types: `.xlsx` and `.csv` (both on every page).
- **Scanned PDFs / photos** of statements or receipts do **not** go here — use the
  **Scan Receipt** / **Import PDF Statement** pages (OCR) for those.
- **Duplicates**: re-uploading the same file imports the rows again (the OCR
  review screen flags likely duplicates, but spreadsheet import does not), so
  avoid importing the same statement twice.

## Files in this pack

- `Analee_Bank_Statement_Template.xlsx` / `.csv` — bank statement sample (15 rows
  covering deposits, withdrawals, charges, `$`/comma amounts, decimals, zero, large).
- `Analee_Invoices_Template.xlsx` / `.csv` — invoice sample (sales invoices,
  supplier bills, a credit note).
- Replace the sample rows with your own; keep the header row and the 3 columns.
