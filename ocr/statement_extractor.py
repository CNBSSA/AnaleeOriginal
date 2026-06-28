"""South African bank-statement extraction — Tier-1 (digital PDF) + Tier-2 (Claude Vision).

Orchestrates the highest-effective pipeline for SA bank statements:
  1. Try fast, free digital-PDF text parse (pypdf).
  2. Fall back to Claude Vision for scans / complex layouts.
  3. Run the Mathematical Integrity Gate before the user reviews rows.
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from config import OCR_MODEL
from nlp_utils import get_claude_client

from .pdf_text_extraction import PdfStatementError, extract_pdf_statement
from .statement_integrity import (
    ExtractionResult,
    ReportCard,
    StatementHeader,
    StatementLine,
    normalize_amount,
    self_audit,
)

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 32 * 1024 * 1024

_SA_BANK_STATEMENT_PROMPT = """You are an expert reader of South African bank statements.

Extract EVERY transaction line from the attached PDF. The statement may be from any
major SA bank (FNB, Standard Bank, Absa, Nedbank, Capitec, Investec, African Bank,
TymeBank, Discovery Bank, Bidvest, etc.).

Respond with JSON ONLY — no markdown fences, no commentary — exactly this shape:
{
  "bank": "<bank name or null>",
  "account_number": "<masked account number or null>",
  "period_start": "<YYYY-MM-DD or null>",
  "period_end": "<YYYY-MM-DD or null>",
  "opening_balance": "<amount as printed>",
  "closing_balance": "<amount as printed>",
  "lines": [
    {
      "date": "<as printed on the line>",
      "description": "<narration / payee / reference — not the date or amounts>",
      "amount": "<signed number: POSITIVE = money IN (credit/deposit), NEGATIVE = money OUT (debit/payment/fee)>",
      "balance": "<running balance after this line, or null>",
      "confidence": <0.0 to 1.0 — your certainty you read this row correctly>
    }
  ]
}

Rules (critical):
- SA dates are usually DD/MM/YYYY or "01 Jan 2026". Preserve meaning; do not swap day/month.
- Amounts use ZAR (R). Debits often show as trailing minus (e.g. "1 234.56-") or in a Debit column.
- If the statement has separate Debit and Credit columns, use ONE signed amount per row
  (debit/out = negative, credit/in = positive). Never double-count.
- Include card purchases, EFTs, fees, interest, reversals — every real transaction line.
- Do NOT include opening/closing balance rows, subtotals, "Brought Forward/Carried Forward",
  page headers, footers, or marketing text as transactions.
- Do NOT invent rows. Lower confidence when unsure.
- Read ALL pages. Multi-page statements must return ALL transaction lines.
"""


@dataclass
class BankStatementExtraction:
    """Outcome of the full extraction pipeline."""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    header: Optional[StatementHeader] = None
    report_card: Optional[ReportCard] = None
    method: str = ""
    error: Optional[str] = None
    error_code: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.rows)


def _parse_optional_balance(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return normalize_amount(text)
    except ValueError:
        return None


def _parse_claude_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start:end + 1])


def _payload_to_result(
    payload: dict,
    opening_override: Optional[Decimal] = None,
    closing_override: Optional[Decimal] = None,
) -> ExtractionResult:
    opening = opening_override
    closing = closing_override
    if opening is None:
        raw = payload.get("opening_balance")
        if raw not in (None, "", "null"):
            opening = normalize_amount(raw)
    if closing is None:
        raw = payload.get("closing_balance")
        if raw not in (None, "", "null"):
            closing = normalize_amount(raw)
    if opening is None or closing is None:
        raise ValueError(
            "opening and/or closing balance not found — enter them on the upload form."
        )

    lines: list[StatementLine] = []
    for raw in payload.get("lines") or []:
        if not isinstance(raw, dict):
            continue
        try:
            conf_raw = raw.get("confidence", 0.85)
            confidence = max(Decimal("0"), min(Decimal("1"), Decimal(str(conf_raw))))
            balance_raw = raw.get("balance")
            balance = None
            if balance_raw not in (None, "", "null"):
                balance = normalize_amount(balance_raw)
            lines.append(StatementLine(
                date=raw["date"],
                description=str(raw.get("description") or "").strip(),
                amount=raw["amount"],
                balance=balance,
                confidence=confidence,
            ))
        except Exception as exc:
            logger.warning("OCR statement line skipped: %s — %s", raw, exc)
            continue

    if not lines:
        raise ValueError("no transaction lines could be read from the statement.")

    return ExtractionResult(
        header=StatementHeader(
            bank=payload.get("bank") or None,
            account_number=payload.get("account_number") or None,
            period_start=payload.get("period_start") or None,
            period_end=payload.get("period_end") or None,
            opening_balance=opening,
            closing_balance=closing,
        ),
        lines=lines,
    )


def _extract_via_claude(
    pdf_bytes: bytes,
    client=None,
    opening_override: Optional[Decimal] = None,
    closing_override: Optional[Decimal] = None,
) -> ExtractionResult:
    client = client or get_claude_client()
    if not client:
        raise RuntimeError(
            "AI service unavailable — set ANTHROPIC_API_KEY in the server environment."
        )
    encoded = base64.standard_b64encode(pdf_bytes).decode("ascii")
    response = client.messages.create(
        model=OCR_MODEL,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": encoded,
                    },
                },
                {"type": "text", "text": _SA_BANK_STATEMENT_PROMPT},
            ],
        }],
    )
    text = response.content[0].text.strip()
    payload = _parse_claude_json(text)
    return _payload_to_result(payload, opening_override, closing_override)


def _result_to_review_rows(result: ExtractionResult) -> List[Dict[str, Any]]:
    """Convert ExtractionResult lines to the review-screen row dicts."""
    return [
        {
            "date": ln.date,
            "description": ln.description,
            "amount": float(ln.amount),
            "confidence": float(ln.confidence),
        }
        for ln in result.lines
    ]


def extract_bank_statement(
    pdf_bytes: bytes,
    *,
    opening_balance: Optional[str] = None,
    closing_balance: Optional[str] = None,
    client=None,
) -> BankStatementExtraction:
    """Run Tier-1 → Tier-2 extraction and integrity gate.

    ``opening_balance`` / ``closing_balance`` are optional form overrides (helpful
    when the parser cannot read header balances from a scan).
    """
    if not pdf_bytes:
        return BankStatementExtraction(
            error="The uploaded file is empty.",
            error_code="EMPTY_FILE",
        )
    if len(pdf_bytes) > MAX_PDF_BYTES:
        return BankStatementExtraction(
            error=f"PDF is too large (max {MAX_PDF_BYTES // (1024 * 1024)} MB).",
            error_code="FILE_TOO_LARGE",
        )

    opening_dec = _parse_optional_balance(opening_balance)
    closing_dec = _parse_optional_balance(closing_balance)

    result: Optional[ExtractionResult] = None
    method = ""

    # Tier 1: digital PDF text (fast, no API cost)
    try:
        result = extract_pdf_statement(pdf_bytes, opening_dec, closing_dec)
        method = "digital_pdf"
        logger.info(
            "Bank statement Tier-1 OK: %d lines, bank=%s",
            len(result.lines), result.header.bank,
        )
    except PdfStatementError as tier1_exc:
        logger.info("Tier-1 PDF parse skipped: %s — trying Claude Vision", tier1_exc)

    # Tier 2: Claude Vision (scans + complex layouts)
    if result is None:
        try:
            result = _extract_via_claude(
                pdf_bytes, client=client,
                opening_override=opening_dec,
                closing_override=closing_dec,
            )
            method = "claude_vision"
            logger.info(
                "Bank statement Tier-2 OK: %d lines, bank=%s",
                len(result.lines), result.header.bank,
            )
        except RuntimeError as exc:
            return BankStatementExtraction(
                error=str(exc),
                error_code="AI_UNAVAILABLE",
            )
        except Exception as exc:
            logger.exception("Bank statement Claude extraction failed")
            hint = ""
            if opening_dec is None or closing_dec is None:
                hint = (
                    " If this is a scanned statement, try entering the opening and "
                    "closing balances on the upload form."
                )
            return BankStatementExtraction(
                error=f"Could not read transactions from that PDF ({type(exc).__name__}).{hint}",
                error_code="EXTRACTION_FAILED",
            )

    report = self_audit(result)
    rows = _result_to_review_rows(result)

    return BankStatementExtraction(
        rows=rows,
        header=result.header,
        report_card=report,
        method=method,
    )
