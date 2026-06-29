"""Unit tests for PDF chunking helpers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.pdf_chunking import count_pdf_pages, needs_chunking, split_pdf_bytes  # noqa: E402
from ocr.statement_extractor import _merge_chunk_payloads  # noqa: E402


def test_needs_chunking_single_page():
    # Minimal valid-ish PDF header bytes won't parse as multi-page; returns 1 page
    assert needs_chunking(b"%PDF-1.4\n", threshold=4) is False


def test_merge_chunk_payloads_dedupes():
    p1 = {
        "opening_balance": "1000.00",
        "lines": [{"date": "2026-03-01", "description": "A", "amount": "-10"}],
    }
    p2 = {
        "closing_balance": "990.00",
        "lines": [
            {"date": "2026-03-01", "description": "A", "amount": "-10"},
            {"date": "2026-03-02", "description": "B", "amount": "5"},
        ],
    }
    merged = _merge_chunk_payloads([p1, p2])
    assert merged["opening_balance"] == "1000.00"
    assert merged["closing_balance"] == "990.00"
    assert len(merged["lines"]) == 2
