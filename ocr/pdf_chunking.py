"""PDF page counting and chunking for multi-page bank statements."""
from __future__ import annotations

import io
import logging
from typing import List

logger = logging.getLogger(__name__)

# Chunk when a statement exceeds this many pages (Claude output token limit).
CHUNK_PAGE_THRESHOLD = 4
PAGES_PER_CHUNK = 4


def _pdf_reader(pdf_bytes: bytes):
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        from PyPDF2 import PdfReader  # type: ignore
    return PdfReader(io.BytesIO(pdf_bytes))


def count_pdf_pages(pdf_bytes: bytes) -> int:
    try:
        return len(_pdf_reader(pdf_bytes).pages)
    except Exception as exc:
        logger.warning("Could not count PDF pages: %s", exc)
        return 1


def split_pdf_bytes(pdf_bytes: bytes, pages_per_chunk: int = PAGES_PER_CHUNK) -> List[bytes]:
    """Split a PDF into smaller PDF byte strings (each valid for Claude document block)."""
    reader = _pdf_reader(pdf_bytes)
    total = len(reader.pages)
    if total <= pages_per_chunk:
        return [pdf_bytes]
    chunks: List[bytes] = []
    try:
        from pypdf import PdfWriter
    except ImportError:  # pragma: no cover
        from PyPDF2 import PdfWriter  # type: ignore
    for start in range(0, total, pages_per_chunk):
        writer = PdfWriter()
        end = min(start + pages_per_chunk, total)
        for idx in range(start, end):
            writer.add_page(reader.pages[idx])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    logger.info("Split %d-page PDF into %d chunk(s) of up to %d pages", total, len(chunks), pages_per_chunk)
    return chunks


def needs_chunking(pdf_bytes: bytes, threshold: int = CHUNK_PAGE_THRESHOLD) -> bool:
    return count_pdf_pages(pdf_bytes) > threshold
