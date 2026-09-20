"""One way to read a date off a bank statement — used by every import path.

THE DEFECT THIS EXISTS TO END (2026-09-20). Five import paths each parsed dates
their own way and disagreed about what a date means:

    bank_statements/format_detector.py   pd.to_datetime(v, dayfirst=True)
    bank_statements/services.py          pd.to_datetime(v)
    bank_statements/upload_validator.py  pd.to_datetime(v)
    historical_data/upload_diagnostics.py pd.to_datetime(v)
    routes.py::process_transaction_rows  pd.to_datetime(v)

Neither setting is right on its own, and pandas 2.2.3 behaves as follows:

    input                  dayfirst=False (default)   dayfirst=True
    '03/04/2025'           2025-03-04  (4 March)      2025-04-03  (3 April)
    '13/04/2025'           2025-04-13                 2025-04-13
    '2025-04-03'           2025-04-03                 2025-03-04   <-- swapped
    '2025-04-03 00:00:00'  2025-04-03                 2025-03-04   <-- swapped

So:

* The DEFAULT is wrong for South Africa. A statement written DD/MM/YYYY has
  every date in the first twelve days of a month silently transposed — 03/04
  becomes 4 March instead of 3 April. Days 13-31 come out right, because month
  13 is impossible and pandas falls back, so a single file ends up internally
  inconsistent rather than uniformly wrong.
* ``dayfirst=True`` is wrong for an ISO date. pandas reads '2025-04-03' as
  YYYY-DD-MM and returns 4 March. This is not hypothetical here: the main
  bank-statement path reads every cell with ``dtype=str`` (excel_reader), so a
  real Excel date cell arrives as '2025-04-03 00:00:00' and was being
  transposed by the very ``dayfirst=True`` that protects the DD/MM files.

The fix is not to pick one flag. It is to look at the value first: a date
written year-first is already unambiguous and must be parsed as written;
anything else is day-first, because that is how South Africa writes dates.

Nothing here reaches the frozen analysis engine — this is ingress, which the
capability freeze leaves as working surface ("how data GETS TO the engine").

NOTE ON ALREADY-IMPORTED DATA. This corrects what happens from now on. It
cannot repair rows already stored: the uploaded file is written to /tmp and
deleted in a finally block, and ``UploadedFile`` keeps only the filename, so
there is nothing left to re-read. An affected statement has to be re-uploaded.
``services/date_swap_audit.py`` finds the uploads that carry the fingerprint.
"""
from __future__ import annotations

import re
from datetime import date as _date, datetime as _datetime
from typing import Any, Optional

import pandas as pd

# A leading four-digit year: 2025-04-03, 2025/4/3, 2025.04.03 (optionally with
# a time after it). Written this way the date is already unambiguous, so it
# must NOT be read day-first.
_YEAR_FIRST = re.compile(r'^\s*\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}')


def is_year_first(text: str) -> bool:
    """True when the text starts with an unambiguous year-first date."""
    return bool(_YEAR_FIRST.match(text or ''))


def parse_statement_date(value: Any) -> Optional[pd.Timestamp]:
    """Read a bank-statement date, or return None if it cannot be read.

    * a real date/datetime/Timestamp is taken as it stands — there is nothing
      to interpret, and re-parsing one through a string is how day and month
      get swapped;
    * a year-first string (ISO) is parsed as written;
    * everything else is parsed day-first, the South African convention.

    Returns a ``pd.Timestamp`` so callers can take ``.date()`` or
    ``.to_pydatetime()``. Never raises.
    """
    if value is None:
        return None
    # datetime is a subclass of date, so this covers both, plus pd.Timestamp.
    if isinstance(value, (pd.Timestamp, _datetime, _date)):
        return pd.Timestamp(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return None

    parsed = pd.to_datetime(text, errors='coerce', dayfirst=not is_year_first(text))
    return None if pd.isna(parsed) else parsed


def parse_statement_date_or_raise(value: Any, *, field: str = 'Date') -> pd.Timestamp:
    """``parse_statement_date`` for callers that treated a bad date as fatal.

    Keeps the previous behaviour of the paths that let pandas raise, but with a
    message that names the value instead of surfacing a parser internal.
    """
    parsed = parse_statement_date(value)
    if parsed is None:
        raise ValueError(
            f'{field} could not be read as a date: {value!r}. Dates may be '
            f'written 03/04/2025 (day first), 2025-04-03, or 3 Apr 2025.')
    return parsed
