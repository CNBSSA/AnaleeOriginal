"""
Excel reader service for bank statements
Handles CNBS / SA bank export formats with auto header detection
"""
import csv
import io
import logging
from typing import List, Optional
import pandas as pd

from .format_detector import find_header_row, normalize_bank_statement_dataframe

_CSV_DELIMITERS = (',', ';', '\t', '|')


def _decode(raw: bytes) -> str:
    """UTF-8 (with or without a byte-order mark), else Windows-1252."""
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        return raw.decode('cp1252', errors='replace')


def _rows_for(text: str, delimiter: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter,
                           skipinitialspace=True))
    width = max((len(r) for r in rows), default=0)
    return [r + [''] * (width - len(r)) for r in rows]


def read_csv_rows(file_path: str) -> pd.DataFrame:
    """Read a bank CSV into a raw, header-less, all-string DataFrame.

    Why not ``pd.read_csv``: SA bank exports are not tidy CSV. FNB puts account
    details above the transactions (lines of 2 fields, then rows of 4), which
    made pandas raise "Expected 2 fields in line 4, saw 4" and the whole file
    was refused; Afrikaans/European-locale exports use ``;``, which pandas read
    as one column. ``csv.reader`` tolerates ragged rows, and the delimiter is
    the one under which a real header row (Date + an amount column) is found.
    """
    with open(file_path, 'rb') as handle:
        text = _decode(handle.read())

    candidates = []
    for delimiter in _CSV_DELIMITERS:
        rows = _rows_for(text, delimiter)
        if find_header_row(rows) is not None:
            return pd.DataFrame(rows, dtype=str)
        width = rows and len(rows[0]) or 0
        candidates.append((width, delimiter, rows))
    # No recognisable header under any delimiter: hand back the widest reading
    # so the normaliser can name the headings it did find.
    _width, _delimiter, rows = max(candidates, key=lambda c: c[0])
    return pd.DataFrame(rows, dtype=str)

logger = logging.getLogger(__name__)


class BankStatementExcelReader:
    """Handles reading and validation of bank statement Excel/CSV files"""

    def __init__(self):
        self.required_columns = ['Date', 'Description', 'Amount']
        self.errors = []

    def read_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read bank statement file and return normalized Date/Description/Amount rows."""
        self.errors = []
        try:
            logger.info('Attempting to read bank statement file: %s', file_path)
            if file_path.lower().endswith('.csv'):
                raw_df = read_csv_rows(file_path)
            else:
                raw_df = pd.read_excel(file_path, engine='openpyxl', header=None, dtype=str)

            logger.info('Raw sheet shape: %s', raw_df.shape)
            df = normalize_bank_statement_dataframe(raw_df)
            logger.info('Successfully normalized %s transaction rows', len(df))
            return df

        except ValueError as exc:
            error_msg = str(exc)
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
        except Exception as exc:
            error_msg = f'Error reading bank statement file: {exc}'
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None

    def get_errors(self) -> List[str]:
        """Return list of errors encountered during reading"""
        return self.errors
