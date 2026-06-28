"""
Excel reader service for bank statements
Handles CNBS / SA bank export formats with auto header detection
"""
import logging
from typing import List, Optional
import pandas as pd

from .format_detector import normalize_bank_statement_dataframe

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
                try:
                    raw_df = pd.read_csv(file_path, header=None, dtype=str, keep_default_na=False)
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(
                        file_path, header=None, dtype=str, keep_default_na=False, encoding='latin1'
                    )
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
