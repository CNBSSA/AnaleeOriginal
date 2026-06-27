"""
Bank statement reader service.
Reads .xlsx and .csv files into a cleaned Date/Description/Amount DataFrame.
"""
import logging
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class BankStatementExcelReader:
    """Handles reading and validation of bank statement files (.xlsx / .csv)."""

    def __init__(self):
        self.required_columns = ['Date', 'Description', 'Amount']
        self.errors = []

    def read_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read a bank statement file, dispatching by extension (.csv or .xlsx)."""
        if str(file_path).lower().endswith('.csv'):
            return self.read_csv(file_path)
        return self.read_excel(file_path)

    def read_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read a .xlsx bank statement. Returns a cleaned DataFrame or None."""
        try:
            logger.info(f"Attempting to read Excel file: {file_path}")
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            error_msg = f"Error reading Excel file: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
        return self._process(df)

    def read_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read a .csv bank statement (UTF-8 with a latin-1 fallback)."""
        try:
            logger.info(f"Attempting to read CSV file: {file_path}")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            error_msg = f"Error reading CSV file: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
        return self._process(df)

    def _process(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate required columns and clean Date/Amount/Description.

        Shared by the Excel and CSV readers so both formats behave identically.
        Invalid dates/amounts drop their row; missing required columns fail.
        """
        try:
            # Log the columns found
            logger.info(f"Found columns: {df.columns.tolist()}")

            # Check for required columns (case-insensitive)
            df.columns = [col.strip() if isinstance(col, str) else str(col) for col in df.columns]
            missing_columns = []
            for required_col in self.required_columns:
                if not any(col.lower() == required_col.lower() for col in df.columns):
                    missing_columns.append(required_col)

            if missing_columns:
                error_msg = f"Missing required columns: {', '.join(missing_columns)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                return None

            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                for req_col in self.required_columns:
                    if col.lower() == req_col.lower():
                        column_mapping[col] = req_col
            df = df.rename(columns=column_mapping)

            # Convert date column to datetime
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                invalid_dates = df['Date'].isna()
                if invalid_dates.any():
                    logger.warning(f"Found {invalid_dates.sum()} rows with invalid dates")
                    df = df.dropna(subset=['Date'])
            except Exception as e:
                error_msg = f"Error converting dates: {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                return None

            # Convert amount to float and handle formatting
            try:
                # Remove any currency symbols and commas
                df['Amount'] = df['Amount'].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
                invalid_amounts = df['Amount'].isna()
                if invalid_amounts.any():
                    logger.warning(f"Found {invalid_amounts.sum()} rows with invalid amounts")
                    df = df.dropna(subset=['Amount'])
            except Exception as e:
                error_msg = f"Error converting amounts: {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                return None

            # Clean up description field
            df['Description'] = df['Description'].astype(str).str.strip()

            logger.info(f"Successfully processed {len(df)} valid rows")
            return df

        except Exception as e:
            error_msg = f"Error processing bank statement data: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None

    def get_errors(self) -> List[str]:
        """Return list of errors encountered during reading"""
        return self.errors

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the data in the DataFrame with enhanced checks
        Returns True if valid, False otherwise
        """
        try:
            # Check for empty dataframe
            if df.empty:
                self.errors.append("No data found in file")
                return False

            # Check for null values
            null_counts = df[self.required_columns].isnull().sum()
            if null_counts.any():
                for col, count in null_counts.items():
                    if count > 0:
                        self.errors.append(f"Found {count} empty values in {col}")
                return False

            # Validate date range
            date_range = df['Date'].agg(['min', 'max'])
            if (date_range['max'] - date_range['min']).days > 366:
                self.errors.append("Statement period exceeds 1 year")
                return False

            # Validate amount range (optional)
            amount_range = df['Amount'].abs().max()
            if amount_range > 1000000000:  # Billion
                self.errors.append(f"Suspicious transaction amount detected: {amount_range}")
                return False

            return True

        except Exception as e:
            self.errors.append(f"Error validating data: {str(e)}")
            return False
