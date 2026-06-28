"""
Bank statement validation module
Handles specific validation rules for bank statements
"""
import logging
import pandas as pd
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Tuple
import os
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

class BankStatementValidator:
    """Validates and processes bank statement uploads"""

    REQUIRED_COLUMNS = ['Date', 'Description', 'Amount']
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx'}
    REJECTED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png'}

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.processed_rows = 0
        self.total_rows = 0

    def validate_and_process(self, file, account_id: int, user_id: int) -> bool:
        """
        Validate and process the uploaded bank statement
        Returns True if successful, False otherwise
        """
        try:
            # Validate file extension
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in self.REJECTED_EXTENSIONS:
                self.errors.append(
                    'Analee accepts bank statements only (CSV or Excel). '
                    'PDF and other document uploads are not supported.'
                )
                return False
            if file_ext not in self.ALLOWED_EXTENSIONS:
                self.errors.append(f"Invalid file format. Allowed formats: {', '.join(self.ALLOWED_EXTENSIONS)}")
                return False

            # Read file with logging
            try:
                logger.info(f"Attempting to read file: {filename}")
                if file_ext == '.xlsx':
                    df = pd.read_excel(file, engine='openpyxl')
                else:
                    try:
                        df = pd.read_csv(file, encoding='utf-8')
                    except UnicodeDecodeError:
                        file.seek(0)
                        df = pd.read_csv(file, encoding='latin1')
                logger.info(f"Successfully read file with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                self.errors.append(f"Error reading file: {str(e)}")
                return False

            # Validate structure
            if not self._validate_structure(df):
                logger.error("Structure validation failed")
                return False

            # Process rows with enhanced logging
            self.total_rows = len(df)
            success = self._process_rows(df, account_id, user_id)
            if success:
                logger.info(f"Successfully processed {self.processed_rows} out of {self.total_rows} rows")
            else:
                logger.error(f"Processing failed. Processed {self.processed_rows} out of {self.total_rows} rows")

            return success

        except Exception as e:
            logger.error(f"Error processing bank statement: {str(e)}")
            self.errors.append(f"Error processing file: {str(e)}")
            return False

    def _validate_structure(self, df: pd.DataFrame) -> bool:
        """Validate the basic structure of the uploaded file"""
        try:
            if df.empty:
                self.errors.append("The uploaded file is empty")
                return False

            # Check for required columns (case-insensitive)
            df.columns = [col.strip() for col in df.columns]
            missing_columns = []
            for required_col in self.REQUIRED_COLUMNS:
                if not any(col.lower() == required_col.lower() for col in df.columns):
                    missing_columns.append(required_col)

            if missing_columns:
                self.errors.append(f"Missing required columns: {', '.join(missing_columns)}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file structure: {str(e)}")
            self.errors.append(f"Error validating file structure: {str(e)}")
            return False

    def _process_rows(self, df: pd.DataFrame, account_id: int, user_id: int) -> bool:
        """Process each row of the bank statement"""
        try:
            valid_rows = []
            for idx, row in df.iterrows():
                self.processed_rows += 1
                logger.debug(f"Processing row {idx + 1}")

                # Validate row data
                cleaned_data = self._validate_row(row, idx + 2)
                if cleaned_data:
                    valid_rows.append({
                        'date': cleaned_data['date'],
                        'description': cleaned_data['description'],
                        'amount': cleaned_data['amount'],
                        'account_id': account_id,
                        'user_id': user_id
                    })

            # If we have valid rows, consider it a success
            if valid_rows:
                logger.info(f"Found {len(valid_rows)} valid rows")
                return True
            else:
                logger.warning("No valid rows found")
                return False

        except Exception as e:
            logger.error(f"Error processing rows: {str(e)}")
            self.errors.append(f"Error processing rows: {str(e)}")
            return False

    def _validate_row(self, row: pd.Series, row_num: int) -> Dict:
        """
        Validate a single row of data
        Returns cleaned data dictionary if valid, None if invalid
        """
        try:
            cleaned_data = {}

            # Date validation with enhanced error handling
            try:
                if pd.isna(row['Date']):
                    logger.warning(f"Row {row_num}: Missing date")
                    self.errors.append(f"Row {row_num}: Missing date")
                    return None

                date_value = pd.to_datetime(row['Date'])
                if date_value > datetime.now():
                    logger.warning(f"Row {row_num}: Future date detected")
                    self.errors.append(f"Row {row_num}: Date cannot be in the future")
                    return None
                cleaned_data['date'] = date_value.date()
            except Exception as e:
                logger.error(f"Row {row_num}: Invalid date format: {str(e)}")
                self.errors.append(f"Row {row_num}: Invalid date format: {str(e)}")
                return None

            # Amount validation with enhanced error handling
            try:
                if pd.isna(row['Amount']):
                    logger.warning(f"Row {row_num}: Missing amount")
                    self.errors.append(f"Row {row_num}: Missing amount")
                    return None

                # Clean amount string by removing currency symbols and commas
                amount_str = str(row['Amount']).replace('$', '').replace(',', '').strip()
                amount = Decimal(amount_str)

                if amount == 0:
                    logger.warning(f"Row {row_num}: Zero amount detected")
                    self.warnings.append(f"Row {row_num}: Zero amount transaction")
                cleaned_data['amount'] = amount
            except (InvalidOperation, ValueError) as e:
                logger.error(f"Row {row_num}: Invalid amount format: {str(e)}")
                self.errors.append(f"Row {row_num}: Invalid amount format: {str(e)}")
                return None

            # Description validation
            if pd.isna(row['Description']) or str(row['Description']).strip() == '':
                logger.warning(f"Row {row_num}: Missing description")
                self.errors.append(f"Row {row_num}: Missing description")
                return None
            cleaned_data['description'] = str(row['Description']).strip()[:200]

            return cleaned_data

        except Exception as e:
            logger.error(f"Error validating row {row_num}: {str(e)}")
            self.errors.append(f"Error validating row {row_num}: {str(e)}")
            return None

    def get_error_messages(self) -> List[str]:
        """Get list of error messages"""
        return self.errors

    def get_warning_messages(self) -> List[str]:
        """Get list of warning messages"""
        return self.warnings

    def get_progress(self) -> Dict:
        """Get processing progress information"""
        return {
            'processed_rows': self.processed_rows,
            'total_rows': self.total_rows,
            'success': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings
        }