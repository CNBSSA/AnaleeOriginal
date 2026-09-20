"""
Bank statement upload diagnostics system
Provides comprehensive validation and error reporting for bank statement uploads
"""
import logging
import pandas as pd
from utils.date_parsing import parse_statement_date
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Tuple, Any
from datetime import datetime
from flask import request

logger = logging.getLogger(__name__)

# Configure file handler for detailed logging
file_handler = logging.FileHandler('upload_diagnostics.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class UploadDiagnostics:
    """Handles validation and diagnostics for bank statement uploads"""

    REQUIRED_COLUMNS = ['Date', 'Description', 'Amount']

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'processed_rows': 0
        }
        self.request_info = {}

    def capture_request_info(self):
        """Capture detailed information about the current request"""
        try:
            self.request_info = {
                'method': request.method,
                'content_type': request.content_type,
                'content_length': request.content_length,
                'files_present': bool(request.files),
                'form_data_present': bool(request.form),
                'headers': dict(request.headers)
            }

            # Log request details
            logger.debug("Request Details:")
            for key, value in self.request_info.items():
                logger.debug(f"{key}: {value}")

            if request.files:
                for file_key in request.files:
                    file = request.files[file_key]
                    logger.debug(f"File: {file_key}")
                    logger.debug(f"Filename: {file.filename}")
                    logger.debug(f"Content Type: {file.content_type}")

        except Exception as e:
            logger.error(f"Error capturing request info: {str(e)}")
            self.errors.append({
                'type': 'request_validation',
                'message': f'Error analyzing request: {str(e)}',
                'severity': 'error'
            })

    def validate_file_structure(self, df: pd.DataFrame) -> bool:
        """
        Validate the basic structure of the uploaded file
        Returns True if structure is valid, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                self.errors.append({
                    'type': 'file_structure',
                    'message': 'The uploaded file is empty',
                    'severity': 'error'
                })
                return False

            # Update total rows stat
            self.stats['total_rows'] = len(df)

            # Check for required columns
            missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                self.errors.append({
                    'type': 'missing_columns',
                    'message': f"Missing required columns: {', '.join(missing_columns)}",
                    'details': missing_columns,
                    'severity': 'error'
                })
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file structure: {str(e)}")
            self.errors.append({
                'type': 'validation_error',
                'message': f'Error validating file structure: {str(e)}',
                'severity': 'error'
            })
            return False

    def validate_row(self, row: pd.Series, row_num: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a single row of data
        Returns (is_valid, cleaned_data)
        """
        row_errors = []
        cleaned_data = {}

        try:
            # Date validation
            try:
                if pd.isna(row['Date']):
                    row_errors.append('Missing date')
                else:
                    date_value = parse_statement_date(row['Date'])
                    if date_value is None:
                        raise ValueError(f"unreadable date {row['Date']!r}")
                    if date_value > pd.Timestamp(datetime.now()):
                        row_errors.append('Date cannot be in the future')
                    cleaned_data['date'] = date_value.date()
            except Exception as e:
                row_errors.append(f'Invalid date format: {str(e)}')

            # Amount validation
            try:
                if pd.isna(row['Amount']):
                    row_errors.append('Missing amount')
                else:
                    amount = Decimal(str(row['Amount']))
                    if amount == 0:
                        self.warnings.append({
                            'row': row_num,
                            'message': 'Zero amount transaction'
                        })
                    cleaned_data['amount'] = amount
            except (InvalidOperation, ValueError) as e:
                row_errors.append(f'Invalid amount format: {str(e)}')

            # Description validation
            if pd.isna(row['Description']) or str(row['Description']).strip() == '':
                row_errors.append('Missing description')
            else:
                cleaned_data['description'] = str(row['Description']).strip()[:200]

            # Update statistics
            if row_errors:
                self.stats['invalid_rows'] += 1
                self.errors.append({
                    'type': 'row_validation',
                    'row': row_num,
                    'messages': row_errors,
                    'severity': 'error'
                })
                return False, {}
            else:
                self.stats['valid_rows'] += 1
                return True, cleaned_data

        except Exception as e:
            logger.error(f"Error validating row {row_num}: {str(e)}")
            self.errors.append({
                'type': 'row_validation',
                'row': row_num,
                'messages': [f'Unexpected error: {str(e)}'],
                'severity': 'error'
            })
            return False, {}

    def get_diagnostic_summary(self) -> Dict:
        """
        Get a summary of the validation results including request info
        """
        return {
            'stats': self.stats,
            'errors': self.errors,
            'warnings': self.warnings,
            'has_errors': len(self.errors) > 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'request_info': self.request_info
        }

    def get_user_friendly_messages(self) -> List[Dict]:
        """
        Get user-friendly error messages
        """
        messages = []

        # Request validation errors
        request_errors = [e for e in self.errors if e['type'] == 'request_validation']
        if request_errors:
            messages.append({
                'type': 'error',
                'message': 'Error processing upload request. Please try again.'
            })

        # File structure errors
        structure_errors = [e for e in self.errors if e['type'] in ('file_structure', 'missing_columns')]
        if structure_errors:
            for error in structure_errors:
                messages.append({
                    'type': 'error',
                    'message': error['message']
                })

        # Row validation errors (show first 5)
        row_errors = [e for e in self.errors if e['type'] == 'row_validation'][:5]
        if row_errors:
            for error in row_errors:
                messages.append({
                    'type': 'error',
                    'message': f"Row {error['row']}: {'; '.join(error['messages'])}"
                })
            if len(row_errors) < len([e for e in self.errors if e['type'] == 'row_validation']):
                messages.append({
                    'type': 'info',
                    'message': 'Additional errors found. Please check the detailed log.'
                })

        # Add warnings if present
        if self.warnings:
            messages.append({
                'type': 'warning',
                'message': f"{len(self.warnings)} warning(s) found. Check detailed log for more information."
            })

        # Add success message if no errors
        if not self.errors:
            messages.append({
                'type': 'success',
                'message': f"Successfully validated {self.stats['valid_rows']} rows of data."
            })

        return messages