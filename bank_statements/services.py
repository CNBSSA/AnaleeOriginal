"""
Service layer for bank statement processing
Handles business logic separately from routes
Enhanced with user-friendly error notifications
"""
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional
from werkzeug.utils import secure_filename
from sqlalchemy.exc import SQLAlchemyError
from .models import BankStatementUpload, Transaction, UploadedFile
from .excel_reader import BankStatementExcelReader
from models import db
import pandas as pd
from utils.date_parsing import parse_statement_date_or_raise
from ocr.service import mark_duplicates

logger = logging.getLogger(__name__)

class BankStatementService:
    """Service for handling bank statement uploads and processing"""

    def __init__(self):
        self.excel_reader = BankStatementExcelReader()
        self.errors = []

    def get_friendly_error_message(self, error_type: str, details: str = None) -> str:
        """
        Convert technical errors into user-friendly messages
        """
        error_messages = {
            'file_type': "Please upload only Excel (.xlsx) or CSV (.csv) files.",
            'missing_columns': "Your file is missing required columns. Please ensure it includes: Date, Description, and Amount.",
            'invalid_date': "Some dates in your statement are not in the correct format. Please check the date format.",
            'invalid_amount': "Some amounts are not in the correct format. Please ensure amounts are numbers.",
            'future_date': "We noticed some future dates in your statement. Please check the dates.",
            'empty_file': "The uploaded file appears to be empty. Please check the file contents.",
            'unreadable': "We could not read the transactions in this file.",
            'processing_error': "We encountered an issue while processing your file. Please try again.",
            'db_error': "There was a problem saving your data. Please try again.",
            'unknown': "An unexpected error occurred. Please try again or contact support.",
            'file_save_error': "Failed to save the uploaded file. Please try again."
        }
        base_message = error_messages.get(error_type, error_messages['unknown'])
        # Only a problem with the FILE (our own validation) is worth showing the
        # user; database and unexpected errors carry internal text, so they are
        # logged and the plain sentence is shown (2026-09-30).
        if details and error_type == 'processing_error':
            return f"{base_message} Details: {details}"
        return base_message

    def create_uploaded_file_record(
        self,
        filename: str,
        user_id: int,
        bank_account_id: Optional[int] = None,
    ) -> UploadedFile:
        """
        Create a record for the uploaded file
        """
        uploaded_file = UploadedFile(
            filename=secure_filename(filename),
            user_id=user_id,
            bank_account_id=bank_account_id,
        )
        db.session.add(uploaded_file)
        db.session.commit()
        return uploaded_file

    def create_transactions(
        self,
        df: pd.DataFrame,
        account_id: int,
        user_id: int,
        file_id: int
    ) -> List[Transaction]:
        """
        Create transaction records from the processed DataFrame
        """
        transactions = []
        for _, row in df.iterrows():
            try:
                txn_date = parse_statement_date_or_raise(row['Date']).to_pydatetime()
                transaction = Transaction(
                    date=txn_date,
                    description=str(row['Description'])[:200],
                    amount=float(row['Amount']),
                    user_id=user_id,
                    account_id=account_id,
                    file_id=file_id,
                    category=row.get('Category', None)
                )
                transactions.append(transaction)
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing transaction row: {str(e)}", exc_info=True)
                raise ValueError(f"Error processing transaction: {str(e)}")

        return transactions

    def save_transactions(self, transactions: List[Transaction]) -> bool:
        """
        Save transactions to database with error handling
        """
        try:
            db.session.add_all(transactions)
            db.session.commit()
            return True
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Database error while saving transactions: {str(e)}", exc_info=True)
            raise

    def process_upload(
        self,
        file,
        account_id: int,
        user_id: int,
        *,
        confirm_duplicates: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a bank statement upload with enhanced error handling and validation
        Returns (success, response_data)
        """
        temp_path = None
        try:
            logger.info(f"Starting to process upload for user {user_id}, account {account_id}")

            # Create upload record
            upload = BankStatementUpload(
                filename=secure_filename(file.filename),
                account_id=account_id,
                user_id=user_id,
                status='processing'
            )
            db.session.add(upload)
            db.session.commit()

            # Validate file extension
            file_ext = os.path.splitext(secure_filename(file.filename))[1].lower()
            if file_ext not in ['.csv', '.xlsx']:
                error_msg = self.get_friendly_error_message('file_type')
                upload.set_error(error_msg)
                db.session.commit()
                return False, {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'file_type'
                }

            # Save file temporarily — under a UNIQUE name (2026-09-30). This used
            # to be /tmp/<the uploaded file's name>, so two users uploading
            # "statement.xlsx" at the same moment wrote the same path: one
            # user's rows could be read into the other user's account, and the
            # first request's cleanup deleted the second's file mid-read.
            # mkstemp creates the file atomically with a random name (mode
            # 0600); the extension is kept because the reader picks CSV vs
            # Excel by it.
            try:
                fd, temp_path = tempfile.mkstemp(prefix='analee-upload-', suffix=file_ext)
                os.close(fd)
                file.save(temp_path)
            except Exception as e:
                error_msg = self.get_friendly_error_message('file_save_error', str(e))
                upload.set_error(error_msg)
                db.session.commit()
                return False, {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'file_save_error'
                }

            # Process file and create transactions
            try:
                df = self.excel_reader.read_excel(temp_path)
                rows_skipped = len(self.excel_reader.last_rows_skipped)
                processing_notes = list(self.excel_reader.last_processing_notes)

                if df is None or df.empty:
                    reader_errors = self.excel_reader.get_errors()
                    # A file whose layout could not be read is not "empty":
                    # say what was wrong, in the reader's own words.
                    error_type = 'unreadable' if reader_errors else 'empty_file'
                    details = '; '.join(reader_errors) or 'No readable rows'
                    error_msg = self.get_friendly_error_message(error_type, details)
                    upload.set_error(error_msg)
                    db.session.commit()
                    return False, {
                        'success': False,
                        'error': error_msg,
                        'error_type': error_type,
                        'details': reader_errors,
                    }

                candidate_rows = [
                    {
                        'date': row['Date'].strftime('%Y-%m-%d')
                        if hasattr(row['Date'], 'strftime')
                        else str(row['Date']),
                        'description': str(row['Description']),
                        'amount': float(row['Amount']),
                    }
                    for _, row in df.iterrows()
                ]
                dup_flags = self._flag_spreadsheet_duplicates(
                    candidate_rows, user_id, account_id)
                duplicate_count = sum(1 for r in dup_flags if r.get('duplicate'))

                if duplicate_count and not confirm_duplicates:
                    error_msg = (
                        f'{duplicate_count} row(s) match existing transactions '
                        '(same date, amount and description). Submit again with '
                        '"Import anyway" to add only the non-duplicate rows, '
                        'or cancel to leave your books unchanged.'
                    )
                    upload.set_error(error_msg)
                    db.session.commit()
                    return False, {
                        'success': False,
                        'duplicate_warning': True,
                        'requires_confirmation': True,
                        'duplicate_count': duplicate_count,
                        'error': error_msg,
                        'rows_skipped': rows_skipped,
                        'processing_notes': processing_notes,
                    }

                if duplicate_count and confirm_duplicates:
                    keep_indexes = {
                        i for i, row in enumerate(dup_flags) if not row.get('duplicate')
                    }
                    df = df.iloc[sorted(keep_indexes)].reset_index(drop=True)
                    processing_notes.append(
                        f'Skipped {duplicate_count} duplicate row(s) on confirmed import.'
                    )

                uploaded_file = self.create_uploaded_file_record(
                    file.filename, user_id, bank_account_id=account_id)

                transactions = self.create_transactions(
                    df, account_id, user_id, uploaded_file.id
                )
                if not transactions:
                    db.session.delete(uploaded_file)
                    db.session.commit()
                    error_msg = self.get_friendly_error_message(
                        'empty_file', 'No valid transactions after parsing'
                    )
                    upload.set_error(error_msg)
                    db.session.commit()
                    return False, {
                        'success': False,
                        'error': error_msg,
                        'error_type': 'empty_file',
                    }

                self.save_transactions(transactions)

                # Update upload status
                upload.set_success(
                    f"Successfully processed {len(transactions)} transactions"
                )
                db.session.commit()

                return True, {
                    'success': True,
                    'message': 'File processed successfully',
                    'transactions_processed': len(transactions),
                    'upload_id': upload.id,
                    'file_id': uploaded_file.id,
                    'rows_skipped': rows_skipped,
                    'processing_notes': processing_notes,
                }

            except ValueError as e:
                error_msg = self.get_friendly_error_message('processing_error', str(e))
                upload.set_error(error_msg)
                db.session.commit()
                return False, {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'processing_error'
                }
            except SQLAlchemyError as e:
                error_msg = self.get_friendly_error_message('db_error', str(e))
                upload.set_error(error_msg)
                db.session.commit()
                return False, {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'db_error'
                }

        except Exception as e:
            logger.error(f"Error processing bank statement: {str(e)}", exc_info=True)
            error_msg = self.get_friendly_error_message('unknown', str(e))
            if 'upload' in locals():
                upload.set_error(error_msg)
                db.session.commit()
            return False, {
                'success': False,
                'error': error_msg,
                'error_type': 'unknown',
            }
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Cleaned up temporary file")

    def _flag_spreadsheet_duplicates(
        self,
        rows: List[Dict[str, Any]],
        user_id: int,
        account_id: int,
    ) -> List[Dict[str, Any]]:
        """Flag rows that match existing transactions (same semantics as PDF OCR)."""
        if not rows:
            return rows
        try:
            date_strings = {r['date'] for r in rows if r.get('date')}
            if not date_strings:
                for r in rows:
                    r['duplicate'] = False
                return rows
            date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in date_strings]
            existing_q = Transaction.query.filter(
                Transaction.user_id == user_id,
                Transaction.date >= min(date_objs),
                Transaction.date < max(date_objs) + timedelta(days=1),
            ).all()
            existing = [
                (t.date.strftime('%Y-%m-%d'), t.amount, (t.description or ''))
                for t in existing_q
            ]
            return mark_duplicates(rows, existing)
        except Exception as exc:
            logger.error('Spreadsheet duplicate check skipped: %s', exc)
            for r in rows:
                r.setdefault('duplicate', False)
            return rows