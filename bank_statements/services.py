"""
Service layer for bank statement processing
Handles business logic separately from routes
Enhanced with user-friendly error notifications
"""
import logging
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
from werkzeug.utils import secure_filename
from sqlalchemy.exc import SQLAlchemyError
from .models import BankStatementUpload, Transaction, UploadedFile
from .excel_reader import BankStatementExcelReader
from models import db
import pandas as pd

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
            'processing_error': "We encountered an issue while processing your file. Please try again.",
            'db_error': "There was a problem saving your data. Please try again.",
            'unknown': "An unexpected error occurred. Please try again or contact support.",
            'file_save_error': "Failed to save the uploaded file. Please try again."
        }
        base_message = error_messages.get(error_type, error_messages['unknown'])
        if details:
            return f"{base_message} Details: {details}"
        return base_message

    def create_uploaded_file_record(
        self,
        filename: str,
        user_id: int
    ) -> UploadedFile:
        """
        Create a record for the uploaded file
        """
        uploaded_file = UploadedFile(
            filename=secure_filename(filename),
            user_id=user_id
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
                txn_date = pd.to_datetime(row['Date']).to_pydatetime()
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
        user_id: int
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

            # Save file temporarily
            try:
                temp_path = os.path.join('/tmp', secure_filename(file.filename))
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

                if df is None or df.empty:
                    details = '; '.join(self.excel_reader.get_errors()) or 'No readable rows'
                    error_msg = self.get_friendly_error_message('empty_file', details)
                    upload.set_error(error_msg)
                    db.session.commit()
                    return False, {
                        'success': False,
                        'error': error_msg,
                        'error_type': 'empty_file',
                        'details': self.excel_reader.get_errors(),
                    }

                uploaded_file = self.create_uploaded_file_record(file.filename, user_id)

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
                    'file_id': uploaded_file.id
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
                'details': [str(e)]
            }
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Cleaned up temporary file")