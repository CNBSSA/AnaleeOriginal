"""Bank statement upload validation."""
import pytest

from bank_statements.upload_validator import BankStatementValidator


def test_rejects_pdf_upload():
    validator = BankStatementValidator()

    class FakeFile:
        filename = 'statement-scan.pdf'

    assert validator.validate_and_process(FakeFile(), account_id=1, user_id=1) is False
    assert any('PDF' in err for err in validator.errors)
