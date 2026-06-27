"""
The bank-statement reader handles BOTH .xlsx and .csv identically.

Regression for the gap where the Bank Statements / Upload Data pages always
called read_excel, so a .csv silently failed. read_file() now dispatches by
extension and both formats go through the same Date/Description/Amount cleaning.
"""
import io

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from bank_statements.excel_reader import BankStatementExcelReader

ROWS = [
    ("2026-01-02", "Deposit", "10000.00"),
    ("2026-01-08", "Withdrawal", "-200.00"),
    ("2026-01-12", "Formatted", "$1,234.56"),   # $ and comma must be cleaned
    ("not-a-date", "Bad date", "50.00"),          # dropped
    ("2026-01-15", "Bad amount", "abc"),          # dropped
]
def _csv(tmp_path):
    p = tmp_path / "stmt.csv"
    # pandas quotes fields containing commas (e.g. "$1,234.56") correctly.
    pd.DataFrame(ROWS, columns=["Date", "Description", "Amount"]).to_csv(p, index=False)
    return str(p)


def _xlsx(tmp_path):
    p = tmp_path / "stmt.xlsx"
    pd.DataFrame(ROWS, columns=["Date", "Description", "Amount"]).to_excel(p, index=False)
    return str(p)


def test_read_file_csv_parses_and_cleans(tmp_path):
    df = BankStatementExcelReader().read_file(_csv(tmp_path))
    assert df is not None
    # 2 invalid rows dropped -> 3 valid
    assert len(df) == 3
    assert str(df["Amount"].dtype).startswith("float")
    assert 1234.56 in set(df["Amount"].tolist())  # $1,234.56 cleaned


def test_read_file_xlsx_matches_csv(tmp_path):
    df = BankStatementExcelReader().read_file(_xlsx(tmp_path))
    assert df is not None and len(df) == 3


def test_csv_and_xlsx_produce_same_result(tmp_path):
    c = BankStatementExcelReader().read_file(_csv(tmp_path))
    x = BankStatementExcelReader().read_file(_xlsx(tmp_path))
    assert sorted(c["Amount"].tolist()) == sorted(x["Amount"].tolist())


def test_missing_required_column_fails(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Date,Memo,Value\n2026-01-01,x,10\n")
    assert BankStatementExcelReader().read_file(str(p)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
