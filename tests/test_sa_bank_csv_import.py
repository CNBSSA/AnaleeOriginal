"""Real South African bank CSV / Excel exports must import (QA 2026-09-28).

Live QA: "CSV/Excel bank-statement upload rejects valid files (only PDF works)".
Reproduced against the reader with the layouts SA banks actually export:

* FNB puts account details above the transactions, so the first lines have 2
  fields and the transactions 4 -> pandas raised "Expected 2 fields in line 4,
  saw 4" and the whole file was refused;
* a semicolon-delimited export (decimal comma) was read as one column;
* Capitec's "Money In" / "Money Out" / "Fee" headings were not recognised ->
  "Could not find required columns";
* an amount in brackets "(1,150.00)" was silently dropped;
* "150,00" (decimal comma) was read as 15 000.

Ingress only: nothing here touches the frozen analysis engine.
"""
import io
import os
import tempfile
from datetime import datetime

import pytest

from bank_statements.excel_reader import BankStatementExcelReader
from bank_statements.format_detector import _parse_number, _signed_amount


def _read(text, suffix='.csv', encoding='utf-8'):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, 'wb') as handle:
        handle.write(text.encode(encoding) if isinstance(text, str) else text)
    try:
        reader = BankStatementExcelReader()
        df = reader.read_excel(path)
        return df, reader.get_errors()
    finally:
        os.remove(path)


def _amounts(df):
    return [round(float(a), 2) for a in df['Amount']]


def test_fnb_csv_with_account_preamble():
    df, errors = _read(
        'ACC-NO,62000000000\n'
        'Opening Balance,1000.00\n'
        '\n'
        'Date, Amount, Balance, Description\n'
        '03/04/2025, -150.00, 850.00, POS PURCHASE WOOLWORTHS\n'
        '04/04/2025, 15000.00, 15850.00, SALARY ACME\n'
    )
    assert df is not None, errors
    assert _amounts(df) == [-150.0, 15000.0]
    # Day-first: 03/04/2025 is 3 April.
    assert df.iloc[0]['Date'].month == 4 and df.iloc[0]['Date'].day == 3


def test_semicolon_delimited_with_decimal_comma():
    df, errors = _read(
        'Datum;Description;Amount;Balance\n'.replace('Datum', 'Date')
        + '03/04/2025;POS PURCHASE;-150,00;850,00\n'
        '04/04/2025;SALARY;15000,00;15850,00\n'
    )
    assert df is not None, errors
    assert _amounts(df) == [-150.0, 15000.0]


def test_capitec_money_in_money_out_and_fee():
    df, errors = _read(
        'Nr,Account,Posting Date,Transaction Date,Description,Original Description,'
        'Parent Category,Category,Money In,Money Out,Fee,Balance\n'
        '1,123,2025/04/03,2025/04/03,Woolies,WOOLWORTHS,Food,Groceries,,-150.00,,850.00\n'
        '2,123,2025/04/04,2025/04/04,Salary,ACME,Income,Salary,15000.00,,,15850.00\n'
        '3,123,2025/04/05,2025/04/05,Monthly fee,FEE,Fees,Fees,,,-7.50,15842.50\n'
    )
    assert df is not None, errors
    assert _amounts(df) == [-150.0, 15000.0, -7.5]


def test_bom_and_windows_1252():
    df, errors = _read('﻿Date,Description,Amount\n2025-04-03,CAFÉ,-12.00\n')
    assert df is not None, errors
    assert _amounts(df) == [-12.0]
    df, errors = _read('Date,Description,Amount\n2025-04-03,CAFÉ,-12.00\n',
                       encoding='cp1252')
    assert df is not None, errors
    assert 'CAF' in df.iloc[0]['Description']


def test_brackets_rand_spaces_and_separate_debit_credit():
    df, errors = _read(
        'Transaction Date,Transaction Description,Debit,Credit,Balance\n'
        '03 Apr 2025,POS,"R 1,150.00",,850\n'
        '04/04/2025,SAL,,"R15 000.00",15850\n'
        '05/04/2025,FEE,(12.50),,837.50\n'
    )
    assert df is not None, errors
    assert _amounts(df) == [-1150.0, 15000.0, -12.5]


def test_bracketed_negative_in_amount_column_is_not_dropped():
    df, errors = _read(
        'Date,Description,Amount\n'
        '2025-04-03,Rent,"(1,150.00)"\n'
        '2025-04-04,Deposit,200.00\n'
    )
    assert df is not None, errors
    assert _amounts(df) == [-1150.0, 200.0]


@pytest.mark.parametrize('raw, expected', [
    ('-150.00', -150.0),
    ('R -45.50', -45.5),
    ('R 1,000.00', 1000.0),
    ('(1,150.00)', -1150.0),
    ('150.00-', -150.0),
    ('150.00 Dr', -150.0),
    ('150.00Cr', 150.0),
    ('150,00', 150.0),
    ('15,000', 15000.0),
    ('1.234,56', 1234.56),
    ('R15 000.00', 15000.0),
    ('ZAR 99.99', 99.99),
    ('', None),
    ('abc', None),
])
def test_amount_parsing(raw, expected):
    assert _parse_number(raw) == (pytest.approx(expected) if expected is not None else None)


def test_negative_money_out_stays_money_out():
    # Capitec writes Money Out as a negative; subtracting it made a receipt.
    assert _signed_amount('-150.00', '') == pytest.approx(-150.0)
    assert _signed_amount('150.00', '') == pytest.approx(-150.0)
    assert _signed_amount('', '200.00') == pytest.approx(200.0)


def test_xlsx_with_title_rows_and_money_in_out():
    openpyxl = pytest.importorskip('openpyxl')
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Capitec Bank statement'])
    ws.append([])
    ws.append(['Transaction Date', 'Description', 'Money In', 'Money Out', 'Balance'])
    ws.append([datetime(2025, 4, 3), 'Woolies', None, -150.0, 850.0])
    ws.append([datetime(2025, 4, 4), 'Salary', 15000.0, None, 15850.0])
    buf = io.BytesIO()
    wb.save(buf)
    df, errors = _read(buf.getvalue(), suffix='.xlsx')
    assert df is not None, errors
    assert _amounts(df) == [-150.0, 15000.0]
    assert df.iloc[0]['Date'].day == 3 and df.iloc[0]['Date'].month == 4


def test_file_without_headings_gets_a_plain_message_naming_what_was_found():
    df, errors = _read('Name,Phone\nJohn,0821234567\n')
    assert df is None
    message = ' '.join(errors)
    assert 'Date column' in message
    assert 'Name' in message and 'Phone' in message
    assert 'tokenizing' not in message.lower()


def test_upload_route_imports_a_real_fnb_csv(canary_app):
    """End to end through the endpoint users hit."""
    from models import db, User, Account, Transaction

    app = canary_app
    with app.app_context():
        user = User(username='csv', email='csv@example.com',
                    subscription_status='active')
        user.set_password('Sup3rSecret!')
        db.session.add(user)
        db.session.commit()
        bank = Account(name='Bank', link='ca.810.001', category='Assets',
                       user_id=user.id, is_active=True)
        db.session.add(bank)
        db.session.commit()
        user_id, account_id = user.id, bank.id

    client = app.test_client()
    resp = client.post('/auth/login', data={'email': 'csv@example.com',
                                            'password': 'Sup3rSecret!'})
    assert '/login' not in resp.headers.get('Location', '')

    csv_bytes = (
        'ACC-NO,62000000000\nOpening Balance,1000.00\n\n'
        'Date,Amount,Balance,Description\n'
        '03/04/2025,-150.00,850.00,POS PURCHASE\n'
        '04/04/2025,15000.00,15850.00,SALARY\n'
    ).encode()
    resp = client.post('/bank-statements/upload', data={
        'account': str(account_id),
        'file': (io.BytesIO(csv_bytes), 'fnb.csv'),
    }, content_type='multipart/form-data',
        headers={'X-Requested-With': 'XMLHttpRequest'})
    body = resp.get_json()
    assert resp.status_code == 200, body
    assert body['success'] is True
    with app.app_context():
        assert Transaction.query.filter_by(user_id=user_id).count() == 2
