"""Trial balance export — on-screen Excel download and BooksXperts workbook."""
from datetime import datetime
from decimal import Decimal
from io import BytesIO

import openpyxl
import pytest

from models import UploadedFile, Account, CompanySettings, Transaction, User, db
from reports.tb_page_excel import PAGE_TB_COLUMNS, build_trial_balance_page_xlsx
from reports.trial_balance_service import (
    BOOKSXPERTS_TB_COLUMNS,
    TrialBalanceRow,
    build_booksxperts_trial_balance_xlsx,
    export_filename,
    load_trial_balance,
)


def _settings(user_id: int, fy_end: int = 2) -> CompanySettings:
    settings = CompanySettings(
        user_id=user_id,
        company_name='ACME Pty Ltd',
        financial_year_end=fy_end,
    )
    db.session.add(settings)
    db.session.commit()
    return settings


def test_build_xlsx_uses_booksxperts_columns():
    rows = (
        TrialBalanceRow('ca.810.001', 'Bank Cheque Account 1', Decimal('50000')),
        TrialBalanceRow('i.100.000', 'Sales', Decimal('-50000')),
    )
    data = build_booksxperts_trial_balance_xlsx(
        rows,
        company_name='Test Co',
        period_end=datetime(2026, 2, 28),
    )
    wb = openpyxl.load_workbook(BytesIO(data))
    ws = wb.active
    assert list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0] == BOOKSXPERTS_TB_COLUMNS
    data_rows = list(ws.iter_rows(min_row=2, max_row=3, values_only=True))
    assert data_rows[0] == ('ca.810.001', 'Bank Cheque Account 1', 50000.0)
    assert data_rows[1] == ('i.100.000', 'Sales', -50000.0)
    footer_rows = [row[0] for row in ws.iter_rows(min_row=5, values_only=True) if row[0]]
    footer_text = ' '.join(str(cell) for cell in footer_rows)
    assert 'BooksXperts' in footer_text
    assert 'Accountants' in footer_text


def test_export_filename_includes_period_end():
    # ``export_filename`` takes the exclusive ``end_date`` from ``load_trial_balance``.
    end_exclusive = datetime(2026, 3, 1)
    assert export_filename(end_exclusive) == 'analee-trial-balance-2026-02-28.xlsx'


def test_load_trial_balance_builds_signed_amounts(app, sample_user):
    with app.app_context():
        _settings(sample_user)
        bank = Account(
            link='ca.810.001',
            name='Bank Cheque Account 1',
            category='Assets',
            sub_category='Current Asset',
            user_id=sample_user,
        )
        sales = Account(
            link='i.100.000',
            name='Sales',
            category='Income',
            sub_category='Income',
            user_id=sample_user,
        )
        db.session.add_all([bank, sales])
        db.session.flush()
        # ONE bank line (money in, bank-signed) on a statement whose bank is
        # recorded, categorised to Sales — what a real import produces. The
        # trial balance posts both sides (2026-09-28 double entry).
        statement = UploadedFile(filename='statement.csv', user_id=sample_user,
                                 bank_account_id=bank.id)
        db.session.add(statement)
        db.session.flush()
        db.session.add(
            Transaction(
                date=datetime(2026, 4, 15),
                description='Sale',
                amount=100.0,
                user_id=sample_user,
                account_id=sales.id,
                file_id=statement.id,
            ),
        )
        db.session.commit()

        ctx = load_trial_balance(sample_user)
        assert len(ctx.rows) == 2
        by_link = {r.link: r.amount for r in ctx.rows}
        assert by_link['ca.810.001'] == Decimal('100.00')
        assert by_link['i.100.000'] == Decimal('-100.00')
        assert ctx.total_debits == ctx.total_credits == Decimal('100.00')


def test_load_trial_balance_skips_zero_balances(app, sample_user):
    with app.app_context():
        _settings(sample_user)
        empty = Account(
            link='e.460.000',
            name='Salaries',
            category='Expenses',
            sub_category='Expense',
            user_id=sample_user,
        )
        db.session.add(empty)
        db.session.commit()

        ctx = load_trial_balance(sample_user)
        assert ctx.rows == ()


def _page_table_rows(xlsx_bytes: bytes):
    """Body + total rows from the on-screen export workbook."""
    wb = openpyxl.load_workbook(BytesIO(xlsx_bytes))
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    body = []
    total = None
    for row in rows[1:]:
        if row[0] == 'Total':
            total = row
            break
        if row[0] in PAGE_TB_COLUMNS or row[0] is None:
            continue
        body.append(row)
    return header, body, total


def test_build_page_xlsx_matches_on_screen_table(app, sample_user):
    with app.app_context():
        _settings(sample_user)
        bank = Account(
            link='ca.810.001',
            name='Bank Cheque Account 1',
            category='Assets',
            sub_category='Current Asset',
            user_id=sample_user,
        )
        sales = Account(
            link='i.100.000',
            name='Sales',
            category='Income',
            sub_category='Income',
            user_id=sample_user,
        )
        db.session.add_all([bank, sales])
        db.session.flush()
        statement = UploadedFile(
            filename='statement.csv',
            user_id=sample_user,
            bank_account_id=bank.id,
        )
        db.session.add(statement)
        db.session.flush()
        db.session.add(
            Transaction(
                date=datetime(2026, 4, 15),
                description='Sale',
                amount=100.0,
                user_id=sample_user,
                account_id=sales.id,
                file_id=statement.id,
            ),
        )
        db.session.commit()

        ctx = load_trial_balance(sample_user)
        data = build_trial_balance_page_xlsx(ctx, company_name='ACME Pty Ltd')
        header, body, total = _page_table_rows(data)
        assert header == PAGE_TB_COLUMNS
        assert len(body) == len(ctx.accounts)
        amounts = {row.link: row.amount for row in ctx.rows}
        for link, name, debit, credit in body:
            bal = amounts.get(link, Decimal('0'))
            assert debit == (float(bal) if bal > 0 else 0.0)
            assert credit == (float(-bal) if bal < 0 else 0.0)
        assert total[2] == float(ctx.total_debits)
        assert total[3] == float(ctx.total_credits)


def _canary_user_with_tb(app, *, email='tb-export@example.com', password='password'):
    """Logged-in canary client with company settings and one categorised sale."""
    with app.app_context():
        user = User(
            username='tbexport',
            email=email,
            subscription_status='active',
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        uid = user.id
        _settings(uid)
        bank = Account(
            link='ca.810.001',
            name='Bank',
            category='Assets',
            sub_category='Current Asset',
            user_id=uid,
        )
        sales = Account(
            link='i.100.000',
            name='Sales',
            category='Income',
            sub_category='Income',
            user_id=uid,
        )
        db.session.add_all([bank, sales])
        db.session.flush()
        statement = UploadedFile(filename='s.csv', user_id=uid, bank_account_id=bank.id)
        db.session.add(statement)
        db.session.flush()
        db.session.add(
            Transaction(
                date=datetime(2026, 4, 15),
                description='Sale',
                amount=50.0,
                user_id=uid,
                account_id=sales.id,
                file_id=statement.id,
            ),
        )
        db.session.commit()
    client = app.test_client()
    client.post('/auth/login', data={'email': email, 'password': password})
    return client, uid


def test_export_route_returns_valid_xlsx_matching_service(canary_app):
    client, uid = _canary_user_with_tb(canary_app)
    with canary_app.app_context():
        ctx = load_trial_balance(uid)
        expected = build_trial_balance_page_xlsx(ctx)
    resp = client.get('/trial-balance/export')
    assert resp.status_code == 200
    assert resp.mimetype.endswith('spreadsheetml.sheet')
    assert '.xlsx' in resp.headers['Content-Disposition']
    assert _page_table_rows(resp.data) == _page_table_rows(expected)


def test_export_route_honours_financial_year_query(canary_app):
    client, uid = _canary_user_with_tb(canary_app, email='tb-fy@example.com')
    with canary_app.app_context():
        bank = Account.query.filter_by(user_id=uid, link='ca.810.001').one()
        statement = UploadedFile(filename='prior.csv', user_id=uid, bank_account_id=bank.id)
        db.session.add(statement)
        db.session.flush()
        db.session.add(
            Transaction(
                date=datetime(2025, 6, 1),
                description='Prior year',
                amount=25.0,
                user_id=uid,
                account_id=bank.id,
                file_id=statement.id,
            ),
        )
        db.session.commit()
        closed = load_trial_balance(uid, as_at=datetime(2025, 6, 30))
        current = load_trial_balance(uid)
    resp = client.get('/trial-balance/export?financial_year=2025')
    assert resp.status_code == 200
    _, body_closed, total_closed = _page_table_rows(resp.data)
    assert total_closed[2] == float(closed.total_debits)

    resp_now = client.get('/trial-balance/export')
    _, _, total_now = _page_table_rows(resp_now.data)
    assert total_now[2] == float(current.total_debits)
    assert total_closed != total_now or closed.total_debits == current.total_debits


def test_export_route_empty_period_redirects_with_message(canary_app):
    with canary_app.app_context():
        user = User(
            username='tbempty',
            email='tb-empty@example.com',
            subscription_status='active',
        )
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        _settings(user.id)
    client = canary_app.test_client()
    client.post('/auth/login', data={'email': 'tb-empty@example.com', 'password': 'password'})
    resp = client.get('/trial-balance/export', follow_redirects=True)
    assert resp.status_code == 200
    assert b'No trial balance to export' in resp.data
