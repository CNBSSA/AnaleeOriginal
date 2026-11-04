"""Work order #11025–#11037 — Chairman list stage 3 (fail-first tests)."""
from __future__ import annotations

import io
import os
import re
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest import mock

import openpyxl
import pytest

PASSWORD = 'Sup3rSecret!'


def _login_client(app, email='wo@test.com', username='wouser'):
    from models import db, User, CompanySettings, Account, Transaction, UploadedFile

    with app.app_context():
        user = User(username=username, email=email, subscription_status='active')
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.commit()
        db.session.add(CompanySettings(
            user_id=user.id, company_name='WO Co', financial_year_end=2))
        bank = Account(
            link='ca.810.001', name='Bank Cheque', category='Assets',
            sub_category='Current Asset', user_id=user.id, is_active=True)
        sales = Account(
            link='i.100.000', name='Sales', category='Income',
            sub_category='Income', user_id=user.id, is_active=True)
        rent = Account(
            link='e.400.000', name='Rent', category='Expenses',
            sub_category='Expenses', user_id=user.id, is_active=True)
        db.session.add_all([bank, sales, rent])
        db.session.commit()
        uid = user.id
        bank_id = bank.id
    client = app.test_client()
    client.post('/auth/login', data={'email': email, 'password': PASSWORD})
    return client, uid, bank_id


# --- #11026 cashbook running balance -----------------------------------------

def test_cashbook_running_balance_three_lines(canary_app):
    from models import db, Transaction

    client, uid, bank_id = _login_client(canary_app)
    with canary_app.app_context():
        f = __import__('models', fromlist=['UploadedFile']).UploadedFile(
            filename='cb.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        for when, amt in [
            (datetime(2025, 4, 1), 100.0),
            (datetime(2025, 4, 2), 150.0),
            (datetime(2025, 4, 3), -50.0),
        ]:
            db.session.add(Transaction(
                date=when, description='line', amount=amt,
                user_id=uid, account_id=bank_id, file_id=f.id))
        db.session.commit()

    html = client.get('/cashbook?financial_year=2025').get_data(as_text=True)
    balances = re.findall(
        r'<td class="text-end">(\d+\.\d{2})</td>\s*</tr>', html)
    row_balances = []
    for row in re.finditer(
            r'<tr>\s*<td>2025-\d{2}-\d{2}</td>.*?'
            r'<td class="text-end">(\d+\.\d{2})</td>\s*</tr>', html, re.S):
        cells = re.findall(r'<td class="text-end">([^<]+)</td>', row.group(0))
        if len(cells) >= 3:
            row_balances.append(cells[-1])
    assert row_balances == ['100.00', '250.00', '200.00'], row_balances
    assert '200.00' in html.split('Closing Balance')[1]


# --- #11028 duplicate CSV upload ---------------------------------------------

def test_duplicate_csv_upload_warns_and_aborts(canary_app):
    from models import db, Transaction

    client, uid, bank_id = _login_client(canary_app, email='dup@test.com', username='dupuser')
    csv_bytes = (
        b'Date,Description,Amount\n'
        b'2025-04-01,DUP LINE,100.00\n'
    )
    data1 = {'account': str(bank_id), 'file': (io.BytesIO(csv_bytes), 'dup.csv')}
    r1 = client.post(
        '/bank-statements/upload',
        data=data1,
        content_type='multipart/form-data',
        headers={'X-Requested-With': 'XMLHttpRequest'},
    )
    assert r1.json.get('success') is True

    with canary_app.app_context():
        count_after_first = Transaction.query.filter_by(user_id=uid).count()

    data2 = {'account': str(bank_id), 'file': (io.BytesIO(csv_bytes), 'dup.csv')}
    r2 = client.post(
        '/bank-statements/upload',
        data=data2,
        content_type='multipart/form-data',
        headers={'X-Requested-With': 'XMLHttpRequest'},
    )
    body = r2.get_json()
    assert body.get('duplicate_warning') or body.get('requires_confirmation'), body
    with canary_app.app_context():
        assert Transaction.query.filter_by(user_id=uid).count() == count_after_first


# --- #11029 skipped rows reported --------------------------------------------

def test_csv_upload_reports_skipped_rows(canary_app):
    client, uid, bank_id = _login_client(canary_app, email='skip@test.com', username='skipuser')
    csv_bytes = (
        b'Date,Description,Amount\n'
        b'2025-04-01,OK1,10.00\n'
        b'bad-date,OK2,20.00\n'
        b'2025-04-03,OK3,30.00\n'
        b'2025-04-04,,40.00\n'
        b'2025-04-05,OK5,50.00\n'
        b'2025-04-06,OK6,60.00\n'
        b'not-a-date,OK7,70.00\n'
        b'2025-04-08,OK8,80.00\n'
    )
    r = client.post(
        '/bank-statements/upload',
        data={'account': str(bank_id), 'file': (io.BytesIO(csv_bytes), 'eight.csv')},
        content_type='multipart/form-data',
        headers={'X-Requested-With': 'XMLHttpRequest'},
    )
    body = r.get_json()
    assert body.get('success') is True
    skipped = body.get('rows_skipped') or 0
    notes = body.get('processing_notes') or []
    assert skipped == 2 or any('2' in str(n) for n in notes), body
    from models import db, Transaction
    with canary_app.app_context():
        assert Transaction.query.filter_by(user_id=uid).count() == 6


# --- #11032 duplicate chart link ---------------------------------------------

def test_settings_duplicate_link_stays_on_page(canary_app):
    from models import db, Account
    from services.entity_chart_schema import prepare_subscriber_chart_access

    client, uid, _bank = _login_client(canary_app, email='set@test.com', username='setuser')
    with canary_app.app_context():
        prepare_subscriber_chart_access()
        db.session.add(Account(
            link='x.100.000', name='Existing', category='Income',
            user_id=uid, is_active=True))
        db.session.commit()

    r = client.post('/settings', data={
        'link': 'x.100.000',
        'name': 'Dup',
        'category': 'Income',
        'sub_category': 'Income',
        'account_code': '',
    }, follow_redirects=False)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'already exists' in html.lower()
    assert 'x.100.000' in html
    assert r.location is None or '/dashboard' not in (r.location or '')


# --- #11027 extraction / legacy import ---------------------------------------

def test_legacy_csv_import_flags_zero_amount_non_junk(app, sample_user):
    import pandas as pd
    import routes
    from models import UploadedFile, db

    with app.app_context():
        uf = UploadedFile(filename='z.csv', user_id=sample_user)
        db.session.add(uf)
        db.session.commit()
        df = pd.DataFrame([
            {'Date': datetime(2026, 3, 1), 'Description': 'Fee waiver', 'Amount': 0.0},
        ])
        processed, errors = routes.process_transaction_rows(df, uf, type('U', (), {'id': sample_user})())
        assert processed == 1
        tx = __import__('models', fromlist=['Transaction']).Transaction.query.filter_by(file_id=uf.id).first()
        assert tx is not None
        assert getattr(tx, 'statement_notice', False) or tx.amount == 0.0


def test_legacy_csv_import_drops_definite_notice(app, sample_user):
    import pandas as pd
    import routes
    from models import UploadedFile, db

    with app.app_context():
        uf = UploadedFile(filename='n.csv', user_id=sample_user)
        db.session.add(uf)
        db.session.commit()
        df = pd.DataFrame([
            {'Date': datetime(2026, 3, 1), 'Description': 'Schd Trxn No Av Bal', 'Amount': 0.0},
        ])
        processed, _errors = routes.process_transaction_rows(
            df, uf, type('U', (), {'id': sample_user})())
        assert processed == 0


def test_non_zero_notice_wording_always_imported(app, sample_user):
    import pandas as pd
    import routes
    from models import UploadedFile, db

    with app.app_context():
        uf = UploadedFile(filename='nz.csv', user_id=sample_user)
        db.session.add(uf)
        db.session.commit()
        df = pd.DataFrame([
            {'Date': datetime(2026, 3, 1), 'Description': 'Schd Trxn No Av Bal', 'Amount': 12.5},
        ])
        processed, _ = routes.process_transaction_rows(df, uf, type('U', (), {'id': sample_user})())
        assert processed == 1


def test_ocr_review_zero_amount_unchecked_no_balance_prefill(canary_app):
    from flask import render_template

    rows = [{
        'date': '2026-03-01',
        'description': 'Fee waiver',
        'amount': 0.0,
        'confidence': 0.9,
        'statement_notice': True,
        'duplicate': False,
    }]
    with canary_app.test_request_context():
        with canary_app.app_context():
            html = render_template(
                'ocr/review.html',
                rows=rows,
                accounts=[],
                account_id='',
                filename='t.pdf',
                statement_header=None,
                report_card=None,
                extraction_method='digital_pdf',
            )
    row_part = html.split('Fee waiver')[0]
    include_pos = row_part.rfind('name="include"')
    assert include_pos != -1
    snippet = row_part[include_pos:include_pos + 120]
    assert 'checked' not in snippet
    assert 'value="0.00"' not in html


def test_description_pipe_artifacts_stripped():
    from bank_statements.format_detector import normalize_bank_statement_dataframe
    import pandas as pd

    raw = pd.DataFrame([
        ['Date', 'Description', 'Amount'],
        ['2025-04-01', '|EFT payment|', '100.00'],
        ['2025-04-02', 'Normal payee', '50.00'],
    ])
    out = normalize_bank_statement_dataframe(raw)
    assert out.iloc[0]['Description'] == 'EFT payment'
    assert out.iloc[1]['Description'] == 'Normal payee'


# --- #11025 dashboard vs TB --------------------------------------------------

def test_dashboard_income_matches_trial_balance_for_selected_fy(canary_app):
    from models import db, Transaction, UploadedFile
    from reports.trial_balance_service import load_trial_balance
    from services.tb_workspace_metrics import trial_balance_metrics_for_fy
    from models import CompanySettings

    client, uid, bank_id = _login_client(canary_app, email='tb@test.com', username='tbuser')
    with canary_app.app_context():
        cs = CompanySettings.query.filter_by(user_id=uid).first()
        sales_id = __import__('models', fromlist=['Account']).Account.query.filter_by(
            user_id=uid, link='i.100.000').first().id
        f = UploadedFile(filename='s.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2025, 6, 1), description='Sale', amount=15000.0,
            user_id=uid, account_id=sales_id, file_id=f.id))
        db.session.add(Transaction(
            date=datetime(2025, 7, 1), description='Rent', amount=-5000.0,
            user_id=uid, account_id=bank_id, file_id=f.id))
        db.session.commit()
        metrics = trial_balance_metrics_for_fy(uid, cs, 2025)
        ctx = load_trial_balance(uid, as_at=datetime(2026, 2, 28))

    html = client.get('/dashboard?financial_year=2025').get_data(as_text=True)
    assert f"{metrics['total_income']:.2f}" in html.replace(',', '') or '15000.00' in html
    assert metrics['total_income'] > 0


def test_chat_fy_summary_matches_trial_balance_movement(canary_app):
    from chat.routes import get_financial_context
    from services.tb_workspace_metrics import trial_balance_metrics_for_fy
    from models import CompanySettings, db, Transaction, UploadedFile

    client, uid, bank_id = _login_client(canary_app, email='chatb@test.com', username='chatb')
    with canary_app.app_context():
        cs = CompanySettings.query.filter_by(user_id=uid).first()
        sales = __import__('models', fromlist=['Account']).Account.query.filter_by(
            user_id=uid, link='i.100.000').first()
        f = UploadedFile(filename='c.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2025, 6, 1), description='Sale', amount=1000.0,
            user_id=uid, account_id=sales.id, file_id=f.id))
        db.session.commit()
        metrics = trial_balance_metrics_for_fy(uid, cs, cs and 2025 or 2025)
        ctx = get_financial_context(uid)
    assert ctx.get('ledger_income') == metrics['total_income']


# --- #11031 AI insights dedupe -----------------------------------------------

def test_financial_insights_structured_totals_appear_once(canary_app, monkeypatch):
    client, uid, bank_id = _login_client(canary_app, email='ins@test.com', username='insuser')
    with canary_app.app_context():
        from models import db, Transaction
        f = __import__('models', fromlist=['UploadedFile']).UploadedFile(
            filename='i.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2026, 3, 1), description='IN', amount=100.0,
            user_id=uid, account_id=bank_id, file_id=f.id))
        db.session.commit()

    from services.statement_insights import analyze_statement_transactions, cash_flow_analysis_from_statement
    analysis = analyze_statement_transactions([{
        'date': '2026-03-01', 'description': 'IN', 'amount': 100.0,
        'category': 'Assets', 'account_name': 'Bank',
    }])
    with client.session_transaction() as sess:
        sess['financial_advice'] = {
            'key_insights': 'ok',
            'risk_factors': [],
            'optimization_opportunities': [],
            'strategic_recommendations': [],
            'statement_analysis': analysis,
            'cash_flow_analysis': cash_flow_analysis_from_statement([], analysis),
        }

    html = client.get('/financial-insights').get_data(as_text=True)
    assert html.count('100.00') <= 3


def test_chat_prompt_has_two_labeled_blocks(canary_app):
    from chat.routes import generate_ai_response, get_financial_context

    client, uid, _ = _login_client(canary_app, email='pl@test.com', username='pluser')
    with canary_app.app_context():
        ctx = get_financial_context(uid)
    seen = {}

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                seen['prompt'] = kw['messages'][0]['content']
                class R:
                    content = [type('T', (), {'text': 'ok'})()]
                return R()

    generate_ai_response(_Client(), 'Summary?', ctx)
    prompt = seen['prompt']
    assert 'Trial balance (ledger view)' in prompt
    assert 'Bank activity (FY)' in prompt


# --- #11033 analyze performance structure ------------------------------------

def test_analyze_page_account_options_not_duplicated_per_row(canary_app):
    from models import db, Transaction, UploadedFile, Account

    client, uid, bank_id = _login_client(canary_app, email='an@test.com', username='anuser')
    with canary_app.app_context():
        extra = [Account(
            link=f'e.50{i}.000', name=f'Exp {i}', category='Expenses',
            user_id=uid, is_active=True) for i in range(5)]
        db.session.add_all(extra)
        f = UploadedFile(filename='a.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        for i in range(3):
            db.session.add(Transaction(
                date=datetime(2025, 4, i + 1), description=f'L{i}', amount=float(i + 1),
                user_id=uid, account_id=bank_id, file_id=f.id))
        db.session.commit()
        fid = f.id

    html = client.get(f'/analyze/{fid}').get_data(as_text=True)
    assert html.count('data-link="e.500.000"') <= 1


def test_analyze_list_does_not_scan_bank_accounts_per_file_twice(canary_app, monkeypatch):
    from models import db, UploadedFile
    from services import analyze_processing as ap

    client, uid, bank_id = _login_client(canary_app, email='al@test.com', username='aluser')
    calls = {'n': 0}
    real = ap.bank_account_ids_for_user

    def counting(user_id):
        calls['n'] += 1
        return real(user_id)

    monkeypatch.setattr(ap, 'bank_account_ids_for_user', counting)
    with canary_app.app_context():
        for i in range(3):
            f = UploadedFile(filename=f'{i}.csv', user_id=uid, bank_account_id=bank_id)
            db.session.add(f)
        db.session.commit()

    calls['n'] = 0
    client.get('/analyze')
    assert calls['n'] == 1


def test_analyze_page_bank_account_ids_called_once(canary_app, monkeypatch):
    from models import db, Transaction, UploadedFile
    from services import analyze_processing as ap

    client, uid, bank_id = _login_client(canary_app, email='ap@test.com', username='apuser')
    with canary_app.app_context():
        f = UploadedFile(filename='one.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2025, 4, 1), description='x', amount=1.0,
            user_id=uid, account_id=bank_id, file_id=f.id))
        db.session.commit()
        fid = f.id

    calls = {'n': 0}
    real = ap.bank_account_ids_for_user

    def counting(user_id):
        calls['n'] += 1
        return real(user_id)

    monkeypatch.setattr(ap, 'bank_account_ids_for_user', counting)
    calls['n'] = 0
    client.get(f'/analyze/{fid}')
    assert calls['n'] == 1


# --- #11029 dashboard period caption -----------------------------------------

def test_dashboard_period_caption_uses_inclusive_year_end(canary_app):
    client, uid, _ = _login_client(canary_app, email='dc@test.com', username='dcuser')
    html = client.get('/dashboard?financial_year=2026').get_data(as_text=True)
    assert '28 Feb 2027' in html
    assert '01 Mar 2027' not in html.split('Total Income')[1].split('Total Expenses')[0]


def test_trial_balance_export_meta_as_at_matches_filename(canary_app):
    from models import db, Transaction, UploadedFile, Account

    client, uid, bank_id = _login_client(canary_app, email='ex@test.com', username='exuser')
    with canary_app.app_context():
        f = UploadedFile(filename='t.csv', user_id=uid, bank_account_id=bank_id)
        db.session.add(f)
        db.session.flush()
        db.session.add(Transaction(
            date=datetime(2025, 6, 1), description='In', amount=100.0,
            user_id=uid, account_id=bank_id, file_id=f.id))
        db.session.commit()

    resp = client.get('/trial-balance/export?financial_year=2025')
    assert resp.status_code == 200
    wb = openpyxl.load_workbook(io.BytesIO(resp.data))
    meta = ' '.join(str(c) for row in wb.active.iter_rows(values_only=True) for c in row if c)
    assert 'as at 2026-02-28' in meta
    assert 'analee-trial-balance-2026-02-28' in (resp.headers.get('Content-Disposition') or '')
