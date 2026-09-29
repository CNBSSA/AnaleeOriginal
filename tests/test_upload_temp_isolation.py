"""Two users uploading a file with the SAME name must never share a temp file
(2026-09-30). The upload used to be written to /tmp/<uploaded file name>, so
simultaneous "statement.xlsx" uploads wrote the same path — one user's rows
could be read into the other's account, and one request's cleanup deleted the
other's file mid-read.
"""
from __future__ import annotations

import io
import itertools
import os
import tempfile
from unittest import mock

import pandas as pd
from werkzeug.datastructures import FileStorage

from bank_statements import services as svc
from models import Account, CompanySettings, User, db

_seq = itertools.count(1)


def _user_and_bank():
    n = next(_seq)
    user = User(username=f'tmp{n}', email=f'tmp{n}@example.com', subscription_status='active')
    user.set_password('password')
    db.session.add(user)
    db.session.commit()
    db.session.add(CompanySettings(user_id=user.id, company_name='TEST', financial_year_end=2))
    bank = Account(link='ca.810.001', name='Bank', category='Assets', sub_category='Assets',
                   user_id=user.id)
    db.session.add(bank)
    db.session.commit()
    return user.id, bank.id


def _upload(name='statement.csv'):
    return FileStorage(stream=io.BytesIO(b'Date,Description,Amount\n2025-05-01,TEST,1.00\n'),
                       filename=name, content_type='text/csv')


def test_same_name_uploads_get_distinct_private_temp_files_and_are_cleaned_up(app):
    seen = []

    def fake_read(self, path):
        seen.append(path)
        assert os.path.exists(path), 'the file this request saved is still there while it is read'
        return pd.DataFrame()

    with app.app_context():
        uid_a, bank_a = _user_and_bank()
        uid_b, bank_b = _user_and_bank()
        with mock.patch.object(svc.BankStatementExcelReader, 'read_excel', fake_read):
            service = svc.BankStatementService()
            service.process_upload(_upload(), bank_a, uid_a)
            service.process_upload(_upload(), bank_b, uid_b)

    assert len(seen) == 2
    assert seen[0] != seen[1], 'two users with the same file name must not share a path'
    for path in seen:
        assert os.path.dirname(path) == tempfile.gettempdir()
        assert os.path.basename(path).startswith('analee-upload-')
        assert path.endswith('.csv'), 'the reader picks CSV vs Excel by the extension'
        assert not path.endswith('/statement.csv')
        assert not os.path.exists(path), 'cleaned up after processing'


def test_source_no_longer_builds_the_path_from_the_uploaded_name():
    src = open(svc.__file__).read()
    assert "os.path.join('/tmp', secure_filename(file.filename))" not in src
    assert 'tempfile.mkstemp(' in src
