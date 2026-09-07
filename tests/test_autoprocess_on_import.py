"""Importing a statement starts the work; the upload does not wait for it.

The accountant should not have to find a button. But a 465-row statement is
~19 batched AI calls, and doing that inside the upload request is precisely the
mistake that made PDF extraction unusable: gunicorn kills the worker from
outside Flask, so no error handler runs and the user gets a bare 500.

These tests pin both halves — that the work is started, and that the request is
never the thing doing it.
"""
import threading
import time

import pytest

pytest.importorskip("flask_sqlalchemy")

from services.auto_process import (
    MAX_BATCHES,
    autoprocess_enabled,
    run_file_autoprocess,
    schedule_file_autoprocess,
)


# --- the switch ------------------------------------------------------------

def test_disabled_by_default(monkeypatch):
    """Dark by default (Festus, 2026-09-07) — an import must not spend API
    credit unless someone asked for it."""
    monkeypatch.delenv('ANALEE_AUTOPROCESS_ON_IMPORT', raising=False)
    assert autoprocess_enabled() is False


@pytest.mark.parametrize('value', ['1', 'true', 'yes', 'TRUE'])
def test_can_be_switched_on(monkeypatch, value):
    monkeypatch.setenv('ANALEE_AUTOPROCESS_ON_IMPORT', value)
    assert autoprocess_enabled() is True


@pytest.mark.parametrize('value', ['0', 'false', 'no', 'FALSE'])
def test_can_be_switched_off_without_a_code_change(monkeypatch, value):
    monkeypatch.setenv('ANALEE_AUTOPROCESS_ON_IMPORT', value)
    assert autoprocess_enabled() is False


def test_disabled_means_nothing_is_started(monkeypatch):
    monkeypatch.setenv('ANALEE_AUTOPROCESS_ON_IMPORT', '0')
    assert schedule_file_autoprocess(object(), 1, 1) is False


# --- the loop --------------------------------------------------------------

def test_it_drains_the_file_until_there_is_no_more(app, monkeypatch):
    calls = []

    def _fake_batch(file_id, user_id, offset=0, **kwargs):
        calls.append(offset)
        done = len(calls) >= 3
        return {'processed': 25, 'applied': 20, 'explained': 25,
                'from_history': 5, 'has_more': not done,
                'next_offset': offset + 5}

    monkeypatch.setattr('services.analyze_processing.process_transaction_batch', _fake_batch)

    summary = run_file_autoprocess(app, file_id=1, user_id=1)

    assert summary['batches'] == 3
    assert summary['processed'] == 75
    assert summary['explained'] == 75
    assert summary['from_history'] == 15
    assert summary['error'] is None


def test_it_stops_instead_of_spinning_when_no_progress_is_possible(app, monkeypatch):
    """AI offline and no history: every batch does nothing. It must stop, not
    burn through the batch cap."""
    calls = []

    def _stuck(file_id, user_id, offset=0, **kwargs):
        calls.append(offset)
        return {'processed': 0, 'applied': 0, 'explained': 0, 'from_history': 0,
                'has_more': True, 'next_offset': offset}

    monkeypatch.setattr('services.analyze_processing.process_transaction_batch', _stuck)
    summary = run_file_autoprocess(app, file_id=1, user_id=1)

    assert summary['batches'] == 1, "spun on a batch that could make no progress"


def test_it_is_bounded_even_when_work_never_ends(app, monkeypatch):
    def _endless(file_id, user_id, offset=0, **kwargs):
        return {'processed': 25, 'applied': 0, 'explained': 25, 'from_history': 0,
                'has_more': True, 'next_offset': offset + 25}

    monkeypatch.setattr('services.analyze_processing.process_transaction_batch', _endless)
    summary = run_file_autoprocess(app, file_id=1, user_id=1)

    assert summary['batches'] == MAX_BATCHES


def test_a_failure_is_reported_not_raised(app, monkeypatch):
    """An import that has already committed must never be undone by a failure
    in the optional work that follows it."""
    def _boom(file_id, user_id, offset=0, **kwargs):
        raise RuntimeError('anthropic exploded')

    monkeypatch.setattr('services.analyze_processing.process_transaction_batch', _boom)

    summary = run_file_autoprocess(app, file_id=1, user_id=1)  # must not raise
    assert 'anthropic exploded' in summary['error']


# --- off-request -----------------------------------------------------------

def test_both_import_paths_start_it():
    """PDF and CSV/Excel imports must both kick this off — a statement should
    never sit untouched just because it came in through the other door."""
    import inspect

    from bank_statements import routes as bs_routes
    from ocr import routes as ocr_routes

    csv_source = inspect.getsource(bs_routes.upload)
    assert 'schedule_file_autoprocess' in csv_source, "CSV/Excel import never starts it"

    ocr_source = inspect.getsource(ocr_routes.confirm_receipt)
    assert '_start_autoprocess' in ocr_source, "PDF import never starts it"


def test_scheduling_returns_immediately_and_runs_off_request(app, monkeypatch):
    """The caller must not wait for the work."""
    started = threading.Event()
    release = threading.Event()

    def _slow(file_id, user_id, offset=0, **kwargs):
        started.set()
        release.wait(timeout=5)
        return {'processed': 0, 'applied': 0, 'explained': 0, 'from_history': 0,
                'has_more': False, 'next_offset': offset}

    monkeypatch.setattr('services.analyze_processing.process_transaction_batch', _slow)
    monkeypatch.setenv('ANALEE_AUTOPROCESS_ON_IMPORT', '1')

    began = time.monotonic()
    assert schedule_file_autoprocess(app, 1, 1) is True
    elapsed = time.monotonic() - began

    assert elapsed < 1.0, "scheduling blocked the caller"
    assert started.wait(timeout=5), "the work never actually started"
    release.set()
