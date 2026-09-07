"""Process a statement the moment it is imported, without blocking the upload.

The accountant should not have to find a button. When a statement lands, the
rows that can be decided are decided — history first, model second — so that by
the time the file is opened only the genuine exceptions are waiting.

Why a background thread and not the request
-------------------------------------------
A 465-row statement is ~19 batched AI calls. Doing that inside the upload
request is exactly the mistake that made PDF extraction unusable for weeks:
gunicorn kills a worker that takes too long, *outside* Flask, so no error
handler runs and the user gets a bare 500. The upload therefore returns
immediately and the work continues on a daemon thread. The deploy runs
``--worker-class gthread --workers 3 --threads 4``, so there are threads to
spare for this.

Safety properties
-----------------
* **Never blocks** the request that triggered it.
* **Never crashes the worker** — every exception is caught and logged.
* **Idempotent.** It only ever fills empty slots, so a run interrupted by a
  deploy or a worker recycle can simply be re-run (or finished by the existing
  Auto-process button) with no double-writing.
* **Bounded.** Stops at ``MAX_BATCHES`` so a pathological file cannot occupy a
  thread indefinitely.
* **Switchable.** ``ANALEE_AUTOPROCESS_ON_IMPORT=0`` turns it off without a
  code change, for when someone wants to import without spending API credit.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

#: A batch is 25 rows, so this caps one import at 2 000 rows of automatic work.
MAX_BATCHES = 80

_FLAG = 'ANALEE_AUTOPROCESS_ON_IMPORT'


def autoprocess_enabled() -> bool:
    """On unless explicitly switched off."""
    return (os.environ.get(_FLAG, '1') or '1').strip().lower() not in ('0', 'false', 'no')


def run_file_autoprocess(app, file_id: int, user_id: int) -> dict:
    """Drain a file's outstanding rows. Synchronous — the thread body.

    Returns a summary dict. Never raises: an import must not be undone by a
    failure in the optional work that follows it.
    """
    summary = {'batches': 0, 'processed': 0, 'applied': 0,
               'explained': 0, 'from_history': 0, 'error': None}
    try:
        from services.analyze_processing import process_transaction_batch
        from models import db

        with app.app_context():
            try:
                offset = 0
                for _ in range(MAX_BATCHES):
                    result = process_transaction_batch(file_id, user_id, offset=offset)
                    summary['batches'] += 1
                    summary['processed'] += result.get('processed', 0)
                    summary['applied'] += result.get('applied', 0)
                    summary['explained'] += result.get('explained', 0)
                    summary['from_history'] += result.get('from_history', 0)

                    if not result.get('has_more'):
                        break
                    next_offset = result.get('next_offset', offset)
                    if result.get('processed', 0) == 0 and next_offset == offset:
                        # No progress possible (e.g. AI offline and no history):
                        # stop rather than spin through the batch cap.
                        break
                    offset = next_offset
                logger.info(
                    "Auto-process finished for file %s: %s batch(es), %s row(s), "
                    "%s account(s) assigned, %s explained, %s from history",
                    file_id, summary['batches'], summary['processed'],
                    summary['applied'], summary['explained'], summary['from_history'],
                )
            finally:
                # This thread owns its scoped session; hand it back.
                db.session.remove()
    except Exception as exc:  # never propagate out of a background thread
        summary['error'] = f"{type(exc).__name__}: {exc}"
        logger.exception("Auto-process failed for file %s", file_id)
    return summary


def schedule_file_autoprocess(app, file_id: int, user_id: int) -> bool:
    """Kick off :func:`run_file_autoprocess` off-request. Returns whether it started.

    Callers should ignore the return value for control flow — a failure to
    start is never a reason to fail an import that has already committed.
    """
    if not autoprocess_enabled():
        logger.info("Auto-process on import is disabled (%s)", _FLAG)
        return False
    try:
        thread = threading.Thread(
            target=run_file_autoprocess,
            args=(app, file_id, user_id),
            name=f"autoprocess-file-{file_id}",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        logger.exception("Could not start auto-process thread for file %s", file_id)
        return False
