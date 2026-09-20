"""Read-only: which uploads carry the fingerprint of a transposed date.

WHY A FINGERPRINT AND NOT A REPAIR. Until 2026-09-20 the bank-statement
importer parsed every cell with ``dayfirst=True`` while ``excel_reader`` handed
it every cell as a string, so an ISO date — '2025-04-03', or an Excel date cell
stringified to '2025-04-03 00:00:00' — was read as YYYY-DD-MM and stored as
4 March. The parser is fixed. The stored rows cannot be, because the uploaded
file is written to /tmp and removed in a finally block and ``UploadedFile``
keeps only a filename: there is nothing left to re-read. An affected statement
has to be uploaded again, so the useful thing this can do is say WHICH ones.

THE FINGERPRINT, AND WHY IT IS NOT A GUESS. Transposing YYYY-MM-DD gives
YYYY-DD-MM, so every affected row in one statement takes its DAY from the
statement's MONTH — all of them the same number — while its MONTH takes the row's
original day, 1..12. A March statement therefore produces a run of rows all
dated the 3rd, scattered across up to twelve different months. Rows dated 13 or
later are untouched: month 13 does not exist, so pandas fell back and read them
correctly, which is why an affected file also keeps a normal-looking cluster in
its real month.

So the signature is: MANY DISTINCT MONTHS, and MOST ROWS SHARING ONE
DAY-OF-MONTH. Real bank statements do not look like that — a month's trading
spreads across many days and one or two months.

This reports a SUSPICION with the evidence attached, never a verdict. It writes
nothing and repairs nothing.
"""
from __future__ import annotations

from collections import Counter

# A single month's statement can legitimately touch two calendar months. Three
# or more distinct months in one upload is where this starts looking wrong.
MIN_DISTINCT_MONTHS = 3
# Most of the rows landing on one day-of-month is the part that does not happen
# by accident.
MIN_SHARE_ON_ONE_DAY = 0.5


def _assess(dates):
    """Score one upload's dates. Pure — takes dates, returns a dict or None."""
    if len(dates) < MIN_DISTINCT_MONTHS:
        return None
    months = {(d.year, d.month) for d in dates}
    if len(months) < MIN_DISTINCT_MONTHS:
        return None
    day_counts = Counter(d.day for d in dates)
    top_day, top_count = day_counts.most_common(1)[0]
    share = top_count / len(dates)
    if share < MIN_SHARE_ON_ONE_DAY:
        return None
    return {
        'rows': len(dates),
        'distinct_months': len(months),
        'shared_day': top_day,
        'rows_on_shared_day': top_count,
        'share_on_shared_day': round(share * 100, 1),
        'first': min(dates),
        'last': max(dates),
        # Under the fingerprint the shared day IS the statement's real month.
        'likely_real_month': top_day if 1 <= top_day <= 12 else None,
    }


def suspect_uploads(user_id=None):
    """Uploads whose transaction dates carry the transposition fingerprint.

    Returns a list of dicts, worst first. Read-only: no writes, no repair.
    """
    from models import Transaction, UploadedFile  # local import — avoids cycles

    files = UploadedFile.query
    if user_id is not None:
        files = files.filter_by(user_id=user_id)

    out = []
    for uploaded in files.all():
        dates = [t.date for t in
                 Transaction.query.filter_by(file_id=uploaded.id).all()
                 if t.date is not None]
        verdict = _assess(dates)
        if verdict:
            verdict.update({
                'file_id': uploaded.id,
                'filename': uploaded.filename,
                'upload_date': uploaded.upload_date,
                'user_id': uploaded.user_id,
            })
            out.append(verdict)

    out.sort(key=lambda r: (-r['distinct_months'], -r['share_on_shared_day']))
    return out
