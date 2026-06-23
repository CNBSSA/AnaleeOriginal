# Bug-Finding Memory

Tracks bugs reported across runs. One line each: location + root cause, PR URL, status, date recorded.
Only open or rejected PRs are kept here.

## Open

- uploads (`app.py` / `ocr` & other upload routes) — no `MAX_CONTENT_LENGTH`; routes read the uploaded file into memory before the size check, so a large body can exhaust worker RAM. Fixed by setting Flask `MAX_CONTENT_LENGTH` from `config.MAX_UPLOAD_BYTES`. PR: https://github.com/CNBSSA/AnaleeOriginal/pull/6 — status: open — recorded: 2026-06-23
