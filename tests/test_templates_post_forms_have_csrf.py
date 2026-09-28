"""Every POST form a browser submits must carry the CSRF token.

Found while fixing QA 2026-09-28 #2/#3 (Edit Account, Generate AI Insights):
the same defect sat on seven more buttons - the iCountant Assistant's
"approve" form, Goals, Historical Data upload, Risk Assessment, Alerts and both
Recommendations forms. CSRFProtect is on app-wide, so each answered 400 "The
CSRF token is missing". This guard stops the next one shipping.

The client-explain wizard is the one deliberate exception: a no-login flow whose
blueprint is csrf.exempt (see app.py) and secured by its signed token instead.
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXEMPT = {os.path.join('templates', 'client_explain', 'wizard.html')}


def _post_forms_without_token():
    missing = []
    for path in glob.glob(os.path.join(ROOT, 'templates', '**', '*.html'), recursive=True):
        rel = os.path.relpath(path, ROOT)
        if rel in EXEMPT:
            continue
        with open(path, encoding='utf-8') as handle:
            text = re.sub(r'<!--.*?-->', '', handle.read(), flags=re.S)
        for match in re.finditer(r'<form\b([^>]*)>(.*?)</form>', text, re.S | re.I):
            attrs, body = match.group(1), match.group(2)
            if not re.search(r'method=["\']?post', attrs, re.I):
                continue
            if 'csrf_token' in body or 'hidden_tag' in body:
                continue
            line = text[:match.start()].count('\n') + 1
            missing.append(f'{rel}:{line}')
    return missing


def test_every_post_form_carries_a_csrf_token():
    missing = _post_forms_without_token()
    assert not missing, (
        'POST forms without a CSRF token (the app answers 400): ' + ', '.join(missing))
