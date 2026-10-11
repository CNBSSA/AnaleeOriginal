"""Analee must never present an invented number as analysis.

``/expense-forecast`` carried ``overall_confidence: 0.85`` and
``reliability_score: 0.80`` as hardcoded literals that nothing recomputed,
rendered as "85.0%" / "80.0%" on the page and in the client-facing PDF.
"""
import inspect
import re

import pytest

pytest.importorskip("flask_sqlalchemy")


def _code_only(func):
    """Source of ``func`` with comment lines stripped.

    The fixes are documented in comments that necessarily quote the removed
    strings, so these guards must look at executable code, not commentary.
    """
    lines = inspect.getsource(func).splitlines()
    return '\n'.join(ln for ln in lines if not ln.strip().startswith('#'))


def test_expense_forecast_has_no_hardcoded_confidence_literals():
    """The forecast must not assert a precision it never computed."""
    import routes

    code = _code_only(routes.expense_forecast)
    # The literals are gone from the payload...
    assert not re.search(r"'overall_confidence':\s*0\.85", code)
    assert not re.search(r"'reliability_score':\s*0\.80", code)
    # ...and the basis that IS measured is reported instead.
    assert 'months_observed' in code


def test_forecast_templates_do_not_render_the_removed_metrics():
    """Both the page and the client-facing PDF must stop printing them —
    a stale reference would also raise at render time."""
    for path in ('templates/expense_forecast.html',
                 'templates/pdf_templates/forecast_pdf.html'):
        with open(path, encoding='utf-8') as handle:
            markup = handle.read()
        assert 'overall_confidence' not in markup, f"{path} still prints a fabricated confidence"
        assert 'reliability_score' not in markup, f"{path} still prints a fabricated reliability"
        assert 'months_observed' in markup, f"{path} must state the basis it does have"
