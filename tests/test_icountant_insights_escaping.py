"""AI insight text is written to the page as TEXT, never as HTML (work-orders #69, S1).

templates/icountant.html did `aiInsightsContent.innerHTML = data.ai_insights`.
The insight is free text the model generated from a transaction description
that came out of an uploaded statement; if the model echoes markup from such a
description, innerHTML executes it in the browser of whoever is reviewing — in
a Practice-Layer workspace, the accountant looking at a client's statement.
"""
import re

TEMPLATE = "templates/icountant.html"


def _source():
    with open(TEMPLATE, encoding="utf-8") as fh:
        return fh.read()


def test_the_insight_text_is_not_assigned_to_innerhtml():
    src = _source()
    assert "innerHTML = data.ai_insights" not in src
    assert not re.search(r"innerHTML\s*=\s*[^;]*data\.ai_insights", src)
    assert re.search(r"textContent\s*=\s*data\.ai_insights", src), \
        "the insight must be written with textContent"


def test_the_error_branch_is_text_too():
    src = _source()
    tail = src[src.index(".catch((error)"):]
    assert "innerHTML" not in tail.split(".finally(")[0]


def test_line_breaks_in_the_text_still_read():
    """textContent collapses newlines unless the container preserves them."""
    src = _source()
    container = re.search(r'<div[^>]*id="aiInsightsContent"[^>]*>', src).group(0)
    assert "pre-wrap" in container
