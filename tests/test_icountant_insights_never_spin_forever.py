"""Ask Analee (iCountant): AI Insights must never spin forever (work-orders #56).

Live QA 2026-10-03: "Loading AI insights..." never resolved on /icountant.
Two template defects, neither in the frozen suggestion machinery:

* the page's script sat in a ``{% block scripts %}`` nested INSIDE
  ``{% block content %}``, so Jinja rendered it twice (inline and again in the
  base layout's scripts block): every visit fired two insights requests, each
  making two AI calls, for one spinner;
* the request had no time limit. The AI client waits up to ten minutes and
  retries, so a slow call left the spinner up for as long as the visitor
  stayed. The page now gives up after a stated time and says so plainly,
  leaving the account dropdown free.

TEST data only. The insights endpoint and the AI calls are unchanged.
"""
from datetime import datetime

import pytest

pytest.importorskip("flask_sqlalchemy")

from test_asf_degradation_guard import EMAIL, _register_and_login


def _waiting_line(app):
    from models import Transaction, User, db
    with app.app_context():
        uid = User.query.filter_by(email=EMAIL).one().id
        db.session.add(Transaction(date=datetime(2026, 3, 1),
                                   description='TEST waiting line', amount=-50.0,
                                   user_id=uid, account_id=None))
        db.session.commit()


def _page(app):
    client = app.test_client()
    _register_and_login(client)
    _waiting_line(app)
    page = client.get('/icountant').get_data(as_text=True)
    assert 'TEST waiting line' in page
    return page


def test_the_insights_request_is_made_once_per_visit(canary_app):
    page = _page(canary_app)
    assert page.count('/api/icountant/${transactionId}/insights') == 1


def test_a_slow_insights_request_gives_up_with_a_plain_message(canary_app):
    page = _page(canary_app)
    assert 'AbortController' in page
    assert 'signal: controller.signal' in page
    assert 'AI insights are taking too long' in page
