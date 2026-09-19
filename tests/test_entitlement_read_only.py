"""A lapsed member keeps sight of their own books. They just cannot do more work.

The landmine this closes (found in the 2026-09-19 estate sweep): the entitlement
gate redirected EVERY route for a non-entitled user. It is dark, so nobody had
been hurt yet — but the day ``ANALEE_ENTITLEMENT_ENFORCED`` was switched on, a
member whose Club membership or subscription lapsed would have lost sight of
statements and trial balances they produced and paid for. Across the whole
product, not five views.

That is byte-for-byte the TrustEasyGo defect Festus ruled on the same day:

    "A normal application doesn't behave that way. Who will buy an application
    like [that]?"

    A gate may stop you WRITING new data. It must never stop you READING your
    own.

The commercial rule Festus set in July is deliberately preserved: a non-entitled
user can look and export, but cannot upload, analyse, categorise or post. Both
halves are tested here — the second set is what stops this "fix" from quietly
giving Analee away.
"""
import pytest

import entitlement

EMAIL = "readonly@example.com"
PASSWORD = "Sup3rSecret!"


def _register_and_login(client):
    client.post("/auth/register", data={
        "username": "readonlyuser", "email": EMAIL,
        "password": PASSWORD, "confirm_password": PASSWORD,
    })
    client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})


@pytest.fixture()
def locked_out(canary_app, monkeypatch):
    """A logged-in user the gate considers NOT entitled, with the gate ON."""
    monkeypatch.setenv("ANALEE_ENTITLEMENT_ENFORCED", "True")
    monkeypatch.setattr(entitlement, "analee_entitled", lambda _user: False)
    client = canary_app.test_client()
    _register_and_login(client)
    return client


# --- the pure policy --------------------------------------------------------

def test_reads_are_allowed():
    assert entitlement.read_only_allowed("GET", "main.dashboard") is True
    assert entitlement.read_only_allowed("HEAD", "main.dashboard") is True
    assert entitlement.read_only_allowed("get", "main.dashboard") is True


def test_writes_are_never_allowed():
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert entitlement.read_only_allowed(method, "main.dashboard") is False, method


def test_a_get_that_grants_access_is_not_a_read():
    """Minting a signed URL lets a THIRD PARTY in. That is not reading."""
    assert entitlement.read_only_allowed("GET", "main.analyze_client_link") is False


def test_a_get_that_spends_money_is_not_a_read():
    """This endpoint calls the AI on every request."""
    assert entitlement.read_only_allowed(
        "GET", "main.icountant_transaction_insights") is False


# --- the gate, end to end ---------------------------------------------------

def test_a_lapsed_member_can_still_see_their_dashboard(locked_out):
    assert locked_out.get("/dashboard").status_code == 200


def test_a_lapsed_member_can_still_reach_their_uploaded_files(locked_out):
    resp = locked_out.get("/analyze")
    assert resp.status_code == 200, (
        "a lapsed member was denied sight of files they uploaded")


def test_a_lapsed_member_cannot_do_new_work(locked_out):
    """Safety twin: without this, the fix would hand Analee away for free."""
    resp = locked_out.post("/analyze", data={})
    assert resp.status_code == 302
    assert "/entitlement-required" in resp.headers.get("Location", "")


def test_a_lapsed_member_cannot_mint_a_client_link(locked_out):
    resp = locked_out.get("/api/analyze/1/client-link")
    assert resp.status_code == 302
    assert "/entitlement-required" in resp.headers.get("Location", "")


def test_the_notice_page_is_still_reachable(locked_out):
    assert locked_out.get("/entitlement-required").status_code == 200


def test_the_gate_stays_a_complete_no_op_while_dark(canary_app, monkeypatch):
    """Vacuity guard: everything above is about the gate being ON. With the
    flag off nothing may change at all — including the new read-only path."""
    monkeypatch.delenv("ANALEE_ENTITLEMENT_ENFORCED", raising=False)

    def _boom(_user):
        raise AssertionError("entitlement helper called while the gate is OFF")

    monkeypatch.setattr(entitlement, "analee_entitled", _boom)
    client = canary_app.test_client()
    _register_and_login(client)
    assert client.get("/dashboard").status_code == 200
    # `!= 302 or True` would pass no matter what. Assert the thing that
    # actually matters: while dark, NOTHING is ever sent to the notice.
    resp = client.post("/analyze", data={})
    assert "/entitlement-required" not in resp.headers.get("Location", "")
