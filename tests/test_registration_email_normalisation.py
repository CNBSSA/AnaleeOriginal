"""Registration refuses an address that matches an existing one case-insensitively
after NFKC normalisation (work-orders #69, R5).

Registration lower-cased and stripped the address but did not normalise
Unicode, and compared it exactly against stored rows — so a full-width or
ligature spelling of an existing address, or a legacy mixed-case stored row,
became a second account that admin lists and support could not tell apart.
"""
import pytest

pytest.importorskip("flask_sqlalchemy")

PASSWORD = "Sup3rSecret!"


def _register(client, username, email):
    return client.post("/auth/register", data={
        "username": username, "email": email,
        "password": PASSWORD, "confirm_password": PASSWORD,
    })


def _count(app, needle):
    from models import User
    from sqlalchemy import func
    with app.app_context():
        return User.query.filter(func.lower(User.email).like(f"%{needle}%")).count()


@pytest.mark.parametrize("lookalike", [
    "TEST.Person@Example.com",      # case
    "  test.person@example.com ",   # whitespace
    "ｔest.person@example.com",     # full-width t → NFKC "t"
    "test.person@ｅxample.com",     # full-width e in the domain
])
def test_a_look_alike_of_an_existing_address_is_refused(canary_app, lookalike):
    client = canary_app.test_client()
    first = _register(client, "first", "test.person@example.com")
    assert first.status_code == 302, first.data[:300]

    second = _register(client, "second", lookalike)
    body = second.data.decode(errors="replace")
    assert second.status_code == 200, "the form is shown again, not accepted"
    assert "already" in body.lower(), body[:500]
    assert _count(canary_app, "test.person") == 1


def test_a_legacy_mixed_case_row_is_matched_too(canary_app):
    """Rows stored before addresses were lower-cased must still block a clash."""
    from models import db, User
    with canary_app.app_context():
        legacy = User(username="legacy", email="Legacy.User@Example.com",
                      subscription_status="active")
        legacy.set_password(PASSWORD)
        db.session.add(legacy)
        db.session.commit()

    resp = _register(canary_app.test_client(), "newcomer", "legacy.user@example.com")
    assert resp.status_code == 200
    assert "already" in resp.data.decode(errors="replace").lower()
    assert _count(canary_app, "legacy.user") == 1


def test_the_stored_address_is_the_normalised_one(canary_app):
    from models import User
    _register(canary_app.test_client(), "wide", "Ｗide.Name@Example.com")
    with canary_app.app_context():
        assert User.query.filter_by(email="wide.name@example.com").count() == 1


def test_a_genuinely_different_address_still_registers(canary_app):
    client = canary_app.test_client()
    assert _register(client, "aa", "a.person@example.com").status_code == 302
    assert _register(client, "bb", "b.person@example.com").status_code == 302
