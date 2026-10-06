"""`?next=` is honoured on login, safely (work-orders #69, R9).

A user bounced to the login page from, say, the Trial Balance page was sent to
/auth/login?next=%2Ftrial-balance by Flask-Login — and after signing in always
landed on /dashboard, because the route read session['next'], which nothing ever
set. The place they were going is now honoured, but ONLY for a same-origin path:
it must start with a single "/" and contain no backslash and no "//". Anything
else (an absolute URL, a protocol-relative one, a backslash trick) goes to the
dashboard, so the login page can never be used as an open redirect.
"""
import pytest

pytest.importorskip("flask_sqlalchemy")

EMAIL = "next@example.com"
PASSWORD = "Sup3rSecret!"


def _register(client):
    return client.post("/auth/register", data={
        "username": "nexter", "email": EMAIL,
        "password": PASSWORD, "confirm_password": PASSWORD,
    }, follow_redirects=True)


def test_the_user_lands_where_they_were_going(canary_app):
    client = canary_app.test_client()
    _register(client)

    bounced = client.get("/trial-balance")
    assert bounced.status_code == 302
    assert "next=%2Ftrial-balance" in bounced.headers["Location"]

    resp = client.post("/auth/login?next=%2Ftrial-balance",
                       data={"email": EMAIL, "password": PASSWORD})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/trial-balance")


def test_the_login_form_carries_next_through_the_post(canary_app):
    page = canary_app.test_client().get("/auth/login?next=%2Ftrial-balance")
    assert (b'action="/auth/login?next=/trial-balance"' in page.data
            or b'action="/auth/login?next=%2Ftrial-balance"' in page.data)


@pytest.mark.parametrize("bad", [
    "https://evil.example/",
    "//evil.example/",
    "/\\evil.example",
    "\\\\evil.example",
    "/trial-balance//x",
    "trial-balance",
    "",
])
def test_an_unsafe_next_falls_back_to_the_dashboard(canary_app, bad):
    client = canary_app.test_client()
    _register(client)
    resp = client.post("/auth/login", query_string={"next": bad},
                       data={"email": EMAIL, "password": PASSWORD})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard"), resp.headers["Location"]


def test_without_next_the_dashboard_is_still_the_default(canary_app):
    client = canary_app.test_client()
    _register(client)
    resp = client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})
    assert resp.status_code == 302 and resp.headers["Location"].endswith("/dashboard")
