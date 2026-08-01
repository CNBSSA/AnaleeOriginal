"""Club hub Phase C — POST /api/practice/club/client (dark, fail-closed)."""

URL = "/api/practice/club/client"
TOKEN = "club-sync-token"


def _enable(monkeypatch):
    monkeypatch.setenv("CLUB_ENABLED", "True")
    monkeypatch.setenv("CLUB_PRACTICE_SYNC_TOKEN", TOKEN)


def test_dark_when_club_disabled(canary_app, monkeypatch):
    monkeypatch.delenv("CLUB_ENABLED", raising=False)
    client = canary_app.test_client()
    r = client.post(URL, json={"member_id": 1, "client": {"id": 2, "name": "Co"}},
                    headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 404


def test_503_when_sync_token_unset(canary_app, monkeypatch):
    monkeypatch.setenv("CLUB_ENABLED", "True")
    monkeypatch.delenv("CLUB_PRACTICE_SYNC_TOKEN", raising=False)
    client = canary_app.test_client()
    r = client.post(URL, json={"member_id": 1, "client": {"id": 2}},
                    headers={"Authorization": "Bearer x"})
    assert r.status_code == 503


def test_401_on_wrong_bearer(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(URL, json={"member_id": 1, "client": {"id": 2}},
                    headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_creates_workspace_and_returns_external_ref(canary_app, monkeypatch):
    _enable(monkeypatch)
    client = canary_app.test_client()
    r = client.post(
        URL,
        json={
            "member_id": 9,
            "client": {"id": 4, "name": "Acme Ltd", "entity_type": "company"},
        },
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["external_ref"] == "club-9-4"
    assert body.get("workspace_user_id")

    r2 = client.post(
        URL,
        json={"member_id": 9, "client": {"id": 4, "name": "Acme Ltd"}},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r2.status_code == 200
    assert r2.get_json()["created"] is False
