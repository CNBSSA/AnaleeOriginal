"""
Tests for the friendly 413 (Request Entity Too Large) handler.

Plain form posts get a flash + redirect; AJAX/JSON requests get a JSON 413.
"""
import pytest

pytest.importorskip("flask")

from flask import Flask, request

# app.py pulls in the full app stack; skip cleanly if those deps are unavailable.
_app_module = pytest.importorskip("app", reason="app module deps unavailable")
_request_entity_too_large = _app_module._request_entity_too_large


def _make_app():
    app = Flask(__name__)
    app.secret_key = "test"
    app.config["MAX_CONTENT_LENGTH"] = 100  # tiny ceiling to trigger 413
    # Stub the fallback redirect target.
    app.add_url_rule("/upload", endpoint="main.upload", view_func=lambda: "upload page")
    app.register_error_handler(413, _request_entity_too_large)

    @app.route("/up", methods=["POST"])
    def up():
        request.get_data()  # accessing the body triggers the ceiling check
        return "ok"

    return app


def test_oversized_form_post_redirects_with_flash():
    app = _make_app()
    client = app.test_client()
    resp = client.post("/up", data=b"x" * 500)
    assert resp.status_code in (301, 302, 303)
    assert "/upload" in resp.headers["Location"]


def test_oversized_ajax_post_returns_json_413():
    app = _make_app()
    client = app.test_client()
    resp = client.post("/up", data=b"x" * 500,
                       headers={"X-Requested-With": "XMLHttpRequest"})
    assert resp.status_code == 413
    body = resp.get_json()
    assert body and body["success"] is False and "too large" in body["error"].lower()


def test_external_referrer_is_not_used_for_redirect():
    app = _make_app()
    client = app.test_client()
    # A spoofed external Referer must NOT be honored (open-redirect guard).
    resp = client.post("/up", data=b"x" * 500,
                       headers={"Referer": "https://evil.example.com/x"})
    assert resp.status_code in (301, 302, 303)
    location = resp.headers["Location"]
    assert "evil.example.com" not in location
    assert "/upload" in location


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
