"""FileIt credential handling (post-implementation audit). The hub hands this service
the FileIt credential of the practice it is acting for. Four guards:
the hub's ``base_url`` is never trusted (the secret only goes to the configured
FileIt host); an empty token is never cached; a malformed ``expires_in`` is not
a crash; a rotated secret is not served the old token; and a worker thread
forgets the previous request's credential when the next request starts."""
from unittest import mock

import fileit_pull as mod

BASE = "https://fileit.example"


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _cred(secret="s1"):
    return {"client_id": "cid-1", "client_secret": secret, "base_url": "https://evil.example"}


def test_the_secret_only_goes_to_the_configured_fileit_host(monkeypatch):
    monkeypatch.setenv("FILEIT_API_BASE_URL", BASE)
    mod._tokens_by_client.clear()
    mod._ctx.credential = _cred()
    with mock.patch.object(mod.requests, "post", return_value=_Resp({"access_token": "t1", "expires_in": "soon"})) as post:
        assert mod._bearer() == "t1"
    assert post.call_args.args[0].startswith(BASE + "/"), post.call_args
    mod._ctx.credential = None


def test_an_empty_token_is_never_cached_and_a_rotated_secret_is_not_served_the_old_token(monkeypatch):
    monkeypatch.setenv("FILEIT_API_BASE_URL", BASE)
    mod._tokens_by_client.clear()
    mod._ctx.credential = _cred()
    with mock.patch.object(mod.requests, "post", return_value=_Resp({"access_token": ""})) as post:
        mod._bearer()
        mod._bearer()
    assert post.call_count == 2, "an empty token must be asked for again, not served from cache"
    with mock.patch.object(mod.requests, "post", return_value=_Resp({"access_token": "old"})):
        assert mod._bearer() == "old"
    mod._ctx.credential = _cred("s2")
    with mock.patch.object(mod.requests, "post", return_value=_Resp({"access_token": "new"})):
        assert mod._bearer() == "new"
    mod._ctx.credential = None


def test_the_next_request_forgets_the_previous_credential(app):
    mod._ctx.credential = _cred()
    app.test_client().get("/no-such-page-credential-reset")
    assert getattr(mod._ctx, "credential", None) is None
