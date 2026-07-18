"""The Practice Club — SSO consumer tests for Analee (sealed module).

Locks the seal: dark by default (404), token validation (401), the per-user
landing (302 → /dashboard), JIT provisioning without duplicates, and the
env-guarded walkthrough seed (no-op without CLUB_WALKTHROUGH_SEED=1).

Tokens are signed with a throwaway RS256 keypair using the SAME scheme the hub
uses (cryptography, PKCS1v15/SHA256), so this validates the real verifier.
"""
import base64
import json
import os
import time
from unittest import mock

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

ISSUER = "the-accountants-hub"


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return key, pub


PRIV, PUB = _keypair()
_OTHER_PRIV, _OTHER_PUB = _keypair()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _token(member_id=1, seat_id=1, audience="analee", issuer=ISSUER,
           exp_offset=300, key=None):
    key = key or PRIV
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "iss": issuer, "aud": audience, "member_id": member_id,
        "seat_id": seat_id, "exp": int(time.time()) + exp_offset,
    }).encode())
    sig = key.sign(f"{header}.{payload}".encode("ascii"),
                   padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{_b64url(sig)}"


@pytest.fixture
def client(canary_app):
    return canary_app.test_client()


_ON = {"CLUB_ENABLED": "True", "HUB_JWT_PUBLIC_KEY": PUB}


def test_dark_by_default_404(client):
    with mock.patch.dict(os.environ, {"CLUB_ENABLED": "", "HUB_JWT_PUBLIC_KEY": PUB}):
        assert client.get(f"/sso/enter/?token={_token()}").status_code == 404


def test_enabled_without_pubkey_stays_dark(client):
    with mock.patch.dict(os.environ, {"CLUB_ENABLED": "True", "HUB_JWT_PUBLIC_KEY": ""}):
        assert client.get(f"/sso/enter/?token={_token()}").status_code == 404


def test_invalid_token_401(client):
    with mock.patch.dict(os.environ, _ON):
        assert client.get("/sso/enter/?token=garbage").status_code == 401


def test_wrong_signer_401(client):
    with mock.patch.dict(os.environ, _ON):
        tok = _token(key=_OTHER_PRIV)
        assert client.get(f"/sso/enter/?token={tok}").status_code == 401


def test_wrong_audience_401(client):
    with mock.patch.dict(os.environ, _ON):
        tok = _token(audience="booksxpert")
        assert client.get(f"/sso/enter/?token={tok}").status_code == 401


def test_member_lands_in_own_workspace(canary_app, client):
    with mock.patch.dict(os.environ, _ON):
        resp = client.get(f"/sso/enter/?token={_token(member_id=7, seat_id=3)}")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard")
    with client.session_transaction() as sess:
        assert sess["club_session"] is True
        assert sess["club_member_id"] == "7"
    with canary_app.app_context():
        from club_sso.models import ClubMemberLink
        from models import User
        user = User.query.filter_by(
            email="club+7.3@sso.theaccountants.local").one()
        link = ClubMemberLink.query.filter_by(
            hub_member_id="7", seat_id="3").one()
        assert link.user_id == user.id


def test_second_entry_reuses_user(canary_app, client):
    with mock.patch.dict(os.environ, _ON):
        assert client.get(f"/sso/enter/?token={_token(member_id=7, seat_id=3)}").status_code == 302
        assert client.get(f"/sso/enter/?token={_token(member_id=7, seat_id=3)}").status_code == 302
    with canary_app.app_context():
        from club_sso.models import ClubMemberLink
        from models import User
        assert User.query.filter_by(
            email="club+7.3@sso.theaccountants.local").count() == 1
        assert ClubMemberLink.query.filter_by(
            hub_member_id="7", seat_id="3").count() == 1


def test_seed_noops_without_walkthrough_flag(canary_app):
    with canary_app.app_context():
        from club_sso.seed import DEMO_EMAIL, run_seed
        from models import User
        with mock.patch.dict(os.environ, {"CLUB_WALKTHROUGH_SEED": ""}):
            run_seed()
        assert User.query.filter_by(email=DEMO_EMAIL).first() is None


def test_seed_creates_demo_member_and_entry_lands(canary_app, client):
    with canary_app.app_context():
        from club_sso.seed import run_seed
        with mock.patch.dict(os.environ, {"CLUB_WALKTHROUGH_SEED": "1"}):
            run_seed()
    with mock.patch.dict(os.environ, _ON):
        resp = client.get(
            f"/sso/enter/?token={_token(member_id=900001, seat_id=900001)}")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard")


# --- jwt_util hardening backport (2026-07-18) -------------------------------
# Two fixes present in the hub's own hub/jwt_util.py (hardened over time) had
# never been backported to this consumer copy: (1) normalize_pem repairs a PEM
# mangled by a host's env-var UI (escaped or flattened newlines); (2) header/
# payload must be JSON *objects* — a non-dict JSON value must raise JWTError
# cleanly rather than surface as an uncaught AttributeError from `.get()`.

def test_normalize_pem_repairs_escaped_newlines():
    from club_sso.jwt_util import normalize_pem

    body = "".join(f"{i%10}" for i in range(200))  # arbitrary base64-ish body
    valid_pem = (
        f"-----BEGIN PUBLIC KEY-----\n"
        + "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
        + "\n-----END PUBLIC KEY-----\n"
    )
    mangled = valid_pem.replace("\n", "\\n")
    assert mangled != valid_pem  # sanity: the fixture actually mangled it

    assert normalize_pem(mangled) == valid_pem.encode("utf-8")


def test_normalize_pem_leaves_valid_pem_unchanged():
    from club_sso.jwt_util import normalize_pem

    assert normalize_pem(PUB) == PUB.encode("utf-8")


def _b64url_json(value) -> str:
    return _b64url(json.dumps(value).encode())


def test_verify_rejects_non_dict_header():
    from club_sso.jwt_util import JWTError, verify_rs256

    # header segment decodes to a JSON *list*, not an object — must not raise
    # an uncaught AttributeError from header.get("alg").
    header = _b64url_json(["RS256"])
    payload = _b64url_json({"iss": ISSUER, "aud": "analee", "member_id": 1,
                             "seat_id": 1, "exp": int(time.time()) + 300})
    sig = PRIV.sign(f"{header}.{payload}".encode("ascii"),
                     padding.PKCS1v15(), hashes.SHA256())
    token = f"{header}.{payload}.{_b64url(sig)}"

    with pytest.raises(JWTError, match="malformed header"):
        verify_rs256(token, PUB, audience="analee", issuer=ISSUER)


def test_verify_rejects_non_dict_payload():
    from club_sso.jwt_util import JWTError, verify_rs256

    # payload segment decodes to a JSON number, not an object — must not raise
    # an uncaught AttributeError from payload.get("exp").
    header = _b64url_json({"alg": "RS256", "typ": "JWT"})
    payload = _b64url_json(42)
    sig = PRIV.sign(f"{header}.{payload}".encode("ascii"),
                     padding.PKCS1v15(), hashes.SHA256())
    token = f"{header}.{payload}.{_b64url(sig)}"

    with pytest.raises(JWTError, match="malformed payload"):
        verify_rs256(token, PUB, audience="analee", issuer=ISSUER)
