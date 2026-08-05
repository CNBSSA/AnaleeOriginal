"""Zero-downtime JWT rotation — consumer-side multi-key verification.

Mirrors the hub-side change in clubhub #141: during a hub RS256 key rotation
overlap, this SSO consumer must accept a token signed by EITHER the new OR the
previous public key. `verify_rs256` now takes a single PEM (unchanged) OR a list
of PEMs (accept if ANY verifies); `split_pem_blocks` splits a concatenated PEM
blob; and `routes._trusted_public_keys` builds the trusted list from the
multi-key env var, falling back to the existing single key.

Tokens are signed with throwaway RS256 keypairs using the SAME scheme the hub
uses (cryptography, PKCS1v15/SHA256), so this validates the real verifier.
"""
import base64
import json
import os
import time
from unittest import mock

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from club_sso import jwt_util
from club_sso.jwt_util import JWTError, split_pem_blocks, verify_rs256
from club_sso.routes import _trusted_public_keys

ISSUER = "the-accountants-hub"
AUD = "analee"


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return key, pub


PRIV_A, PUB_A = _keypair()
PRIV_B, PUB_B = _keypair()
PRIV_C, PUB_C = _keypair()  # a third, never-trusted signer


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _token(key, member_id=1, seat_id=1, audience=AUD, issuer=ISSUER,
           exp_offset=300):
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "iss": issuer, "aud": audience, "member_id": member_id,
        "seat_id": seat_id, "exp": int(time.time()) + exp_offset,
    }).encode())
    sig = key.sign(f"{header}.{payload}".encode("ascii"),
                   padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{_b64url(sig)}"


# --- multi-key verify --------------------------------------------------------

def test_a_signed_verifies_with_list_containing_a():
    payload = verify_rs256(_token(PRIV_A), [PUB_A], audience=AUD, issuer=ISSUER)
    assert payload["member_id"] == 1


def test_a_signed_fails_with_list_containing_only_b():
    try:
        verify_rs256(_token(PRIV_A), [PUB_B], audience=AUD, issuer=ISSUER)
        assert False, "expected JWTError"
    except JWTError as e:
        assert "bad signature" in str(e)


def test_a_signed_verifies_with_both_orders():
    for keys in ([PUB_A, PUB_B], [PUB_B, PUB_A]):
        payload = verify_rs256(_token(PRIV_A), keys, audience=AUD, issuer=ISSUER)
        assert payload["member_id"] == 1


def test_b_signed_verifies_when_b_in_list():
    # The whole point of rotation: a token from the *other* trusted key passes.
    payload = verify_rs256(_token(PRIV_B), [PUB_A, PUB_B], audience=AUD,
                           issuer=ISSUER)
    assert payload["member_id"] == 1


def test_third_untrusted_key_still_fails():
    try:
        verify_rs256(_token(PRIV_C), [PUB_A, PUB_B], audience=AUD, issuer=ISSUER)
        assert False, "expected JWTError"
    except JWTError as e:
        assert "bad signature" in str(e)


def test_claims_still_checked_on_list_path():
    # Correctly signed by A but wrong audience — claim checks must still bite.
    tok = _token(PRIV_A, audience="booksxpert")
    try:
        verify_rs256(tok, [PUB_A, PUB_B], audience=AUD, issuer=ISSUER)
        assert False, "expected JWTError"
    except JWTError as e:
        assert "bad audience" in str(e)


def test_single_pem_str_path_unchanged():
    # The original single-PEM signature stays byte-identical in behaviour.
    assert verify_rs256(_token(PRIV_A), PUB_A, audience=AUD,
                        issuer=ISSUER)["member_id"] == 1
    try:
        verify_rs256(_token(PRIV_A), PUB_B, audience=AUD, issuer=ISSUER)
        assert False, "expected JWTError"
    except JWTError as e:
        assert "bad signature" in str(e)


# --- split_pem_blocks --------------------------------------------------------

def test_split_pem_blocks_two_then_one():
    two = PUB_A + "\n" + PUB_B
    blocks = split_pem_blocks(two)
    assert len(blocks) == 2
    assert blocks[0] == jwt_util.normalize_pem(PUB_A)
    assert blocks[1] == jwt_util.normalize_pem(PUB_B)

    one = split_pem_blocks(PUB_A)
    assert len(one) == 1
    assert one[0] == jwt_util.normalize_pem(PUB_A)


def test_split_pem_blocks_empty_returns_empty_list():
    assert split_pem_blocks("") == []
    assert split_pem_blocks(None) == []


def test_split_pem_blocks_repairs_escaped_newlines():
    two = (PUB_A + "\n" + PUB_B).replace("\n", "\\n")
    blocks = split_pem_blocks(two)
    assert len(blocks) == 2
    # A token from either key verifies against the repaired list.
    assert verify_rs256(_token(PRIV_B), blocks, audience=AUD,
                        issuer=ISSUER)["member_id"] == 1


# --- config helper (routes._trusted_public_keys) -----------------------------

def test_helper_prefers_multi_var_when_set():
    two = PUB_A + "\n" + PUB_B
    with mock.patch.dict(os.environ, {"HUB_JWT_PUBLIC_KEYS": two,
                                      "HUB_JWT_PUBLIC_KEY": PUB_C}):
        keys = _trusted_public_keys()
    assert len(keys) == 2
    # The single key is NOT consulted while the multi var is set.
    assert jwt_util.normalize_pem(PUB_C) not in keys


def test_helper_accepts_club_prefixed_multi_var():
    two = PUB_A + "\n" + PUB_B
    with mock.patch.dict(os.environ, {"CLUB_JWT_PUBLIC_KEYS": two},
                         clear=False):
        os.environ.pop("HUB_JWT_PUBLIC_KEYS", None)
        keys = _trusted_public_keys()
    assert len(keys) == 2


def test_helper_falls_back_to_single_key():
    with mock.patch.dict(os.environ, {"HUB_JWT_PUBLIC_KEY": PUB_A}):
        os.environ.pop("HUB_JWT_PUBLIC_KEYS", None)
        os.environ.pop("CLUB_JWT_PUBLIC_KEYS", None)
        keys = _trusted_public_keys()
    # Byte-identical single-key set (same `.strip()` the original code applied),
    # dark until the multi var is set.
    assert keys == [PUB_A.strip()]
    # And a token signed by that key verifies through the single-element list.
    assert verify_rs256(_token(PRIV_A), keys, audience=AUD,
                        issuer=ISSUER)["member_id"] == 1


def test_helper_empty_when_nothing_configured():
    with mock.patch.dict(os.environ, {}, clear=False):
        for var in ("HUB_JWT_PUBLIC_KEYS", "CLUB_JWT_PUBLIC_KEYS",
                    "HUB_JWT_PUBLIC_KEY"):
            os.environ.pop(var, None)
        assert _trusted_public_keys() == []
