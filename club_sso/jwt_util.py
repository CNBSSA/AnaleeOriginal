"""RS256 JWT verification for The Practice Club SSO consumer path.

Analee is a *consumer* of the hub (`clubhub`): the hub holds the RS256 private
key and signs short-lived access tokens; Analee validates them here with the
hub's **public** key only. Mirrors the hub's `hub/jwt_util.py` (and
hrxpertsFestus's `routes/club_jwt.py` / TrustEasyGo's `accounts/club_jwt.py`)
verbatim so the implementations never drift — and adds nothing to the auth path
beyond `cryptography` (already pinned in requirements). No PyJWT, no OAuth
server.

Security: `alg` is pinned to RS256 (so `alg=none`/HS256-confusion is rejected),
the signature is verified **before** the payload is trusted, then `exp`/`nbf`/
`iss`/`aud` are checked.
"""
from __future__ import annotations

import base64
import json
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class JWTError(Exception):
    """Any malformed / unverifiable / expired token."""


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def _as_bytes(pem) -> bytes:
    return pem.encode("utf-8") if isinstance(pem, str) else pem


def verify_rs256(token: str, public_key_pem, *, audience=None, issuer=None,
                 leeway: int = 0) -> dict:
    """Verify signature + standard claims; return the payload or raise JWTError."""
    try:
        h_b64, p_b64, s_b64 = token.split(".")
    except (ValueError, AttributeError):
        raise JWTError("malformed token")

    try:
        header = json.loads(_b64url_decode(h_b64))
    except (ValueError, json.JSONDecodeError):
        raise JWTError("malformed header")
    if header.get("alg") != "RS256":  # pin the algorithm — no 'none'/HS256 confusion
        raise JWTError("unexpected alg")

    try:
        key = serialization.load_pem_public_key(_as_bytes(public_key_pem))
    except (ValueError, TypeError):
        raise JWTError("bad public key")
    try:
        key.verify(_b64url_decode(s_b64), f"{h_b64}.{p_b64}".encode("ascii"),
                   padding.PKCS1v15(), hashes.SHA256())
    except (InvalidSignature, ValueError):
        raise JWTError("bad signature")

    try:
        payload = json.loads(_b64url_decode(p_b64))
    except (ValueError, json.JSONDecodeError):
        raise JWTError("malformed payload")

    now = int(time.time())
    exp = payload.get("exp")
    if exp is not None and now > int(exp) + leeway:
        raise JWTError("token expired")
    nbf = payload.get("nbf")
    if nbf is not None and now + leeway < int(nbf):
        raise JWTError("token not yet valid")
    if issuer is not None and payload.get("iss") != issuer:
        raise JWTError("bad issuer")
    if audience is not None and payload.get("aud") != audience:
        raise JWTError("bad audience")
    return payload
