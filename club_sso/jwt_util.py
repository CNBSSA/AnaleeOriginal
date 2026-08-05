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
`iss`/`aud` are checked. Header and payload are also checked to be JSON
*objects* (not e.g. a bare string/number/list) before any `.get()` is called on
them, so a token carrying a malformed header/payload segment cleanly raises
`JWTError` instead of an uncaught `AttributeError` (backported from the hub's
`hub/jwt_util.py` hardening — see CLAUDE.md 2026-07-18 security-fix backport).
"""
from __future__ import annotations

import base64
import json
import re
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class JWTError(Exception):
    """Any malformed / unverifiable / expired token."""


def _b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def normalize_pem(pem) -> bytes:
    """Repair the two ways a host's env-var UI mangles a multi-line key on paste
    (Railway/Heroku style): newlines escaped as literal ``\\n``/``\\r\\n``, or
    every newline flattened to whitespace. ``cryptography.load_pem_*`` rejects
    both with a ValueError that surfaces as a 500. Rebuild the PEM from its
    BEGIN/END markers and re-wrap the base64 body at 64 chars. A PEM that is
    already valid round-trips byte-for-byte. (Backported from the hub's
    `hub/jwt_util.py` — see CLAUDE.md 2026-07-18 security-fix backport.)"""
    if isinstance(pem, bytes):
        pem = pem.decode("utf-8", "ignore")
    s = (pem or "").strip()
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    m = re.search(r"-----BEGIN ([A-Z0-9 ]+)-----(.*?)-----END \1-----", s, re.DOTALL)
    if not m:
        return s.encode("utf-8")
    label, body = m.group(1).strip(), m.group(2)
    b64 = "".join(body.split())
    wrapped = "\n".join(b64[i:i + 64] for i in range(0, len(b64), 64))
    rebuilt = f"-----BEGIN {label}-----\n{wrapped}\n-----END {label}-----\n"
    return rebuilt.encode("utf-8")


def _as_bytes(pem) -> bytes:
    return normalize_pem(pem)


def split_pem_blocks(raw) -> list:
    """Split a blob that may hold several concatenated PEM blocks into a list of
    normalized PEM byte-strings — used for zero-downtime key rotation, where the
    hub's new and previous public keys are supplied together during the overlap.

    Repairs escaped ``\\n``/``\\r\\n`` first (the same host env-var mangling
    ``normalize_pem`` handles), then finds each ``BEGIN…END`` block (its END
    label must match its BEGIN label) and normalizes each via ``normalize_pem``.
    A blob holding a single key returns a 1-element list; empty/whitespace
    returns ``[]``."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    s = (raw or "").strip()
    if not s:
        return []
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    pattern = re.compile(
        r"-----BEGIN ([A-Z0-9 ]+)-----.*?-----END \1-----", re.DOTALL)
    return [normalize_pem(m.group(0)) for m in pattern.finditer(s)]


def _any_key_verifies(pems, signature: bytes, signing_input: bytes) -> bool:
    """True iff at least one candidate PEM verifies the signature. A key that
    fails to load is simply skipped (a rotation list may briefly carry a
    malformed block); the caller raises the SAME ``JWTError`` when none match."""
    for pem in pems:
        try:
            key = serialization.load_pem_public_key(_as_bytes(pem))
            key.verify(signature, signing_input,
                       padding.PKCS1v15(), hashes.SHA256())
            return True
        except (InvalidSignature, ValueError, TypeError):
            continue
    return False


def verify_rs256(token: str, public_key_pem, *, audience=None, issuer=None,
                 leeway: int = 0) -> dict:
    """Verify signature + standard claims; return the payload or raise JWTError.

    ``public_key_pem`` is either a single PEM (``str``/``bytes``) or a
    ``list``/``tuple`` of PEMs. With a list the signature is accepted if ANY
    candidate key verifies it (zero-downtime rotation); the single-PEM path is
    byte-identical to before this change. Algorithm pinning, the header/payload
    object guards, verify-before-trust ordering, and the exp/nbf/iss/aud checks
    are identical in both cases."""
    try:
        h_b64, p_b64, s_b64 = token.split(".")
    except (ValueError, AttributeError):
        raise JWTError("malformed token")

    try:
        header = json.loads(_b64url_decode(h_b64))
    except (ValueError, json.JSONDecodeError):
        raise JWTError("malformed header")
    if not isinstance(header, dict):
        raise JWTError("malformed header")
    if header.get("alg") != "RS256":  # pin the algorithm — no 'none'/HS256 confusion
        raise JWTError("unexpected alg")

    if isinstance(public_key_pem, (list, tuple)):
        # Multi-key path (zero-downtime rotation): accept if ANY trusted key
        # verifies the signature; else raise the same JWTError("bad signature").
        try:
            sig = _b64url_decode(s_b64)
        except (ValueError, TypeError):
            raise JWTError("bad signature")
        if not _any_key_verifies(
                public_key_pem, sig, f"{h_b64}.{p_b64}".encode("ascii")):
            raise JWTError("bad signature")
    else:
        # Single-PEM path — byte-identical to the original single-key verifier.
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
    if not isinstance(payload, dict):
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
