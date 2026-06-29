"""The Accountants Club — SSO consumer session binding (P3, §19.9 product side).

When an Analee user arrives via The Accountants Club SSO, their session carries
a hub **refresh token** + the access token's expiry. Once the short-lived
access token is near expiry, a before-request hook asks the hub to refresh. A
rotated pair extends the session; a hub **401** means the session was ended at
the hub, so Analee logs the member out and bounces them to the hub login — a
hub revocation reaches Analee within ~one access-token TTL.

Ships dark: the before-request hook is a no-op until `club_session` is set in
the Flask session, which only happens after an SSO entry point is wired.

Fail-open on transient hub/network errors; fail-closed only on a hub 401. Uses
stdlib urllib (no new dependency); the refresh token lives in the server-side
session, never in a cookie.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from flask import current_app, session

import club_jwt

logger = logging.getLogger(__name__)

AUDIENCE = "analee"
SKEW_SECONDS = 30
CLUB_SESSION_KEYS = (
    "club_session",
    "club_refresh_token",
    "club_access_exp",
    "club_audience",
    "club_member_id",
    "club_branding",
)


def _hub_base() -> str:
    return (current_app.config.get("HUB_API_BASE_URL", "") or "").strip().rstrip("/")


def store_session_tokens(refresh_token, access_exp, audience=AUDIENCE) -> None:
    if refresh_token:
        session["club_refresh_token"] = refresh_token
    if access_exp is not None:
        session["club_access_exp"] = int(access_exp)
    session["club_audience"] = audience or AUDIENCE


def clear_session_tokens() -> None:
    """Remove Club SSO context from the current Flask session."""
    for key in CLUB_SESSION_KEYS:
        session.pop(key, None)


def session_matches_user(user) -> bool:
    """True when the stored Club marker still belongs to the logged-in user.

    Analee does not yet have a ClubMemberLink model (SSO entry point is a
    future phase), so this is a pass-through guard. The actual revocation
    security is provided by refresh() returning 'ended' on a hub 401.
    """
    if not session.get("club_session"):
        return True
    return True


def needs_refresh(now=None) -> bool:
    if not session.get("club_session"):
        return False
    exp = session.get("club_access_exp")
    if not exp:
        return False
    now = int(time.time()) if now is None else int(now)
    return now >= int(exp) - SKEW_SECONDS


def _verified_exp(access_token):
    pub = (current_app.config.get("HUB_JWT_PUBLIC_KEY", "") or "").strip()
    if not pub or not access_token:
        return None
    try:
        payload = club_jwt.verify_rs256(
            access_token, pub, audience=AUDIENCE,
            issuer=current_app.config.get("HUB_ISSUER", "the-accountants-hub"))
    except club_jwt.JWTError:
        return None
    return payload.get("exp")


def refresh() -> str:
    """Returns 'rotated' | 'ended' | 'skip'."""
    base = _hub_base()
    raw = session.get("club_refresh_token")
    audience = session.get("club_audience") or AUDIENCE
    if not base or not raw:
        return "skip"
    body = json.dumps({"refresh_token": raw, "audience": audience}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/sso/refresh/", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return "ended"
        logger.warning("club refresh unexpected status %s", exc.code)
        return "skip"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("club refresh network error: %s", exc)
        return "skip"
    new_exp = _verified_exp(data.get("access_token"))
    if new_exp is None:
        logger.warning("club refresh returned an unverifiable access token")
        return "skip"
    session["club_refresh_token"] = data.get("refresh_token", raw)
    session["club_access_exp"] = int(new_exp)
    return "rotated"


def hub_login_url() -> str:
    return (current_app.config.get("HUB_LOGIN_URL", "") or _hub_base() or "/").strip()
