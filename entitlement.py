"""Analee entitlement gate (Slice 5, Festus 2026-07-13).

Festus's rule: *"to get Analee, you must be a subscriber of either The
Accountants or a member of the Practice Club."* This module is the single
source of truth for that decision.

Entitled = Practice Club member (arrived via Club SSO — session-bound) **OR**
an Accountants / Analee subscriber (local ``subscription_status`` in
``active`` / ``pending``).

Policy (Festus 2026-07-13): entitlement is **binary — full Analee access, no
limited tier**. Both a Club member and a user who buys THE ACCOUNTANTS as a
standalone get the *same full* access. (The mechanism by which "bought THE
ACCOUNTANTS standalone" reaches Analee — hub JWT claim / provisioning link /
API — is still open; today "subscriber" here means the local
``subscription_status``, and the Club path already flows in via SSO.)

Ships **DARK** behind ``ANALEE_ENTITLEMENT_ENFORCED`` (default off): with the
flag off the app behaves exactly as before.

**New-signup gating (G10/B-A2, done the safe way):** ``User.subscription_status``
now defaults to ``'login_only'`` (see ``models.py``) — a value that CAN log in
(``is_active`` includes it) but is **not** a subscriber, so a casual new sign-up
is not Analee-entitled once the flag is on. This deliberately does NOT reuse
``'inactive'`` (which would also block login). Existing ``'active'`` rows are
**grandfathered** — the default is a Python-side ORM default applied only to new
inserts, so no data migration touches current users. To actually entitle a user:
they subscribe (Accountants → provisioning sets ``'active'``) or join the Club
(session marker). Turning the flag on then gates non-entitled users to the
friendly ``/entitlement-required`` page.
"""
import os

from flask import session


def enforcement_enabled():
    """True when the entitlement gate is switched on for this environment."""
    return os.environ.get("ANALEE_ENTITLEMENT_ENFORCED", "False") == "True"


def is_club_member():
    """True when the current session arrived via Practice Club SSO."""
    return bool(session.get("club_session") or session.get("club_member_id"))


def is_subscriber(user):
    """True when ``user`` is an Accountants / Analee subscriber (local signal)."""
    return getattr(user, "subscription_status", None) in ("active", "pending")


def analee_entitled(user):
    """Single source of truth: may this user use Analee?

    ``True`` for a Practice Club member (session-bound) OR an Accountants /
    Analee subscriber. Anonymous / ``None`` users are never entitled.
    """
    if is_club_member():
        return True
    if user is None or not getattr(user, "is_authenticated", False):
        return False
    return is_subscriber(user)


# --- Read-only access for a non-entitled user (Festus, 2026-09-19) -----------
#
# The rule Festus set after TrustEasyGo hid a firm's own bank statements behind
# a setup gate: "A normal application doesn't behave that way. Who will buy an
# application like [that]?"
#
#     A gate may stop you WRITING new data. It must never stop you READING
#     your own.
#
# Before this, the gate redirected EVERY route for a non-entitled user. It is
# dark, so nobody has been hurt -- but the day it is switched on, a member whose
# Club membership or subscription lapses would lose sight of statements and
# trial balances they produced and paid for. That is the landmine this closes.
#
# The commercial rule still bites, and that is deliberate: a lapsed user can
# LOOK at their books and take their data with them, but cannot do any more
# work -- no uploads, no analysis, no categorising, no posting. Everything that
# creates or changes anything is a POST/PUT/PATCH/DELETE and stays blocked.
#
# HTTP method is the discriminator rather than endpoint names, because a name
# heuristic guesses and a method does not. The two GET endpoints that are not
# really reads are named explicitly below rather than left to the rule.

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: GET endpoints that are NOT reads -- they mint access or spend real money.
NON_READ_GET_ENDPOINTS = frozenset({
    # Mints a signed URL that lets a THIRD PARTY into this file without a
    # login. Handing out access is not reading your own work.
    "main.analyze_client_link",
    # Calls the AI on every request, so it costs money per click.
    "main.icountant_transaction_insights",
})


def read_only_allowed(method, endpoint):
    """May a NON-entitled user still be served this request?

    True for a genuine read; False for anything that writes, spends or grants.
    """
    if (method or "").upper() not in SAFE_METHODS:
        return False
    return (endpoint or "") not in NON_READ_GET_ENDPOINTS
