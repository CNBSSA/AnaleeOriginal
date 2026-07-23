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
