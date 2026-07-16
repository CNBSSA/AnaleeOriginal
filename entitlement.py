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
flag off the app behaves exactly as before. Turning the flag on gates *new*
access decisions; it does **not** by itself lock out existing users, because
``User.subscription_status`` defaults to ``'active'`` (see ``models.py``) — so
every current user counts as a subscriber until Festus explicitly changes that
default and migrates existing rows. That default-change + data migration is the
separate, **destructive** step and is deliberately NOT taken here.
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


def is_workspace_session():
    """True when THE ACCOUNTANTS dropped the user into a client workspace."""
    return bool(session.get("workspace_session"))


def analee_entitled(user):
    """Single source of truth: may this user use Analee?

    ``True`` for a Practice Club member (session-bound), a client workspace
    session (THE ACCOUNTANTS signed login link), OR an Accountants / Analee
    subscriber. Anonymous / ``None`` users are never entitled.
    """
    if is_club_member():
        return True
    if is_workspace_session():
        return True
    if user is None or not getattr(user, "is_authenticated", False):
        return False
    return is_subscriber(user)
