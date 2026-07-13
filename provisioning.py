"""Cross-product Analee provisioning (dark, Slice 5b — Festus 2026-07-13).

THE ACCOUNTANTS and Analee are priced together: buying THE ACCOUNTANTS as a
standalone grants **full** Analee access (policy D-F). Analee learns "this
person is an Accountants subscriber" via a **server-to-server** provisioning
call from THE ACCOUNTANTS that activates (or deactivates) the matching Analee
user **by email**.

Ships **DARK** + **fail-closed**:
- ``ANALEE_PROVISIONING_ENABLED`` (default off) gates the endpoint entirely →
  404 while off, so the surface does not exist until Festus turns it on.
- ``ANALEE_PROVISIONING_SECRET`` must be set; a missing secret → 503; a
  wrong/absent bearer token → 401. Compared in constant time.

Reversible: ``entitled=false`` deactivates. **No schema change** — it reuses the
existing ``subscription_status`` (active/inactive), i.e. the same
``is_active`` gate Analee already uses. ``is_deleted`` is never touched.

IDENTITY = **email** (the natural cross-product key). *Flagged for Festus:*
confirm email-matching, or switch to an explicit account-link, before enabling.
"""
import hmac
import logging
import os

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

provisioning = Blueprint("provisioning", __name__)


def enabled():
    """True when the provisioning endpoint is switched on for this environment."""
    return os.environ.get("ANALEE_PROVISIONING_ENABLED", "False") == "True"


def _secret():
    return os.environ.get("ANALEE_PROVISIONING_SECRET", "") or ""


def _bearer_ok(header_value):
    """Constant-time check of an ``Authorization: Bearer <secret>`` header."""
    secret = _secret()
    if not secret:
        return None  # signals "not configured" → 503
    prefix = "Bearer "
    token = header_value[len(prefix):] if (header_value or "").startswith(prefix) else ""
    return hmac.compare_digest(token, secret)


def provision_by_email(email, entitled):
    """Activate/deactivate the Analee user matching ``email``.

    Returns a small result dict. Never raises for a missing user — returns
    ``found=False`` so the caller can log without leaking existence to the wire
    via status codes. Reuses ``subscription_status`` (no migration); leaves
    ``is_deleted`` untouched.
    """
    from models import db, User

    user = User.query.filter(db.func.lower(User.email) == (email or "").strip().lower()).first()
    if user is None:
        logger.info("provisioning: no Analee user for email (entitled=%s)", entitled)
        return {"found": False, "entitled": bool(entitled)}
    user.subscription_status = "active" if entitled else "inactive"
    db.session.commit()
    logger.info("provisioning: user %s set subscription_status=%s",
                user.id, user.subscription_status)
    return {"found": True, "entitled": bool(entitled),
            "subscription_status": user.subscription_status}


@provisioning.route("/api/provisioning/analee", methods=["POST"])
def provision_analee():
    """S2S entry point for THE ACCOUNTANTS to grant/revoke Analee access."""
    if not enabled():
        # Dark: the surface does not exist until explicitly enabled.
        return jsonify({"error": "not found"}), 404
    ok = _bearer_ok(request.headers.get("Authorization", ""))
    if ok is None:
        return jsonify({"error": "provisioning not configured"}), 503
    if not ok:
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    if not email:
        return jsonify({"error": "email is required"}), 400
    entitled = bool(data.get("entitled", True))
    result = provision_by_email(email, entitled)
    return jsonify(result), 200
