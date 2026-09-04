"""Signed tokens for password reset.

Same house pattern as ``client_explain_tokens`` / ``reports.tb_share_tokens`` /
``provisioning`` — ``itsdangerous`` off the app secret with a purpose salt.

Why this module exists (QA audit, 2026-09-04): the previous reset route took a
``<token>`` in the URL and **never read it**. It looked up a user by an email
posted in the form and reset that account's password. It only ever returned 500
because ``ResetPasswordForm`` has no email field — so it was broken rather than
exploitable. Completing that form would have turned it into unauthenticated
account takeover of any user, including an admin. The token, not the form, must
say whose password is being reset.

Two properties matter and both are tested:

* **User-bound.** The payload carries the user id. A token issued for one
  account can never reset another.
* **Effectively single-use, with no schema change.** The payload also carries a
  prefix of the account's password hash at issue time. Resetting the password
  changes the hash, so the token stops verifying the moment it is used — and any
  older outstanding token dies at the same time.
"""
from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

PASSWORD_RESET_SALT = 'analee-password-reset'
DEFAULT_MAX_AGE_SECONDS = 3600  # one hour
_HASH_BINDING_LEN = 24


def _serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt=PASSWORD_RESET_SALT)


def _binding(password_hash) -> str:
    """A stable fingerprint of the CURRENT credential, not the credential."""
    return (password_hash or '')[:_HASH_BINDING_LEN]


def create_password_reset_token(user, *, secret_key: str) -> str:
    return _serializer(secret_key).dumps({
        'user_id': int(user.id),
        'pw': _binding(user.password_hash),
    })


def verify_password_reset_token(
    token: str,
    *,
    secret_key: str,
    max_age: int = DEFAULT_MAX_AGE_SECONDS,
) -> int:
    """Return the user id this token authorises, or raise.

    Raises ``SignatureExpired`` / ``BadSignature`` — the caller treats both the
    same way, so an attacker learns nothing from which one came back.
    """
    data = _serializer(secret_key).loads(token, max_age=max_age)
    if not isinstance(data, dict):
        raise BadSignature('Invalid password reset token payload.')
    user_id = data.get('user_id')
    if not isinstance(user_id, int):
        raise BadSignature('Invalid password reset token payload.')
    return user_id


def token_matches_current_password(token: str, user, *, secret_key: str,
                                   max_age: int = DEFAULT_MAX_AGE_SECONDS) -> bool:
    """False once the password has changed — i.e. once the token has been used."""
    data = _serializer(secret_key).loads(token, max_age=max_age)
    return isinstance(data, dict) and data.get('pw') == _binding(user.password_hash)


__all__ = [
    'PASSWORD_RESET_SALT',
    'DEFAULT_MAX_AGE_SECONDS',
    'create_password_reset_token',
    'verify_password_reset_token',
    'token_matches_current_password',
    'BadSignature',
    'SignatureExpired',
]
