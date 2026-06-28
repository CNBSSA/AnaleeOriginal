"""Time-limited signed tokens for trial balance transmission."""
from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

TB_SHARE_SALT = 'analee-tb-share'
DEFAULT_MAX_AGE_SECONDS = 86_400  # 24 hours


def _serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt=TB_SHARE_SALT)


def create_share_token(user_id: int, *, secret_key: str) -> str:
    return _serializer(secret_key).dumps({'user_id': user_id})


def verify_share_token(
    token: str,
    *,
    secret_key: str,
    max_age: int = DEFAULT_MAX_AGE_SECONDS,
) -> int:
    """Return ``user_id`` if the token is valid; raise on failure."""
    data = _serializer(secret_key).loads(token, max_age=max_age)
    user_id = data.get('user_id')
    if not isinstance(user_id, int):
        raise BadSignature('Invalid trial balance share token payload.')
    return user_id
