"""Signed tokens for no-login client explanation (ERF) links."""
from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

CLIENT_EXPLAIN_SALT = 'analee-client-explain-erf'
DEFAULT_MAX_AGE_SECONDS = 30 * 86_400  # 30 days


def _serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt=CLIENT_EXPLAIN_SALT)


def create_client_explain_token(
    file_id: int,
    user_id: int,
    *,
    secret_key: str,
) -> str:
    return _serializer(secret_key).dumps({'file_id': file_id, 'user_id': user_id})


def verify_client_explain_token(
    token: str,
    *,
    secret_key: str,
    max_age: int = DEFAULT_MAX_AGE_SECONDS,
) -> tuple[int, int]:
    data = _serializer(secret_key).loads(token, max_age=max_age)
    file_id = data.get('file_id')
    user_id = data.get('user_id')
    if not isinstance(file_id, int) or not isinstance(user_id, int):
        raise BadSignature('Invalid client explain token payload.')
    return file_id, user_id
