"""JWT helpers for the small self-contained API authentication layer."""
from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt


def authenticate(username: str, password: str, expected_username: str, expected_password: str) -> bool:
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(password, expected_password)


def create_access_token(subject: str, secret: str, algorithm: str, expires_minutes: int) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "iat": int(now.timestamp()), "exp": int(exp.timestamp()), "type": "access"}
    return jwt.encode(payload, secret, algorithm=algorithm), expires_minutes * 60


def decode_access_token(token: str, secret: str, algorithm: str) -> dict[str, Any]:
    payload = jwt.decode(token, secret, algorithms=[algorithm])
    if payload.get("type") != "access" or not payload.get("sub"):
        raise jwt.InvalidTokenError("invalid token payload")
    return payload
