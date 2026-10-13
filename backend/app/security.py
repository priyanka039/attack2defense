"""Password hashing and signed access tokens."""

from __future__ import annotations

from datetime import timedelta

import bcrypt
import jwt

from backend.app.db import utcnow

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_token(user_id: int, secret: str, minutes: int) -> tuple[str, int]:
    expires_in = minutes * 60
    payload = {"sub": str(user_id), "exp": utcnow() + timedelta(seconds=expires_in)}
    return jwt.encode(payload, secret, algorithm=ALGORITHM), expires_in


def read_token(token: str, secret: str) -> int | None:
    """Return the user id, or None when the token is invalid or expired."""

    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
