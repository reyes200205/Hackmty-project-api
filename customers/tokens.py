import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

load_dotenv()

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 12


def _get_secret() -> str:
    return os.environ["JWT_SECRET"]


def create_access_token(email: str, full_name: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": email,
        "name": full_name,
        "iat": now,
        "exp": now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Lanza jwt.PyJWTError (o subclases) si el token es invalido o expiro."""
    return jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
