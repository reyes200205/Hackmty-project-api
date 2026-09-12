import jwt
from fastapi import Header, HTTPException

from .lookup import find_by_email
from .tokens import decode_access_token


async def get_current_customer(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Falta el header Authorization: Bearer <token>")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="El token expiro, inicia sesion de nuevo")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalido")

    customer = await find_by_email(payload.get("sub", ""))
    if not customer:
        raise HTTPException(status_code=401, detail="El cliente de este token ya no existe")
    return customer
