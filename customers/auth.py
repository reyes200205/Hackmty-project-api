import binascii
import hashlib
import os

PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 con salt aleatorio, sin dependencias externas.
    No se usa para autenticar la llamada de voz, solo para guardar la
    cuenta del cliente en Mongo de forma segura."""
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return binascii.hexlify(salt + derived).decode("ascii")


def verify_password(password: str, stored_hash: str) -> bool:
    raw = binascii.unhexlify(stored_hash)
    salt, expected = raw[:16], raw[16:]
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return derived == expected
