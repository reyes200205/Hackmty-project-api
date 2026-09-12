"""Siembra clientes de ejemplo en Mongo. Correr con:
    python -m customers.seed
Es idempotente: usa upsert por phone_number, se puede correr varias veces.
"""
import asyncio

from .auth import hash_password
from .db import get_customers_collection
from .phone import normalize_phone

DEMO_CUSTOMERS = [
    {
        "full_name": "María Fernanda López",
        "email": "maria.lopez@example.com",
        "password": "demo1234",
        "phone_number": "+19151234567",
        "card_last4": "4242",
    },
    {
        "full_name": "Carlos Alberto Reyes",
        "email": "carlos.reyes@example.com",
        "password": "demo1234",
        "phone_number": "+19157654321",
        "card_last4": "1881",
    },
]


async def seed() -> None:
    collection = get_customers_collection()
    await collection.create_index("phone_number", unique=True)

    for customer in DEMO_CUSTOMERS:
        doc = {
            "full_name": customer["full_name"],
            "email": customer["email"],
            "password_hash": hash_password(customer["password"]),
            "phone_number": normalize_phone(customer["phone_number"]),
            "card_last4": customer["card_last4"],
        }
        await collection.update_one(
            {"phone_number": doc["phone_number"]},
            {"$set": doc},
            upsert=True,
        )
        print(f"OK: {doc['full_name']} ({doc['phone_number']})")


if __name__ == "__main__":
    asyncio.run(seed())
