from .db import get_customers_collection
from .phone import normalize_phone


async def find_by_phone(phone_number: str) -> dict | None:
    if not phone_number:
        return None
    return await get_customers_collection().find_one({"phone_number": normalize_phone(phone_number)})


async def find_by_card_last4(card_last4: str) -> dict | None:
    if not card_last4:
        return None
    return await get_customers_collection().find_one({"card_last4": card_last4})
