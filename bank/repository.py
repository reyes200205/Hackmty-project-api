import secrets
import uuid
from datetime import datetime, timezone

from customers.db import get_client

from .db import (
    get_accounts_collection,
    get_beneficiaries_collection,
    get_transactions_collection,
    get_transfers_collection,
)


class InsufficientFundsError(Exception):
    pass


async def get_account_by_email(customer_email: str) -> dict | None:
    return await get_accounts_collection().find_one({"customer_email": customer_email}, {"_id": 0})


async def list_transactions(customer_email: str, limit: int, offset: int) -> tuple[list[dict], int]:
    coll = get_transactions_collection()
    total = await coll.count_documents({"customer_email": customer_email})
    cursor = (
        coll.find({"customer_email": customer_email}, {"_id": 0})
        .sort("date", -1)
        .skip(offset)
        .limit(limit)
    )
    items = await cursor.to_list(length=limit)
    return items, total


async def list_beneficiaries(customer_email: str) -> list[dict]:
    cursor = get_beneficiaries_collection().find({"customer_email": customer_email}, {"_id": 0})
    return await cursor.to_list(length=100)


async def get_beneficiary(customer_email: str, beneficiary_id: str) -> dict | None:
    return await get_beneficiaries_collection().find_one(
        {"customer_email": customer_email, "beneficiary_id": beneficiary_id}, {"_id": 0}
    )


async def create_transfer(
    customer_email: str, beneficiary: dict, amount: float, concept: str, confirmation_phrase: str,
) -> dict:
    now = datetime.now(timezone.utc)
    doc = {
        "transfer_id": uuid.uuid4().hex,
        "customer_email": customer_email,
        "beneficiary_id": beneficiary["beneficiary_id"],
        "beneficiary_name": beneficiary["name"],
        "amount": round(amount, 2),
        "concept": concept,
        "status": "pending",
        "confirmation_phrase": confirmation_phrase,
        "confirmation_token": secrets.token_urlsafe(24),
        "call_sid": None,
        "created_at": now,
        "updated_at": now,
    }
    await get_transfers_collection().insert_one(doc)
    doc.pop("_id", None)
    return doc


async def get_transfer(transfer_id: str) -> dict | None:
    return await get_transfers_collection().find_one({"transfer_id": transfer_id}, {"_id": 0})


async def get_transfer_owned_by(customer_email: str, transfer_id: str) -> dict | None:
    return await get_transfers_collection().find_one(
        {"transfer_id": transfer_id, "customer_email": customer_email}, {"_id": 0}
    )


async def list_transfers(customer_email: str, limit: int, offset: int) -> list[dict]:
    cursor = (
        get_transfers_collection()
        .find({"customer_email": customer_email}, {"_id": 0})
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def update_transfer_status(
    transfer_id: str, expected_status: str, new_status: str, extra: dict | None = None,
) -> bool:
    """Actualizacion condicional: solo aplica si el estado actual es el esperado,
    para que un reintento de webhook no pueda doble-procesar la transferencia."""
    fields = {"status": new_status, "updated_at": datetime.now(timezone.utc)}
    if extra:
        fields.update(extra)
    result = await get_transfers_collection().update_one(
        {"transfer_id": transfer_id, "status": expected_status},
        {"$set": fields},
    )
    return result.matched_count == 1


async def atomic_debit_and_log(customer_email: str, amount: float, concept: str, transfer_id: str) -> float:
    """Baja el saldo y registra el movimiento en una transaccion de Mongo
    (Atlas soporta transacciones multi-documento de forma nativa)."""
    client = get_client()
    accounts = get_accounts_collection()
    transactions = get_transactions_collection()

    async with await client.start_session() as session:
        async with session.start_transaction():
            account = await accounts.find_one({"customer_email": customer_email}, session=session)
            if account is None or account["balance"] < amount:
                raise InsufficientFundsError()

            new_balance = round(account["balance"] - amount, 2)
            result = await accounts.update_one(
                {"customer_email": customer_email, "balance": {"$gte": amount}},
                {"$set": {"balance": new_balance, "updated_at": datetime.now(timezone.utc)}},
                session=session,
            )
            if result.matched_count == 0:
                raise InsufficientFundsError()

            await transactions.insert_one({
                "transaction_id": uuid.uuid4().hex,
                "customer_email": customer_email,
                "account_number": account["account_number"],
                "type": "transfer_out",
                "concept": concept,
                "amount": -amount,
                "resulting_balance": new_balance,
                "date": datetime.now(timezone.utc),
                "status": "completed",
                "related_transfer_id": transfer_id,
            }, session=session)

    return new_balance
