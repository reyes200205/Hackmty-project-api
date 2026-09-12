"""Siembra cuentas, beneficiarios y movimientos historicos de ejemplo.
Correr con: python -m bank.seed
Idempotente: usa upsert por las llaves naturales de cada coleccion.
"""
import asyncio
from datetime import datetime, timedelta, timezone

from .db import get_accounts_collection, get_beneficiaries_collection, get_transactions_collection

DEMO_ACCOUNTS = [
    {
        "customer_email": "jorgerenteriareyes4@gmail.com",
        "account_number": "0123456789",
        "account_type": "Cuenta de debito",
        "balance": 23149.50,
        "currency": "MXN",
        "status": "active",
    },
    {
        "customer_email": "dan@example.com",
        "account_number": "9876543210",
        "account_type": "Cuenta de debito",
        "balance": 8300.00,
        "currency": "MXN",
        "status": "active",
    },
]

DEMO_BENEFICIARIES = [
    {
        "customer_email": "jorgerenteriareyes4@gmail.com",
        "beneficiary_id": "benef-jorge-1",
        "name": "Maria Fernanda Lopez",
        "account_number": "0011223344",
        "bank_name": "BBVA",
    },
    {
        "customer_email": "jorgerenteriareyes4@gmail.com",
        "beneficiary_id": "benef-jorge-2",
        "name": "Carlos Alberto Reyes",
        "account_number": "0055667788",
        "bank_name": "Santander",
    },
    {
        "customer_email": "dan@example.com",
        "beneficiary_id": "benef-dan-1",
        "name": "Ana Sofia Torres",
        "account_number": "0099887766",
        "bank_name": "Banorte",
    },
]


async def seed() -> None:
    accounts = get_accounts_collection()
    beneficiaries = get_beneficiaries_collection()
    transactions = get_transactions_collection()

    await accounts.create_index("customer_email", unique=True)
    await accounts.create_index("account_number", unique=True)
    await beneficiaries.create_index([("customer_email", 1), ("beneficiary_id", 1)], unique=True)
    await transactions.create_index("transaction_id", unique=True)

    now = datetime.now(timezone.utc)

    for acc in DEMO_ACCOUNTS:
        doc = {**acc, "opened_at": now - timedelta(days=400), "updated_at": now}
        await accounts.update_one({"customer_email": acc["customer_email"]}, {"$set": doc}, upsert=True)
        print(f"cuenta OK: {acc['customer_email']} -> {acc['account_number']} (${acc['balance']:.2f})")

    for b in DEMO_BENEFICIARIES:
        await beneficiaries.update_one(
            {"customer_email": b["customer_email"], "beneficiary_id": b["beneficiary_id"]},
            {"$set": b},
            upsert=True,
        )
        print(f"beneficiario OK: {b['name']} ({b['customer_email']})")

    historial = [
        {"customer_email": "jorgerenteriareyes4@gmail.com", "account_number": "0123456789",
         "transaction_id": "seed-tx-1", "type": "deposit", "concept": "Deposito de nomina",
         "amount": 12000.0, "resulting_balance": 12000.0, "date": now - timedelta(days=10),
         "status": "completed", "related_transfer_id": None},
        {"customer_email": "jorgerenteriareyes4@gmail.com", "account_number": "0123456789",
         "transaction_id": "seed-tx-2", "type": "purchase", "concept": "Supermercado",
         "amount": -850.5, "resulting_balance": 11149.5, "date": now - timedelta(days=8),
         "status": "completed", "related_transfer_id": None},
        {"customer_email": "jorgerenteriareyes4@gmail.com", "account_number": "0123456789",
         "transaction_id": "seed-tx-3", "type": "deposit", "concept": "Deposito de nomina",
         "amount": 12000.0, "resulting_balance": 23149.5, "date": now - timedelta(days=3),
         "status": "completed", "related_transfer_id": None},
        {"customer_email": "dan@example.com", "account_number": "9876543210",
         "transaction_id": "seed-tx-4", "type": "deposit", "concept": "Deposito inicial",
         "amount": 8300.0, "resulting_balance": 8300.0, "date": now - timedelta(days=15),
         "status": "completed", "related_transfer_id": None},
    ]
    for tx in historial:
        await transactions.update_one({"transaction_id": tx["transaction_id"]}, {"$set": tx}, upsert=True)
    print(f"{len(historial)} movimientos historicos OK")


if __name__ == "__main__":
    asyncio.run(seed())
