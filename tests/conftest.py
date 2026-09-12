import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Aislar los tests de la base de datos real: se usa el mismo cluster de Atlas
# pero una base distinta, para no ensuciar los datos sembrados a mano.
os.environ["MONGODB_DB_NAME"] = "altur_bank_test"

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import main as main_module
from bank.db import get_accounts_collection, get_beneficiaries_collection, get_transfers_collection
from customers.auth import hash_password
from customers.db import get_customers_collection

TEST_EMAIL = "pytest.customer@example.com"
TEST_PASSWORD = "pytest1234"
TEST_PHONE = "+15550001111"

OTHER_EMAIL = "pytest.other@example.com"
OTHER_PASSWORD = "pytest1234"


@pytest_asyncio.fixture
async def seeded_customer():
    now = datetime.now(timezone.utc)
    await get_customers_collection().update_one(
        {"email": TEST_EMAIL},
        {"$set": {
            "full_name": "Pytest Customer",
            "email": TEST_EMAIL,
            "password_hash": hash_password(TEST_PASSWORD),
            "phone_number": TEST_PHONE,
            "card_last4": "0000",
        }},
        upsert=True,
    )
    await get_accounts_collection().update_one(
        {"customer_email": TEST_EMAIL},
        {"$set": {
            "customer_email": TEST_EMAIL,
            "account_number": "TEST0000001",
            "account_type": "Cuenta de prueba",
            "balance": 1000.0,
            "currency": "MXN",
            "status": "active",
            "opened_at": now,
            "updated_at": now,
        }},
        upsert=True,
    )
    await get_beneficiaries_collection().update_one(
        {"customer_email": TEST_EMAIL, "beneficiary_id": "pytest-benef-1"},
        {"$set": {
            "customer_email": TEST_EMAIL,
            "beneficiary_id": "pytest-benef-1",
            "name": "Beneficiario de Prueba",
            "account_number": "TEST9999999",
            "bank_name": "Banco de Pruebas",
        }},
        upsert=True,
    )
    # cada test arranca con saldo limpio y sin transferencias previas
    await get_accounts_collection().update_one({"customer_email": TEST_EMAIL}, {"$set": {"balance": 1000.0}})
    await get_transfers_collection().delete_many({"customer_email": TEST_EMAIL})
    return {"email": TEST_EMAIL, "password": TEST_PASSWORD, "phone": TEST_PHONE}


@pytest_asyncio.fixture
async def other_customer():
    await get_customers_collection().update_one(
        {"email": OTHER_EMAIL},
        {"$set": {
            "full_name": "Pytest Other",
            "email": OTHER_EMAIL,
            "password_hash": hash_password(OTHER_PASSWORD),
            "phone_number": "+15550002222",
            "card_last4": "1111",
        }},
        upsert=True,
    )
    return {"email": OTHER_EMAIL, "password": OTHER_PASSWORD}


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=main_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client, seeded_customer):
    resp = await client.post("/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
