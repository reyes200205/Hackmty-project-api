import os

from customers.db import get_client


def _db():
    db_name = os.environ.get("MONGODB_DB_NAME", "altur_bank")
    return get_client()[db_name]


def get_accounts_collection():
    return _db()["bank_accounts"]


def get_transactions_collection():
    return _db()["transactions"]


def get_beneficiaries_collection():
    return _db()["beneficiaries"]


def get_transfers_collection():
    return _db()["transfers"]
