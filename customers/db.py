import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.environ["MONGODB_URI"]
        _client = AsyncIOMotorClient(uri)
    return _client


def get_customers_collection():
    db_name = os.environ.get("MONGODB_DB_NAME", "altur_bank")
    return get_client()[db_name]["customers"]
