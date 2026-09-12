from datetime import datetime, timezone

from customers.db import get_client
import os


def get_call_logs_collection():
    db_name = os.environ.get("MONGODB_DB_NAME", "altur_bank")
    return get_client()[db_name]["call_logs"]


async def start_call(call_sid: str, stream_sid: str, customer_name: str | None) -> None:
    await get_call_logs_collection().update_one(
        {"call_sid": call_sid},
        {
            "$setOnInsert": {
                "call_sid": call_sid,
                "stream_sid": stream_sid,
                "customer_name": customer_name,
                "started_at": datetime.now(timezone.utc),
            },
            "$set": {"turns": []},
        },
        upsert=True,
    )


async def save_turn(call_sid: str, turn: dict) -> None:
    turn = {**turn, "timestamp": datetime.now(timezone.utc)}
    await get_call_logs_collection().update_one(
        {"call_sid": call_sid},
        {"$push": {"turns": turn}},
        upsert=True,
    )
