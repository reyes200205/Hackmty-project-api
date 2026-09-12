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


async def update_call_verdict(
    call_sid: str,
    is_synthetic: bool,
    confidence: float,
    reason: str | None = None,
    status: str = "in_progress",
) -> None:
    """Actualiza en tiempo real el veredicto consolidado de la llamada en Mongo."""
    update_fields = {
        "is_synthetic": is_synthetic,
        "confidence": round(float(confidence), 4),
        "status": status,
        "last_verdict_at": datetime.now(timezone.utc),
    }
    if reason:
        update_fields["verdict_reason"] = reason

    await get_call_logs_collection().update_one(
        {"call_sid": call_sid},
        {"$set": update_fields},
        upsert=True,
    )


async def finalize_call(
    call_sid: str,
    is_synthetic: bool,
    confidence: float,
    reason: str | None = None,
    status: str = "completed",
) -> None:
    """Cierra la llamada registrando el veredicto final y fecha de terminación."""
    # Preservar status de detección si ya fue marcado
    existing = await get_call_logs_collection().find_one({"call_sid": call_sid}, {"status": 1})
    if existing and existing.get("status") == "terminated_ai_detected":
        status = "terminated_ai_detected"

    await get_call_logs_collection().update_one(
        {"call_sid": call_sid},
        {
            "$set": {
                "is_synthetic": is_synthetic,
                "confidence": round(float(confidence), 4),
                "status": status,
                "final_verdict_reason": reason,
                "ended_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )

