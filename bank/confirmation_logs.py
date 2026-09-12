from datetime import datetime, timezone

from .db import _db


def get_confirmation_logs_collection():
    return _db()["transfer_confirmation_logs"]


async def log_confirmation_attempt(
    transfer_id: str,
    call_sid: str,
    recording_url: str,
    transcript: str,
    phrase_match: bool,
    is_synthetic_voice: bool | None,
    voice_confidence: float | None,
    decision: str,
    reason: str | None = None,
) -> None:
    await get_confirmation_logs_collection().insert_one({
        "transfer_id": transfer_id,
        "call_sid": call_sid,
        "recording_url": recording_url,
        "transcript": transcript,
        "phrase_match": phrase_match,
        "is_synthetic_voice": is_synthetic_voice,
        "voice_confidence": voice_confidence,
        "decision": decision,
        "reason": reason,
        "created_at": datetime.now(timezone.utc),
    })
