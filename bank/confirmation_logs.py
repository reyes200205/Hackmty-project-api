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
    liveness_match: bool | None = None,
    voice_reasoning: str | None = None,
    response_latency_s: float | None = None,
    too_slow: bool | None = None,
) -> None:
    """Guarda el intento completo, incluyendo el desglose de CADA señal (no solo
    el veredicto final), para poder diagnosticar rechazos sin adivinar -- como
    el falso positivo real que se encontro el 13-sep-2026 (digital_silence_ratio
    disparado por audio genuino de Twilio, ver nota en recording_service.py)."""
    await get_confirmation_logs_collection().insert_one({
        "transfer_id": transfer_id,
        "call_sid": call_sid,
        "recording_url": recording_url,
        "transcript": transcript,
        "phrase_match": phrase_match,
        "liveness_match": liveness_match,
        "is_synthetic_voice": is_synthetic_voice,
        "voice_confidence": voice_confidence,
        "voice_reasoning": voice_reasoning,
        "response_latency_s": response_latency_s,
        "too_slow": too_slow,
        "decision": decision,
        "reason": reason,
        "created_at": datetime.now(timezone.utc),
    })
