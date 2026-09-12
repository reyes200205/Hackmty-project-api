import asyncio
import base64
import json
import logging
import time

from fastapi import WebSocket

from .script import CALL_SCRIPT
from .session import CallSession
from .tts import get_phrase_ulaw

logger = logging.getLogger("call-agent")

CHUNK_BYTES = 160  # 20ms de audio mu-law a 8kHz (1 byte/muestra)
CHUNK_SECONDS = 0.02
SILENCE_CUTOFF = 0.7  # si el caller lleva esto callado tras haber hablado, se considera que termino su turno
POLL_INTERVAL = 0.1


async def _send_ulaw(websocket: WebSocket, stream_sid: str, session: CallSession, ulaw_bytes: bytes) -> None:
    for i in range(0, len(ulaw_bytes), CHUNK_BYTES):
        chunk = ulaw_bytes[i:i + CHUNK_BYTES]
        session.ingest_agent_ulaw(chunk)
        payload = base64.b64encode(chunk).decode("ascii")
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }))
        await asyncio.sleep(CHUNK_SECONDS)


async def _listen(session: CallSession, max_seconds: float) -> None:
    """Espera hasta max_seconds, pero corta antes si el caller ya hablo y
    luego se quedo callado (termino su turno de verdad)."""
    session.reset_vad()
    start = time.monotonic()
    while time.monotonic() - start < max_seconds:
        await asyncio.sleep(POLL_INTERVAL)
        if session.voice_active and session.silence_duration() >= SILENCE_CUTOFF:
            logger.info("Caller dejo de hablar, cortando el listen antes de tiempo (%.2fs de %.1fs)",
                        time.monotonic() - start, max_seconds)
            return
    logger.info("Listen agoto su tiempo maximo (%.1fs) sin deteccion clara de fin de turno", max_seconds)


async def run_call_script(websocket: WebSocket, stream_sid: str, session: CallSession) -> None:
    """Recorre el guion de trampas hablando por el WebSocket de Twilio."""
    try:
        for step in CALL_SCRIPT:
            if step["type"] == "speak":
                logger.info("Agente dice: %s", step["text"])
                ulaw = get_phrase_ulaw(step["text"])
                await _send_ulaw(websocket, stream_sid, session, ulaw)
            elif step["type"] == "listen":
                logger.info("Agente escuchando (max %.1fs)", step["seconds"])
                await _listen(session, step["seconds"])
            elif step["type"] == "silence":
                # Trampa deliberada: silencio fijo, no reacciona a lo que haga el caller.
                logger.info("Agente en silencio deliberado por %.1fs", step["seconds"])
                await asyncio.sleep(step["seconds"])
        logger.info("Guion terminado: %s", stream_sid)
    except asyncio.CancelledError:
        logger.info("Guion cancelado (llamada terminada): %s", stream_sid)
        raise
