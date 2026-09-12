import asyncio
import base64
import json
import logging

from fastapi import WebSocket

from .script import CALL_SCRIPT
from .tts import get_phrase_ulaw

logger = logging.getLogger("call-agent")

CHUNK_BYTES = 160  # 20ms de audio mu-law a 8kHz (1 byte/muestra)
CHUNK_SECONDS = 0.02


async def _send_ulaw(websocket: WebSocket, stream_sid: str, ulaw_bytes: bytes) -> None:
    for i in range(0, len(ulaw_bytes), CHUNK_BYTES):
        chunk = ulaw_bytes[i:i + CHUNK_BYTES]
        payload = base64.b64encode(chunk).decode("ascii")
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }))
        await asyncio.sleep(CHUNK_SECONDS)


async def run_call_script(websocket: WebSocket, stream_sid: str) -> None:
    """Recorre el guion de trampas hablando por el WebSocket de Twilio."""
    try:
        for step in CALL_SCRIPT:
            if step["type"] == "speak":
                logger.info("Agente dice: %s", step["text"])
                ulaw = get_phrase_ulaw(step["text"])
                await _send_ulaw(websocket, stream_sid, ulaw)
            elif step["type"] in ("listen", "silence"):
                logger.info("Agente en %s por %.1fs", step["type"], step["seconds"])
                await asyncio.sleep(step["seconds"])
        logger.info("Guion terminado: %s", stream_sid)
    except asyncio.CancelledError:
        logger.info("Guion cancelado (llamada terminada): %s", stream_sid)
        raise
