import asyncio
import base64
import json
import logging
import time

from fastapi import WebSocket

from call_logs.store import save_turn, start_call

from .script import CALL_SCRIPT
from .semantic import judge_response
from .session import VOICE_RMS_THRESHOLD, CallSession
from .tts import get_phrase_ulaw

logger = logging.getLogger("call-agent")

CHUNK_BYTES = 160  # 20ms de audio mu-law a 8kHz (1 byte/muestra)
CHUNK_SECONDS = 0.02
SILENCE_CUTOFF = 0.7  # si el caller lleva esto callado tras haber hablado, se considera que termino su turno
POLL_INTERVAL = 0.1


async def _send_ulaw(websocket: WebSocket, stream_sid: str, session: CallSession, ulaw_bytes: bytes) -> None:
    # Reloj absoluto: si un envio tarda mas de 20ms, el siguiente duerme menos
    # para recuperar el atraso, en vez de acumular retraso frame a frame
    # (eso es lo que sonaba "trabado").
    start = time.monotonic()
    chunk_index = 0
    for i in range(0, len(ulaw_bytes), CHUNK_BYTES):
        chunk = ulaw_bytes[i:i + CHUNK_BYTES]
        session.ingest_agent_ulaw(chunk)
        payload = base64.b64encode(chunk).decode("ascii")
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }))
        chunk_index += 1
        target_time = start + chunk_index * CHUNK_SECONDS
        sleep_time = target_time - time.monotonic()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


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
    logger.info("Listen agoto su tiempo maximo (%.1fs) sin deteccion clara de fin de turno (pico de energia visto: %d, umbral: %d)",
                max_seconds, session.peak_rms, VOICE_RMS_THRESHOLD)


async def _analyze_and_log(call_sid: str, step_index: int, agent_text: str,
                            caller_segment: bytes, peak_rms: int) -> None:
    """Manda el segmento del caller a Gemini y guarda el veredicto en Mongo.
    Corre en background: no debe frenar la conversacion en vivo."""
    try:
        result = await judge_response(agent_text, caller_segment)
        await save_turn(call_sid, {
            "step_index": step_index,
            "agent_text": agent_text,
            "peak_rms": peak_rms,
            **result,
        })
        logger.info("Gemini -> turno %d: sounds_human=%s confianza=%s transcript=%r",
                    step_index, result.get("sounds_human"), result.get("confidence"), result.get("transcript"))
    except Exception:
        logger.exception("Fallo el analisis/guardado del turno %d", step_index)


async def run_call_script(websocket: WebSocket, stream_sid: str, session: CallSession) -> None:
    """Recorre el guion de trampas hablando por el WebSocket de Twilio."""
    if session.call_sid:
        await start_call(session.call_sid, stream_sid, session.customer_name)

    try:
        name_part = f", {session.customer_name}" if session.customer_name else ""
        last_agent_text = ""
        step_index = 0

        for step in CALL_SCRIPT:
            step_index += 1

            if step["type"] == "speak":
                text = step["text"].format(name_part=name_part)
                last_agent_text = text
                logger.info("Agente dice: %s", text)
                ulaw = get_phrase_ulaw(text)
                await _send_ulaw(websocket, stream_sid, session, ulaw)

            elif step["type"] == "listen":
                logger.info("Agente escuchando (max %.1fs)", step["seconds"])
                segment_start = len(session.caller_pcm)
                await _listen(session, step["seconds"])
                caller_segment = bytes(session.caller_pcm[segment_start:])
                if session.call_sid:
                    asyncio.create_task(_analyze_and_log(
                        session.call_sid, step_index, last_agent_text, caller_segment, session.peak_rms,
                    ))

            elif step["type"] == "silence":
                # Trampa deliberada: silencio fijo, no reacciona a lo que haga el caller.
                logger.info("Agente en silencio deliberado por %.1fs", step["seconds"])
                await asyncio.sleep(step["seconds"])

        logger.info("Guion terminado: %s", stream_sid)
    except asyncio.CancelledError:
        logger.info("Guion cancelado (llamada terminada): %s", stream_sid)
        raise
