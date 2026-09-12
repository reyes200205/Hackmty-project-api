import io
import logging
import os

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

logger = logging.getLogger("bank")

TRANSCRIBE_MODEL = "whisper-large-v3-turbo"

_clients: list[AsyncGroq] | None = None
_next_idx = 0


def _get_clients() -> list[AsyncGroq]:
    global _clients
    if _clients is None:
        keys = [k.strip() for k in os.environ["GROQ_API_KEYS"].split(",") if k.strip()]
        _clients = [AsyncGroq(api_key=k) for k in keys]
    return _clients


async def _next_client() -> AsyncGroq:
    global _next_idx
    clients = _get_clients()
    client = clients[_next_idx % len(clients)]
    _next_idx += 1
    return client


async def download_recording(recording_url: str) -> bytes:
    """Descarga el audio de una grabacion de Twilio (requiere Basic Auth con
    las credenciales de la cuenta)."""
    url = recording_url if recording_url.endswith(".wav") else f"{recording_url}.wav"
    auth = (os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, auth=auth)
        resp.raise_for_status()
        return resp.content


async def transcribe_wav(wav_bytes: bytes) -> str:
    """Transcribe con Whisper (Groq), rotando entre las api keys disponibles."""
    clients = _get_clients()
    last_error: Exception | None = None
    for _ in range(len(clients)):
        client = await _next_client()
        try:
            result = await client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=TRANSCRIBE_MODEL,
                language="es",
                response_format="text",
            )
            return str(result).strip()
        except Exception as e:
            last_error = e
            logger.warning("Groq transcripcion fallo con una key, rotando: %s", e)
    logger.error("Transcripcion fallo con todas las keys: %s", last_error)
    return ""


def check_voice_authenticity(wav_bytes: bytes) -> tuple[bool, float]:
    """Reutiliza el clasificador acustico real de deteccion de deepfakes
    (detector.inference.predict_call, de Alessandro/Gera) sin modificar ese
    paquete. Se le pasa un canal 'agente' en silencio ya que aqui solo nos
    interesa la senal acustica de quien confirma la transferencia."""
    from detector.inference import predict_call

    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    caller = data[:, 0]
    agent = np.zeros_like(caller)
    return predict_call(caller, agent, sample_rate)
