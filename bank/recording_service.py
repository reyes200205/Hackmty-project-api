import io
import logging
import os

import httpx
import soundfile as sf
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

logger = logging.getLogger("bank")

TRANSCRIBE_MODEL = "whisper-large-v3-turbo"

# Calibrado con muestras reales (12-sep-2026): 1 TTS limpio (pitch_std~40Hz,
# jitter~0.02) vs 10 grabaciones de voz humana real por telefono (pitch_std
# 50-94Hz, jitter 0.03-0.27). Es un heuristico con pocos datos de calibracion,
# no un clasificador entrenado -- puede necesitar ajuste con mas muestras.
PITCH_STD_THRESHOLD_HZ = 45.0
JITTER_THRESHOLD = 0.025

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


def check_voice_authenticity(wav_bytes: bytes) -> tuple[bool, float, str]:
    """Heuristica DETERMINISTICA basada en features acusticas crudas (pitch_std,
    jitter) calculadas directo de la señal con detector.features (funciones
    puras de procesamiento de señal de Alessandro/Gera, NO su modelo entrenado
    -- ese esta calibrado para llamadas largas del dataset del reto y no
    generaliza a clips cortos de una sola frase, daba siempre ~0.01).

    Tambien se probo un juicio de Groq con metadatos de Whisper, pero esos
    metadatos salen identicos (varianza 0) para cualquier clip corto de un
    solo segmento, sin importar quien hable -- por eso se descarto."""
    from detector.features import extract_features

    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    caller = data[:, 0]
    feats = extract_features(caller, sample_rate)

    pitch_std = feats["pitch_std"]
    jitter = feats["jitter"]

    suspicious_pitch = pitch_std < PITCH_STD_THRESHOLD_HZ
    suspicious_jitter = jitter < JITTER_THRESHOLD
    is_synthetic = suspicious_pitch and suspicious_jitter

    confidence = 0.55 + 0.2 * (suspicious_pitch + suspicious_jitter) if is_synthetic else 0.3
    reasoning = (
        f"pitch_std={pitch_std:.1f}Hz (umbral {PITCH_STD_THRESHOLD_HZ}), "
        f"jitter={jitter:.4f} (umbral {JITTER_THRESHOLD})"
    )
    return is_synthetic, round(confidence, 2), reasoning
