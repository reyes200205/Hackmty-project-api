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

# Calibrado con muestras reales (12-sep-2026): TTS de Windows (SAPI) y de
# Google (gTTS, bajado a 8kHz simulando telefono) vs 10 grabaciones de voz
# humana real por telefono.
#               pitch_std(Hz)  jitter   shimmer
#  SAPI TTS         40.0       0.0205   0.1672
#  Google TTS       43.2       0.0146   0.1140
#  Humanos (rango)  50.0-93.6  0.032-0.27  0.137-0.226
# Ninguna señal sola separa perfectamente los dos TTS de los 10 humanos con
# margen comodo, pero las 3 combinadas si: se cuenta cuantas de las 3 caen en
# el lado "sospechoso" y se marca sintetico con mayoria (2 de 3). Esto le da
# margen si el ruido de una grabacion real (ej. reproducir el TTS por bocina
# hacia otro telefono) "humaniza" una sola señal por ruido de fondo.
PITCH_STD_THRESHOLD_HZ = 45.0
JITTER_THRESHOLD = 0.025
SHIMMER_THRESHOLD = 0.13
MIN_SUSPICIOUS_VOTES = 2

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
    jitter, shimmer) calculadas directo de la señal con detector.features
    (funciones puras de procesamiento de señal de Alessandro/Gera, NO su
    modelo entrenado -- ese esta calibrado para llamadas largas del dataset
    del reto y no generaliza a clips cortos de una sola frase, daba siempre
    ~0.01 sin importar el audio).

    Limitacion conocida: es un heuristico con pocas muestras de calibracion
    (2 TTS, 10 humanos), no un clasificador entrenado. TTS neuronal de muy
    alta calidad, o reproducido con ruido de fondo real, puede seguir
    pasando. La seguridad real de la confirmacion depende principalmente de
    que la frase dinamica no se diga en la llamada (solo en la app)."""
    from detector.features import extract_features

    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    caller = data[:, 0]
    feats = extract_features(caller, sample_rate)

    pitch_std = feats["pitch_std"]
    jitter = feats["jitter"]
    shimmer = feats["shimmer"]

    votes = {
        "pitch_std": pitch_std < PITCH_STD_THRESHOLD_HZ,
        "jitter": jitter < JITTER_THRESHOLD,
        "shimmer": shimmer < SHIMMER_THRESHOLD,
    }
    suspicious_count = sum(votes.values())
    is_synthetic = suspicious_count >= MIN_SUSPICIOUS_VOTES

    confidence = 0.5 + 0.15 * suspicious_count if is_synthetic else max(0.2, 0.5 - 0.1 * suspicious_count)
    reasoning = (
        f"pitch_std={pitch_std:.1f}Hz, jitter={jitter:.4f}, shimmer={shimmer:.4f} "
        f"-> {suspicious_count}/3 señales sospechosas ({', '.join(k for k, v in votes.items() if v) or 'ninguna'})"
    )
    return is_synthetic, round(confidence, 2), reasoning
