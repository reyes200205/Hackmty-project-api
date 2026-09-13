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
# Google (gTTS) INYECTADO DIGITAL DIRECTO (sin pasar por bocina/microfono) vs
# 10 grabaciones de voz humana real por telefono.
#               pitch_std(Hz)  jitter   shimmer
#  SAPI TTS         40.0       0.0205   0.1672
#  Google TTS       43.2       0.0146   0.1140
#  Humanos (rango)  50.0-93.6  0.032-0.27  0.137-0.226
#
# LIMITE CONOCIDO (confirmado empiricamente el 13-sep-2026): si el TTS se
# reproduce por una bocina hacia el microfono del telefono en vez de
# inyectarse directo, el ruido de esa reproduccion fisica ("replay attack" en
# la literatura de anti-spoofing) empuja pitch_std/jitter/shimmer al MISMO
# rango que voces humanas reales (se probo con Google Translate: pitch_std
# 66-91Hz, jitter 0.09-0.23, shimmer 0.20-0.22 -- indistinguible de humano
# con estas 3 features). NINGUN umbral de estas 3 señales resuelve esto; hay
# que subir umbrales con cuidado de no volver a generar falsos positivos
# contra voz humana real (ya paso una vez: umbrales sueltos = ~80% de los
# humanos marcados como falsos positivos). La seguridad real contra este
# escenario depende de que la frase dinamica nunca se diga en la llamada.
PITCH_STD_THRESHOLD_HZ = 45.0
JITTER_THRESHOLD = 0.025
SHIMMER_THRESHOLD = 0.13
MIN_SUSPICIOUS_VOTES = 2

# Fix (13-sep-2026): agrega dos señales que NO dependen de pitch/jitter/shimmer,
# para no repetir el error de subir esos 3 umbrales (ya penalizado con 80% de
# falsos positivos contra humanos reales, ver nota de arriba).
#
# 1) digital_silence_ratio: un TTS inyectado DIRECTO (sin pasar por bocina/mic)
#    deja huecos de silencio exactamente en 0.0 entre palabras -- un telefono
#    real, aunque este en silencio, siempre trae ruido termico de fondo.
#    Medido: TTS directo ~0.21 (21% de las muestras en cero exacto) vs llamadas
#    humanas reales completas del dataset del reto, siempre <0.001. Margen de
#    ~200x, sin riesgo real de falso positivo. Esto SI se pierde si el ataque
#    pasa por bocina+microfono (el ruido de esa regrabacion tapa los ceros),
#    por eso es un voto aparte, no un reemplazo de la señal 2.
#
# 2) tiempo antes de la primera palabra (check_response_latency): el dueño real
#    ya trae la frase en pantalla de su app y la lee casi de inmediato. Alguien
#    copiandola a un traductor y reproduciendola necesita varios segundos extra
#    para prepararla -- una demora que la calidad del audio replayed no puede
#    disimular, porque no es una señal acustica, es de tiempo.
DIGITAL_SILENCE_RATIO_THRESHOLD = 0.02
MAX_SILENCE_BEFORE_SPEECH_S = 5.0

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

    if feats.get("voiced_fraction", 0.0) < 0.05 or feats.get("pitch_mean", 0.0) == 0.0:
        return False, 0.0, "sin habla suficiente detectada para analizar voz"

    pitch_std = feats["pitch_std"]
    jitter = feats["jitter"]
    shimmer = feats["shimmer"]
    digital_silence_ratio = feats["digital_silence_ratio"]

    votes = {
        "pitch_std": pitch_std < PITCH_STD_THRESHOLD_HZ,
        "jitter": jitter < JITTER_THRESHOLD,
        "shimmer": shimmer < SHIMMER_THRESHOLD,
    }
    suspicious_count = sum(votes.values())
    digital_silence_flag = digital_silence_ratio > DIGITAL_SILENCE_RATIO_THRESHOLD

    # DESACTIVADO (13-sep-2026): causo un falso positivo real contra un cliente
    # humano genuino en una llamada de Twilio de verdad (confidence=0.9, exacto
    # el valor que forzaba esta regla). Hipotesis: las llamadas VoIP de Twilio
    # pueden insertar silencio digital exacto en pausas naturales por supresion
    # de silencio de la red, algo que el dataset de Altur (grabaciones ya hechas,
    # sin ese salto VoIP) nunca tiene -- el umbral se calibro solo contra ese
    # dataset, no contra audio real de Twilio. No lo re-habilites sin antes
    # medir digital_silence_ratio en varias grabaciones reales de Twilio (humanas)
    # para confirmar un umbral que no dispare con ellas.
    # La proteccion real contra inyeccion directa/replay sigue en pie via la
    # palabra de vivacidad + tiempo de respuesta (confirmation_service.py).
    is_synthetic = suspicious_count >= MIN_SUSPICIOUS_VOTES

    confidence = 0.5 + 0.15 * suspicious_count if is_synthetic else max(0.2, 0.5 - 0.1 * suspicious_count)

    reasoning = (
        f"pitch_std={pitch_std:.1f}Hz, jitter={jitter:.4f}, shimmer={shimmer:.4f} "
        f"-> {suspicious_count}/3 señales sospechosas ({', '.join(k for k, v in votes.items() if v) or 'ninguna'}); "
        f"digital_silence_ratio={digital_silence_ratio:.4f} "
        f"({'SOSPECHOSO, inyeccion directa probable' if digital_silence_flag else 'normal'})"
    )
    return is_synthetic, round(confidence, 2), reasoning


def check_response_latency(wav_bytes: bytes) -> tuple[bool, float]:
    """True si el silencio antes de la primera palabra es sospechosamente largo.

    El dueño real de la cuenta ya trae la frase de confirmacion en pantalla y
    la lee casi de inmediato despues del beep. Alguien copiando la frase a un
    traductor y reproduciendola por otra bocina necesita varios segundos extra
    para prepararla -- una demora que la calidad del audio no puede disimular
    porque no es una señal acustica del audio en si, es de tiempo.

    Limitacion conocida: umbral elegido por razonamiento (attacker necesita
    copiar+pegar+generar+reproducir), no calibrado contra grabaciones reales
    del ataque -- no existen muestras etiquetadas de esto para validarlo."""
    from detector.conversational import compute_vad_intervals

    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    caller = data[:, 0]
    intervals = compute_vad_intervals(caller, sample_rate)

    if not intervals:
        return True, 999.0

    time_to_first_speech = intervals[0][0]
    return time_to_first_speech > MAX_SILENCE_BEFORE_SPEECH_S, time_to_first_speech
