import json
import logging
import os

import httpx
import numpy as np
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

logger = logging.getLogger("bank")

TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT = """Eres un analista de fraude de un banco. Un cliente acaba de decir una \
frase de confirmacion por telefono para autorizar una transferencia.

Frase que se le pidio decir: "{expected}"
Transcripcion de lo que dijo: "{transcript}"

Metadatos tecnicos calculados directamente del audio (no del texto):
- Duracion: {duration:.1f} segundos
- Confianza promedio de la transcripcion (avg_logprob; entre mas cercano a 0, \
mas "limpia"/uniforme sono la señal para el reconocedor de voz): {avg_logprob}
- Variacion de esa confianza entre segmentos (poca variacion = señal muy uniforme, \
tipica de audio sintetico; mucha variacion = mas tipico de habla humana real): {avg_logprob_std}
- Probabilidad de silencio/ruido detectada: {no_speech_prob}

Una voz generada por texto-a-voz (TTS) suele producir una señal digital muy limpia y \
uniforme, sin el ruido de fondo, las micro-variaciones de tono ni las imperfecciones \
tipicas de una persona real hablando por un telefono normal.

Importante: que el texto coincida exactamente con la frase pedida NO es evidencia de \
fraude por si solo -- una persona real tambien puede repetirla correctamente. Enfocate \
en si las caracteristicas TECNICAS de la señal sugieren un origen sintetico/digital.

Responde SOLO en JSON: {{"is_synthetic": true o false, "confidence": 0.0 a 1.0, "reasoning": "..."}}
"""

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


async def _call_with_rotation(fn):
    clients = _get_clients()
    last_error: Exception | None = None
    for _ in range(len(clients)):
        client = await _next_client()
        try:
            return await fn(client)
        except Exception as e:
            last_error = e
            logger.warning("Groq fallo con una key, rotando a la siguiente: %s", e)
    raise last_error


async def transcribe_with_prosody(wav_bytes: bytes) -> tuple[str, dict]:
    """Transcribe con Whisper (Groq) y de paso extrae metricas de prosodia
    (avg_logprob, no_speech_prob) que Whisper calcula a partir del AUDIO real,
    no del texto -- Groq no tiene un modelo de chat que pueda 'escuchar' tono
    directamente como Gemini, asi que esto es la aproximacion mas cercana con
    las herramientas que tenemos."""
    async def _transcribe(client: AsyncGroq):
        return await client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=TRANSCRIBE_MODEL,
            language="es",
            response_format="verbose_json",
        )

    try:
        result = await _call_with_rotation(_transcribe)
    except Exception as e:
        logger.error("Transcripcion fallo con todas las keys: %s", e)
        return "", {}

    text = (result.text or "").strip()
    segments = result.segments or []
    logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    no_speech = [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None]

    prosody = {
        "duration": float(result.duration or 0.0),
        "avg_logprob": float(np.mean(logprobs)) if logprobs else None,
        "avg_logprob_std": float(np.std(logprobs)) if len(logprobs) > 1 else 0.0,
        "no_speech_prob": float(max(no_speech)) if no_speech else None,
    }
    return text, prosody


async def judge_voice_authenticity(expected_phrase: str, transcript: str, prosody: dict) -> tuple[bool, float, str]:
    """Juicio 'best effort' de si la voz suena sintetica, combinando el texto
    con metadatos derivados del audio. Limitacion conocida: si el TTS repite
    la frase exacta pedida, el texto por si solo no lo delata -- este juicio
    depende de que las metricas de prosodia se vean lo bastante distintas."""
    prompt = JUDGE_PROMPT.format(
        expected=expected_phrase,
        transcript=transcript,
        duration=prosody.get("duration") or 0.0,
        avg_logprob=prosody.get("avg_logprob"),
        avg_logprob_std=prosody.get("avg_logprob_std"),
        no_speech_prob=prosody.get("no_speech_prob"),
    )

    async def _judge(client: AsyncGroq):
        return await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

    try:
        completion = await _call_with_rotation(_judge)
        result = json.loads(completion.choices[0].message.content)
        return bool(result.get("is_synthetic", False)), float(result.get("confidence", 0.0)), result.get("reasoning", "")
    except Exception as e:
        logger.error("Juicio de autenticidad de voz fallo con todas las keys: %s", e)
        return False, 0.0, f"error: {e}"
