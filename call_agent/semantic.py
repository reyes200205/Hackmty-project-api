import asyncio
import io
import json
import logging
import os
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

logger = logging.getLogger("gemini-judge")

MODEL = "gemini-3.6-flash"
MIN_AUDIO_SECONDS = 0.3
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5  # se multiplica por el numero de intento (1.5s, 3s)

_client: genai.Client | None = None

JUDGE_PROMPT = """Eres un analista de fraude de un banco mexicano. Acabas de escuchar el audio \
de la respuesta de un cliente durante una llamada telefonica, justo despues de que el agente \
del banco dijo lo siguiente:

"{agent_text}"

Transcribe lo que dice el cliente en el audio y evalua si esa respuesta suena como la de una \
persona real (dudas, muletillas, lenguaje natural, "no se", "no tengo eso") o como generada \
por una inteligencia artificial (demasiado estructurada, inventa informacion cuando se le \
pregunta por algo que no existe, o reacciona de forma extraña a interrupciones/silencios).
Si el audio esta vacio o es solo silencio/ruido, dilo en el transcript y pon sounds_human en null.

Responde SOLO en JSON con este formato exacto:
{{"transcript": "...", "sounds_human": true, "confidence": 0.0, "reasoning": "..."}}
"""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def judge_response(agent_text: str, caller_pcm: bytes, sample_rate: int = 8000) -> dict:
    """Manda el audio del caller a Gemini para transcribir y juzgar si suena humano.
    Se corre en background (no bloquea la conversacion en vivo)."""
    min_bytes = int(sample_rate * 2 * MIN_AUDIO_SECONDS)
    if len(caller_pcm) < min_bytes:
        return {"transcript": "", "sounds_human": None, "confidence": 0.0, "reasoning": "audio insuficiente"}

    wav_bytes = _pcm_to_wav_bytes(caller_pcm, sample_rate)
    client = _get_client()
    contents = [
        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        JUDGE_PROMPT.format(agent_text=agent_text),
    ]

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return json.loads(response.text)
        except genai_errors.ServerError as e:
            # 503/500: sobrecarga temporal del lado de Google, vale la pena reintentar.
            last_error = e
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning("Gemini 5xx (intento %d/%d), reintentando en %.1fs: %s",
                               attempt, MAX_ATTEMPTS, wait, e)
                await asyncio.sleep(wait)
        except Exception as e:
            # Error no transitorio (4xx, JSON invalido, etc): no vale la pena reintentar.
            last_error = e
            break

    logger.error("Gemini fallo al juzgar la respuesta tras %d intento(s): %s", MAX_ATTEMPTS, last_error)
    return {"transcript": "", "sounds_human": None, "confidence": 0.0, "reasoning": f"error: {last_error}"}
