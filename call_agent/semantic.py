import asyncio
import io
import json
import logging
import os
import wave

from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError

load_dotenv()

logger = logging.getLogger("groq-judge")

TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
JUDGE_MODEL = "openai/gpt-oss-120b"
MIN_AUDIO_SECONDS = 0.3

_clients: list[AsyncGroq] | None = None
_next_idx = 0
_rotation_lock = asyncio.Lock()

JUDGE_PROMPT = """Analiza objetivamente una respuesta transcrita de una llamada telefonica de un \
banco mexicano, sin asumir de antemano que se trata de fraude. El agente del banco dijo:

"{agent_text}"

El cliente respondio (transcrito por un sistema de reconocimiento de voz): "{transcript}"

Importante sobre la transcripcion: el reconocimiento de voz limpia y normaliza el texto -- \
elimina muletillas, dudas ("eh", "este", "mmm") y repeticiones aunque la persona real las haya \
dicho. NO uses "la respuesta suena demasiado limpia/estructurada" como evidencia de IA: eso es \
un artefacto de la transcripcion, no una señal real, y penalizarlo sesga el resultado en contra \
de personas reales que simplemente hablan claro.

Evalua usando solo evidencia robusta que sobrevive a la transcripcion:
- ¿Inventa informacion especifica y detallada cuando se le pregunta por algo que no existe o no \
deberia saber, en vez de decir que no sabe o no tiene esa informacion?
- ¿Ignora por completo la pregunta o responde algo sin relacion (non-sequitur)?
- ¿Repite exactamente la misma frase que ya dijo antes en la conversacion, palabra por palabra?

Si la respuesta es corta, generica (ej. "si", "gracias", "no se") o simplemente no da suficiente \
informacion para decidir con evidencia real, es INCORRECTO adivinar con confianza alta. En ese \
caso responde sounds_human=true con confidence baja (0.5 o menos) -- ante la duda, no acuses.

Responde SOLO en JSON con este formato exacto:
{{"sounds_human": true, "confidence": 0.0, "reasoning": "..."}}
"""


def _get_clients() -> list[AsyncGroq]:
    global _clients
    if _clients is None:
        keys = [k.strip() for k in os.environ["GROQ_API_KEYS"].split(",") if k.strip()]
        _clients = [AsyncGroq(api_key=k) for k in keys]
        logger.info("Groq: %d api keys cargadas para rotacion", len(_clients))
    return _clients


async def _next_client() -> AsyncGroq:
    global _next_idx
    async with _rotation_lock:
        clients = _get_clients()
        client = clients[_next_idx % len(clients)]
        _next_idx += 1
        return client


async def _call_with_rotation(fn):
    """Intenta la llamada rotando entre todas las api keys disponibles.
    Si una key esta rate-limited (429) o falla, se prueba la siguiente."""
    clients = _get_clients()
    last_error: Exception | None = None
    for _ in range(len(clients)):
        client = await _next_client()
        try:
            return await fn(client)
        except RateLimitError as e:
            last_error = e
            logger.warning("Groq key rate-limited, rotando a la siguiente: %s", e)
        except Exception as e:
            last_error = e
            logger.warning("Groq fallo con esta key, rotando a la siguiente: %s", e)
    raise last_error


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def judge_response(agent_text: str, caller_pcm: bytes, sample_rate: int = 8000) -> dict:
    """Transcribe el audio del caller con Whisper (Groq) y usa un LLM (Groq) para
    juzgar si la respuesta suena humana. Se corre en background."""
    min_bytes = int(sample_rate * 2 * MIN_AUDIO_SECONDS)
    if len(caller_pcm) < min_bytes:
        return {"transcript": "", "sounds_human": None, "confidence": 0.0, "reasoning": "audio insuficiente"}

    wav_bytes = _pcm_to_wav_bytes(caller_pcm, sample_rate)

    try:
        async def _transcribe(client: AsyncGroq):
            return await client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=TRANSCRIBE_MODEL,
                language="es",
                response_format="text",
            )

        transcript = str(await _call_with_rotation(_transcribe)).strip()
        if not transcript:
            return {"transcript": "", "sounds_human": None, "confidence": 0.0, "reasoning": "sin habla detectada"}

        async def _judge(client: AsyncGroq):
            return await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(agent_text=agent_text, transcript=transcript),
                }],
                response_format={"type": "json_object"},
                temperature=0,
            )

        completion = await _call_with_rotation(_judge)
        result = json.loads(completion.choices[0].message.content)
        result["transcript"] = transcript
        return result
    except Exception as e:
        logger.error("Groq fallo al juzgar la respuesta (todas las keys agotadas o error no transitorio): %s", e)
        return {"transcript": "", "sounds_human": None, "confidence": 0.0, "reasoning": f"error: {e}"}
