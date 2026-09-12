import audioop
import hashlib
import wave
from pathlib import Path

import pyttsx3

CACHE_DIR = Path(__file__).parent / "tts_cache"
CACHE_DIR.mkdir(exist_ok=True)

SABINA_VOICE_ID = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0"
SPEECH_RATE = 175


def _synthesize_wav(text: str, wav_path: Path) -> None:
    engine = pyttsx3.init()
    engine.setProperty("voice", SABINA_VOICE_ID)
    engine.setProperty("rate", SPEECH_RATE)
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()


def _wav_to_ulaw_8k(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as w:
        assert w.getsampwidth() == 2, "se espera PCM de 16 bits"
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
        channels = w.getnchannels()

    if channels == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)

    if rate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 8000, None)

    return audioop.lin2ulaw(pcm, 2)


def get_phrase_ulaw(text: str) -> bytes:
    """Devuelve el audio de una frase en mu-law 8kHz mono, generandolo y
    cacheandolo en disco la primera vez que se pide."""
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    ulaw_path = CACHE_DIR / f"{key}.ulaw"

    if ulaw_path.exists():
        return ulaw_path.read_bytes()

    wav_path = CACHE_DIR / f"{key}.wav"
    _synthesize_wav(text, wav_path)
    ulaw_bytes = _wav_to_ulaw_8k(wav_path)
    ulaw_path.write_bytes(ulaw_bytes)
    wav_path.unlink(missing_ok=True)
    return ulaw_bytes
