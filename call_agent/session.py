import audioop
import time

VOICE_RMS_THRESHOLD = 150  # amplitud PCM16 por encima de la cual se considera "hablando"


class CallSession:
    """Guarda el estado de una llamada: buffers de audio de ambos canales
    (para el futuro muxeo a WAV estereo) y el estado de VAD del caller."""

    def __init__(self):
        self.caller_pcm = bytearray()
        self.agent_pcm = bytearray()
        self.voice_active = False
        self.last_voice_ts: float | None = None
        self.peak_rms = 0
        self.customer_name: str | None = None

    def ingest_caller_ulaw(self, ulaw_bytes: bytes) -> None:
        pcm = audioop.ulaw2lin(ulaw_bytes, 2)
        self.caller_pcm.extend(pcm)

        rms = audioop.rms(pcm, 2)
        self.peak_rms = max(self.peak_rms, rms)
        if rms > VOICE_RMS_THRESHOLD:
            self.last_voice_ts = time.monotonic()
            self.voice_active = True

    def ingest_agent_ulaw(self, ulaw_bytes: bytes) -> None:
        self.agent_pcm.extend(audioop.ulaw2lin(ulaw_bytes, 2))

    def silence_duration(self) -> float:
        if self.last_voice_ts is None:
            return 0.0
        return time.monotonic() - self.last_voice_ts

    def reset_vad(self) -> None:
        self.voice_active = False
        self.last_voice_ts = None
