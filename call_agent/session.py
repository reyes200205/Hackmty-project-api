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
        self.call_sid: str | None = None

    def ingest_caller_ulaw(self, ulaw_bytes: bytes) -> None:
        pcm = audioop.ulaw2lin(ulaw_bytes, 2)
        self.caller_pcm.extend(pcm)

        rms = audioop.rms(pcm, 2)
        self.peak_rms = max(self.peak_rms, rms)
        if rms > VOICE_RMS_THRESHOLD:
            self.last_voice_ts = time.monotonic()
            self.voice_active = True

    def ingest_agent_ulaw(self, ulaw_bytes: bytes) -> None:
        # Si caller_pcm va más adelante (porque el agente estuvo escuchando en silencio),
        # rellenar agent_pcm con silencio para que los timestamps de ambos canales coincidan.
        diff = len(self.caller_pcm) - len(self.agent_pcm)
        if diff > 0:
            self.agent_pcm.extend(b"\x00" * diff)
        self.agent_pcm.extend(audioop.ulaw2lin(ulaw_bytes, 2))

    def silence_duration(self) -> float:
        if self.last_voice_ts is None:
            return 0.0
        return time.monotonic() - self.last_voice_ts

    def reset_vad(self) -> None:
        self.voice_active = False
        self.last_voice_ts = None

    def get_stereo_arrays(self, sample_rate: int = 8000):
        """Devuelve arrays normalizados en [-1.0, 1.0] de (caller, agent, sample_rate)
        alineados exactamente en la misma línea de tiempo."""
        import numpy as np

        if len(self.caller_pcm) < 2:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), sample_rate

        caller_np = np.frombuffer(self.caller_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        caller_len = len(caller_np)

        if len(self.agent_pcm) < 2:
            agent_np = np.zeros_like(caller_np)
        else:
            raw_agent = np.frombuffer(self.agent_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(raw_agent) < caller_len:
                agent_np = np.pad(raw_agent, (0, caller_len - len(raw_agent)))
            else:
                agent_np = raw_agent[:caller_len]

        return caller_np, agent_np, sample_rate

    def evaluate_live_detection(self) -> dict:
        """Evalúa si el audio acumulado hasta este momento corresponde a voz sintética/IA.
        Utiliza el clasificador multimodal calibrado (acústico + conversacional)."""
        caller_np, agent_np, sr = self.get_stereo_arrays()
        if len(caller_np) < int(sr * 0.5):
            return {
                "evaluated": False,
                "is_synthetic": False,
                "confidence": 0.0,
                "reason": "audio insuficiente (< 0.5s)",
            }

        # Verificar si hubo actividad vocal real del caller para no clasificar silencio
        from detector.conversational import compute_vad_intervals
        caller_vad = compute_vad_intervals(caller_np, sr)
        if not caller_vad:
            return {
                "evaluated": False,
                "is_synthetic": False,
                "confidence": 0.0,
                "reason": "sin habla detectada en el canal del caller",
            }

        from detector.inference import predict_call
        is_synthetic, confidence = predict_call(caller_np, agent_np, sr)
        return {
            "evaluated": True,
            "is_synthetic": bool(is_synthetic),
            "confidence": float(confidence),
            "reason": "modelo hibrido acustico y conversacional",
        }
