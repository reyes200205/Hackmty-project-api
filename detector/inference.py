import base64
import io

import numpy as np
import soundfile as sf


def decode_stereo_wav(audio_b64: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Decodifica un WAV base64 (canal 0 = caller, canal 1 = agente)."""
    wav_bytes = base64.b64decode(audio_b64)
    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    return caller, agent, sample_rate


def _spectral_flatness(x: np.ndarray, frame: int = 1024, hop: int = 512) -> float:
    if len(x) < frame:
        return 0.0
    window = np.hanning(frame)
    flatness_vals = []
    for start in range(0, len(x) - frame, hop):
        seg = x[start:start + frame] * window
        mag = np.abs(np.fft.rfft(seg)) + 1e-10
        gm = np.exp(np.mean(np.log(mag)))
        am = np.mean(mag)
        flatness_vals.append(gm / am)
    return float(np.mean(flatness_vals)) if flatness_vals else 0.0


from .conversational import predict_conversational


def predict_call(caller: np.ndarray, agent: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """Inferencia de la llamada:
    - Fase A3: Señal conversacional basada en VAD cruzado de ambos canales (latencia de respuesta,
      solapes, interrupciones y dinámica temporal).
    - Fase A2 (Acústico): Alessandro sumará aquí sus features acústicos (MFCCs, piso de ruido).
    """
    is_synthetic, confidence, _ = predict_conversational(caller, agent, sample_rate)
    return is_synthetic, confidence
