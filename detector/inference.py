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


def predict_call(caller: np.ndarray, agent: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """Baseline A1: heuristica de planitud espectral del canal del caller.
    TODO(A2): reemplazar por MFCC + jitter/shimmer + piso de ruido/silencio digital
    + clasificador entrenado sobre el dataset. Esto solo garantiza una respuesta
    evaluable desde el dia 1.
    """
    flatness = _spectral_flatness(caller)
    confidence = float(np.clip(flatness * 4.0, 0.0, 1.0))
    is_synthetic = confidence >= 0.5
    return is_synthetic, confidence
