import base64
import functools
import io
from pathlib import Path

import numpy as np
import soundfile as sf

from detector.features import feature_vector, spectral_flatness

MODEL_PATH = Path(__file__).parent / "model" / "classifier.joblib"


def decode_stereo_wav(audio_b64: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Decodifica un WAV base64 (canal 0 = caller, canal 1 = agente)."""
    wav_bytes = base64.b64decode(audio_b64)
    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    return caller, agent, sample_rate


@functools.lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def predict_call(caller: np.ndarray, agent: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """A2: MFCC + pitch/jitter/shimmer + piso de ruido/silencio digital -> clasificador entrenado.
    Si detector/model/classifier.joblib no existe todavia (falta correr detector/train.py),
    cae de vuelta a la heuristica de planitud espectral de A1 para no romper el endpoint.
    """
    bundle = _load_model()
    if bundle is None:
        flatness = spectral_flatness(caller)
        confidence = float(np.clip(flatness * 4.0, 0.0, 1.0))
        return confidence >= 0.5, confidence

    vec = feature_vector(caller, sample_rate).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    confidence = float(bundle["model"].predict_proba(vec_scaled)[0, 1])
    is_synthetic = confidence >= 0.5
    return is_synthetic, confidence
