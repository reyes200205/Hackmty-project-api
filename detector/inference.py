import base64
import functools
import io
from pathlib import Path

import numpy as np
import soundfile as sf

from detector.features import feature_vector, spectral_flatness

MODEL_PATH = Path(__file__).parent / "model" / "classifier.joblib"
CALIBRATOR_PATH = Path(__file__).parent / "model" / "calibrator.joblib"


def decode_stereo_wav(audio_b64: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Decodifica un WAV base64 (canal 0 = caller, canal 1 = agente)."""
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]
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


@functools.lru_cache(maxsize=1)
def _load_calibrator():
    if not CALIBRATOR_PATH.exists():
        return None
    import joblib
    bundle = joblib.load(CALIBRATOR_PATH)
    return bundle.get("calibrator")


from .conversational import predict_conversational

# A4: peso de cada senal en la fusion. 50/50 se eligio porque en validacion (71 llamadas)
# da el mismo resultado (acc=0.986, auc=1.000) que darle mas peso a lo acustico, y es mas
# simple de explicar. Ver eval_fusion en el historial de la conversacion para los numeros.
ACOUSTIC_WEIGHT = 0.5


def _acoustic_confidence(caller: np.ndarray, sample_rate: int) -> float:
    """A2: MFCC + pitch/jitter/shimmer + piso de ruido/silencio digital -> clasificador entrenado.
    Si detector/model/classifier.joblib no existe todavia, cae a la heuristica de A1."""
    bundle = _load_model()
    if bundle is None:
        flatness = spectral_flatness(caller)
        return float(np.clip(flatness * 4.0, 0.0, 1.0))

    vec = feature_vector(caller, sample_rate).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    return float(bundle["model"].predict_proba(vec_scaled)[0, 1])


def predict_call(caller: np.ndarray, agent: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """A4: fusion de la senal acustica (A2, Alessandro) y la conversacional (A3, Gera).
    Cada una da una probabilidad de 0 a 1 de que la llamada sea sintetica; se promedian
    (ver ACOUSTIC_WEIGHT) y el resultado final se compara contra 0.5.
    """
    conv_is_synthetic, conv_confidence, _ = predict_conversational(caller, agent, sample_rate)
    # predict_conversational regresa "confianza en su veredicto" (0.5-0.99), no P(sintetico);
    # se reconstruye la probabilidad de sintetico antes de promediar con la senal acustica.
    conv_prob_synthetic = conv_confidence if conv_is_synthetic else (1.0 - conv_confidence)
    acoustic_confidence = _acoustic_confidence(caller, sample_rate)

    raw_confidence = ACOUSTIC_WEIGHT * acoustic_confidence + (1 - ACOUSTIC_WEIGHT) * conv_prob_synthetic
    # La decision se toma sobre el score crudo, nunca sobre el calibrado: con pocos ejemplos
    # de entrenamiento cerca de 0.5, la regresion isotonica puede tener tramos planos en
    # exactamente 0.5000, y un >= ahi volteria el veredicto sin ninguna razon real.
    is_synthetic = raw_confidence >= 0.5

    confidence = raw_confidence
    calibrator = _load_calibrator()
    if calibrator is not None:
        confidence = float(np.clip(float(calibrator.predict([raw_confidence])[0]), 0.001, 0.999))

    return is_synthetic, round(confidence, 4)
