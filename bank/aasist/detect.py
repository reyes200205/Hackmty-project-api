"""Wrapper de inferencia para AASIST (NAVER Corp, MIT license -- codigo y
pesos vendorizados en este paquete desde https://github.com/clovaai/aasist,
ver LICENSE_AASIST). Red neuronal preentrenada de deteccion de voz sintetica,
opera directo sobre la forma de onda cruda de clips cortos (~4s) -- justo el
tamano de las confirmaciones de transferencia, donde ni el heuristico de
pitch/jitter/shimmer ni el modelo principal de /detect funcionan bien.

Validado el 13-sep-2026 con audio real de Google Translate (el ataque
reportado): humanos del dataset de Altur ~0.9 a 1.5, Google TTS directo
-6.44, "regrabado" simulado por bocina+microfono -5.53 -- SI distingue el
ataque real, a diferencia del heuristico acustico que el regrabado engañaba
por completo.

LIMITES CONOCIDOS (importante leer antes de subir el umbral de decision):
1. NO generaliza al dataset del reto de Altur en general (accuracy 0.60, auc
   0.585 en 40 llamadas) -- nunca usar esto para /detect, solo aqui.
2. En una muestra mas amplia de humanos reales del dataset de Altur, unos
   pocos sacaron scores negativos (-1.05, -3.25, -4.76) -- no tan extremos
   como el ataque real (-6.44/-5.53), pero cerca. Un corte en 0 arriesga
   falsos positivos contra humanos reales (~15% en esa muestra). NO se ha
   podido validar contra audio real grabado por Twilio (solo contra el
   dataset de Altur y contra TTS generado localmente), asi que el
   comportamiento real en produccion es incierto -- exactamente el mismo
   tipo de sorpresa que ya paso con digital_silence_ratio (ver nota en
   recording_service.py). Por eso el umbral por default es conservador
   (-4.0, no 0.0) y esto entra como señal informativa, no como rechazo
   automatico, hasta que se confirme con llamadas reales de Twilio.
"""
import functools
import io
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
try:
    import torch
    from .model import Model
except Exception:
    torch = None
    Model = None

_HERE = Path(__file__).parent
_WEIGHTS_PATH = _HERE / "AASIST.pth"
_SAMPLE_RATE = 16000
_MAX_LEN = 64600  # ~4s a 16kHz, tamano fijo que espera el modelo

# Conservador a proposito (ver limite #2 arriba): el humano mas bajo visto en
# el dataset de Altur saco -4.76, y el ataque real de Google Translate -5.53
# y -6.44 -- -5.0 queda entre ambos con margen. Sube esto hacia 0.0 solo
# despues de confirmar con varias llamadas reales de Twilio (humanas) que no
# se disparan falsos positivos.
SYNTHETIC_SCORE_THRESHOLD = -5.0

_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


@functools.lru_cache(maxsize=1)
def _load_model():
    model = Model(_MODEL_CONFIG)
    state = torch.load(_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _pad_or_crop(x: np.ndarray, max_len: int = _MAX_LEN) -> np.ndarray:
    if len(x) >= max_len:
        return x[:max_len]
    reps = int(max_len / len(x)) + 1
    return np.tile(x, reps)[:max_len]


def aasist_score(wav_bytes: bytes) -> float:
    """Score crudo del modelo: mientras mas alto, mas 'genuino'; mientras mas
    negativo, mas 'sintetico/replay'. No es una probabilidad calibrada."""
    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    x = data[:, 0]
    if sample_rate != _SAMPLE_RATE:
        n = int(len(x) * _SAMPLE_RATE / sample_rate)
        x = sig.resample(x, n).astype(np.float32)
    x = _pad_or_crop(x)

    model = _load_model()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, out = model(xt)
    return float(out[0, 1])


def check_aasist_authenticity(wav_bytes: bytes) -> tuple[bool, float, str]:
    """Mismo formato que check_voice_authenticity: (is_synthetic, confidence, reasoning).
    is_synthetic solo se activa con SYNTHETIC_SCORE_THRESHOLD (conservador),
    no en 0.0, por el riesgo de falso positivo documentado arriba."""
    if torch is None or Model is None:
        return False, 0.5, "torch no disponible en entorno"
    raw_score = aasist_score(wav_bytes)
    is_synthetic = raw_score < SYNTHETIC_SCORE_THRESHOLD
    confidence = float(np.clip(0.5 + abs(raw_score) * 0.05, 0.5, 0.99))
    reasoning = (
        f"aasist_score={raw_score:.2f} "
        f"({'sintetico/replay probable' if is_synthetic else 'sin evidencia suficiente'})"
    )
    return is_synthetic, round(confidence, 2), reasoning
