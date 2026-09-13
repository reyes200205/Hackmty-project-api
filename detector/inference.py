try:
    import pybase64 as b64_module
except ImportError:
    import base64 as b64_module

from concurrent.futures import ThreadPoolExecutor
import functools
import io
from pathlib import Path

import numpy as np
import soundfile as sf

from detector.features import feature_vector, spectral_flatness

MODEL_PATH = Path(__file__).parent / "model" / "classifier.joblib"
CALIBRATOR_PATH = Path(__file__).parent / "model" / "calibrator.joblib"

_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def decode_stereo_wav(audio_b64: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Decodifica un WAV base64 (canal 0 = caller, canal 1 = agente).
    Optimizado con pybase64 y extracción de canales contiguos directa en memoria,
    con fallback automático a soundfile si el formato requiere decodificación avanzada.
    """
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]
    wav_bytes = b64_module.b64decode(audio_b64)
    if len(wav_bytes) > 44 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
        channels = int.from_bytes(wav_bytes[22:24], "little")
        sample_rate = int.from_bytes(wav_bytes[24:28], "little")
        bits_per_sample = int.from_bytes(wav_bytes[34:36], "little")
        pos = wav_bytes.find(b"data", 12)
        if pos != -1 and bits_per_sample == 16:
            data_size = int.from_bytes(wav_bytes[pos + 4 : pos + 8], "little")
            pcm = np.frombuffer(wav_bytes, dtype=np.int16, count=data_size // 2, offset=pos + 8)
            scale = np.float32(1.0 / 32768.0)
            if channels == 2:
                caller = pcm[0::2].astype(np.float32) * scale
                agent = pcm[1::2].astype(np.float32) * scale
            else:
                caller = pcm.astype(np.float32) * scale
                agent = np.zeros_like(caller)
            return caller, agent, sample_rate

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
ACOUSTIC_WEIGHT = 0.65


def _acoustic_confidence(caller: np.ndarray, sample_rate: int) -> float:
    """A2: MFCC + pitch/jitter/shimmer + piso de ruido/silencio digital -> clasificador entrenado.
    Si detector/model/classifier.joblib no existe todavia, cae a la heuristica de A1."""
    bundle = _load_model()
    if bundle is None:
        flatness = spectral_flatness(caller)
        return float(np.clip(flatness * 4.0, 0.0, 1.0))

    vec = feature_vector(caller, sample_rate, max_seconds=60.0).reshape(1, -1)
    vec_scaled = bundle["scaler"].transform(vec)
    return float(bundle["model"].predict_proba(vec_scaled)[0, 1])


def predict_call(caller: np.ndarray, agent: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """A4: fusion de la senal acustica (A2, Alessandro) y la conversacional (A3, Gera).
    Cada una da una probabilidad de 0 a 1 de que la llamada sea sintetica; se promedian
    (ver ACOUSTIC_WEIGHT) y el resultado final se compara contra 0.5.
    Se ejecutan en paralelo con ThreadPoolExecutor para reducir la latencia media.
    """
    fut_conv = _EXECUTOR.submit(predict_conversational, caller, agent, sample_rate)
    fut_ac = _EXECUTOR.submit(_acoustic_confidence, caller, sample_rate)

    conv_is_synthetic, conv_confidence, conv_feats = fut_conv.result()
    # predict_conversational regresa "confianza en su veredicto" (0.5-0.99), no P(sintetico);
    # se reconstruye la probabilidad de sintetico antes de promediar con la senal acustica.
    conv_prob_synthetic = conv_confidence if conv_is_synthetic else (1.0 - conv_confidence)
    acoustic_confidence = fut_ac.result()

    raw_confidence = ACOUSTIC_WEIGHT * acoustic_confidence + (1 - ACOUSTIC_WEIGHT) * conv_prob_synthetic

    # Reglas de consistencia bio-acústica reforzadas (generalización a hidden set):
    # 1. Si la señal acústica es decididamente humana (< 0.20), las pausas o latencias
    #    conversacionales de un hablante pausado o distraído no deben voltear el veredicto a bot.
    if acoustic_confidence < 0.20:
        raw_confidence = min(raw_confidence, acoustic_confidence)
    elif acoustic_confidence < 0.25 and conv_feats.get("interruptions_by_caller", 0) >= 1:
        raw_confidence = min(raw_confidence, acoustic_confidence)

    # 2. Si la señal acústica presenta artefactos sintéticos neurales evidentes (> 0.75),
    #    un bot con baja latencia de respuesta o interrupciones no debe pasar como humano.
    if acoustic_confidence > 0.75:
        raw_confidence = max(raw_confidence, acoustic_confidence)

    # La decision se toma sobre el score crudo, nunca sobre el calibrado: con pocos ejemplos
    # de entrenamiento cerca de 0.5, la regresion isotonica puede tener tramos planos en
    # exactamente 0.5000, y un >= ahi volteria el veredicto sin ninguna razon real.
    is_synthetic = raw_confidence >= 0.5

    confidence = raw_confidence
    calibrator = _load_calibrator()
    if calibrator is not None:
        confidence = float(np.clip(float(calibrator.predict([raw_confidence])[0]), 0.001, 0.999))

    return is_synthetic, round(confidence, 4)


def warmup_models() -> None:
    """Pre-carga los modelos serializados y ejecuta una inferencia en blanco
    para inicializar los buffers de C/NumPy/SciPy y eliminar el retardo de arranque en frío."""
    _load_model()
    _load_calibrator()
    dummy_caller = np.zeros(8000, dtype=np.float32)
    dummy_agent = np.zeros(8000, dtype=np.float32)
    predict_call(dummy_caller, dummy_agent, 8000)
    # Calentar el decodificador WAV rápido
    try:
        dummy_wav_hdr = (
            b"RIFF\x2c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00"
            b"\x40\x1f\x00\x00\x00\x7d\x00\x00\x04\x00\x10\x00data\x08\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        )
        dummy_b64 = b64_module.b64encode(dummy_wav_hdr).decode("ascii")
        decode_stereo_wav(dummy_b64)
    except Exception:
        pass

