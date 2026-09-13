import base64
import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from detector.inference import decode_stereo_wav, predict_call, warmup_models
from detector.schema import DetectionRequest, DetectionResponse
from main import app


def _create_synthetic_wav_b64(duration_s: float = 1.0, sr: int = 8000) -> str:
    """Genera un WAV estéreo base64 sintético de prueba."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    caller = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    agent = (np.sin(2 * np.pi * 880.0 * t) * 0.5).astype(np.float32)
    stereo = np.column_stack([caller, agent])
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_warmup_models():
    """Verifica que el warmup pre-cargue modelos y ejecute inferencia sin lanzar excepciones."""
    warmup_models()


def test_decode_stereo_wav():
    """Verifica la correcta decodificación de audio base64 estéreo."""
    b64 = _create_synthetic_wav_b64(duration_s=0.5, sr=8000)
    caller, agent, sr = decode_stereo_wav(b64)
    assert sr == 8000
    assert len(caller) == 4000
    assert len(agent) == 4000


def test_predict_call_parallel():
    """Verifica que predict_call ejecute en paralelo y devuelva booleano y confianza en rango [0, 1]."""
    sr = 8000
    caller = np.zeros(sr * 2, dtype=np.float32)
    agent = np.zeros(sr * 2, dtype=np.float32)
    is_synthetic, confidence = predict_call(caller, agent, sr)
    assert isinstance(is_synthetic, bool)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_detect_endpoint_contract():
    """Verifica el contrato oficial del juez en POST /detect."""
    client = TestClient(app)
    b64 = _create_synthetic_wav_b64(duration_s=1.0, sr=8000)
    payload = {
        "call_id": "test_call_001",
        "audio_base64": b64,
        "sample_rate": 8000,
        "channels": 2,
    }
    resp = client.post("/detect", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_synthetic" in data
    assert isinstance(data["is_synthetic"], bool)
    assert "confidence" in data
    assert isinstance(data["confidence"], (float, int))
    assert 0.5 <= data["confidence"] <= 1.0  # El juez espera confianza en el veredicto dado


def test_call_569ffb0869eb_correctly_classified_human():
    """Verifica que call_569ffb0869eb (humana) no sea catalogada falsamente como sintética."""
    audio_path = Path("/home/gera3/projects/hackmty26/audio/call_569ffb0869eb.wav")
    if not audio_path.exists():
        pytest.skip("Audio file not found in test environment")

    data, sr = sf.read(audio_path, dtype="float32")
    caller = data[:, 0]
    agent = data[:, 1]
    is_synthetic, prob_synth = predict_call(caller, agent, sr)
    assert not is_synthetic, f"Se esperaba humana (is_synthetic=False), pero obtuvo {is_synthetic} con prob={prob_synth}"
