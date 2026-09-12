import base64
import time
from pathlib import Path
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

AUDIO_FILE = Path("../hackmty26/audio/call_0294f969f98b.wav")


def test_detect():
    assert AUDIO_FILE.exists(), f"No se encontró el audio {AUDIO_FILE}"
    audio_b64 = base64.b64encode(AUDIO_FILE.read_bytes()).decode("ascii")

    # Warmup
    _ = client.post("/detect", json={"audio_b64": audio_b64})

    # Medición de latencia real end-to-end
    t0 = time.perf_counter()
    response = client.post("/detect", json={"audio_b64": audio_b64})
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"Status Code: {response.status_code}")
    print(f"Respuesta JSON: {response.json()}")
    print(f"Latencia End-to-End: {latency_ms:.2f} ms")

    assert response.status_code == 200
    data = response.json()
    assert "is_synthetic" in data
    assert "confidence" in data
    assert isinstance(data["is_synthetic"], bool)
    assert 0.0 <= data["confidence"] <= 1.0
    print("¡Test de endpoint /detect exitoso!")


if __name__ == "__main__":
    test_detect()
