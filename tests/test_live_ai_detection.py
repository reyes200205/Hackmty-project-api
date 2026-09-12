import asyncio
import audioop
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest

from call_agent.session import CallSession
from call_agent.tts import generate_busy_tone_ulaw
from call_logs import store


def _make_ulaw_tone(freq: float = 440.0, duration: float = 1.0, sr: int = 8000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return audioop.lin2ulaw(sig.tobytes(), 2)


def test_busy_tone_generator():
    ulaw = generate_busy_tone_ulaw(duration_seconds=0.5)
    assert isinstance(ulaw, bytes)
    assert len(ulaw) == 4000  # 8000 Hz * 0.5s = 4000 bytes


def test_session_timeline_synchronization():
    session = CallSession()

    # Caller habla 1 segundo (8000 muestras = 16000 bytes PCM)
    caller_audio = _make_ulaw_tone(freq=300, duration=1.0)
    session.ingest_caller_ulaw(caller_audio)
    assert len(session.caller_pcm) == 16000
    assert len(session.agent_pcm) == 0

    # Agente habla 0.5 segundos después de que el caller habló 1.0s
    agent_audio = _make_ulaw_tone(freq=600, duration=0.5)
    session.ingest_agent_ulaw(agent_audio)

    # El canal del agente debe haberse rellenado con 1.0s de silencio antes del audio del agente
    assert len(session.agent_pcm) == 16000 + 8000

    caller_np, agent_np, sr = session.get_stereo_arrays()
    assert sr == 8000
    assert len(caller_np) == len(agent_np)


def test_evaluate_live_detection_empty_audio():
    session = CallSession()
    res = session.evaluate_live_detection()
    assert res["evaluated"] is False
    assert res["is_synthetic"] is False


def test_evaluate_live_detection_silent_audio():
    session = CallSession()
    # 2 segundos de silencio absoluto (amplitud cero)
    silence = audioop.lin2ulaw(b"\x00" * 32000, 2)
    session.ingest_caller_ulaw(silence)

    res = session.evaluate_live_detection()
    assert res["evaluated"] is False
    assert "sin habla" in res["reason"]


def test_evaluate_live_detection_synthetic_vs_human():
    session = CallSession()
    speech = _make_ulaw_tone(freq=220, duration=2.0)
    session.ingest_caller_ulaw(speech)

    with patch("detector.conversational.compute_vad_intervals", return_value=[(0.1, 1.9)]), \
         patch("detector.inference.predict_call", return_value=(True, 0.95)):
        res = session.evaluate_live_detection()
        assert res["evaluated"] is True
        assert res["is_synthetic"] is True
        assert res["confidence"] == 0.95

    with patch("detector.conversational.compute_vad_intervals", return_value=[(0.1, 1.9)]), \
         patch("detector.inference.predict_call", return_value=(False, 0.12)):
        res = session.evaluate_live_detection()
        assert res["evaluated"] is True
        assert res["is_synthetic"] is False
        assert res["confidence"] == 0.12


@pytest.mark.asyncio
async def test_store_update_verdict_and_finalize():
    mock_coll = MagicMock()
    mock_coll.update_one = AsyncMock()
    mock_coll.find_one = AsyncMock(return_value={"status": "in_progress"})

    with patch.object(store, "get_call_logs_collection", return_value=mock_coll):
        await store.update_call_verdict("CAtest123", is_synthetic=True, confidence=0.92, reason="prueba de IA")
        assert mock_coll.update_one.called
        call_args = mock_coll.update_one.call_args[0]
        assert call_args[0] == {"call_sid": "CAtest123"}
        assert call_args[1]["$set"]["is_synthetic"] is True
        assert call_args[1]["$set"]["confidence"] == 0.92

        await store.finalize_call("CAtest123", is_synthetic=True, confidence=0.92, reason="finalizado", status="completed")
        assert mock_coll.update_one.call_count == 2
        final_args = mock_coll.update_one.call_args[0]
        assert final_args[1]["$set"]["status"] == "completed"


@pytest.mark.asyncio
async def test_runner_disconnects_when_ai_detected():
    from call_agent import runner

    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.close = AsyncMock()

    session = CallSession()
    session.call_sid = "CA_ai_call_123"

    # Mock de detección que dispara positivo para IA con alta confianza
    mock_eval = {
        "evaluated": True,
        "is_synthetic": True,
        "confidence": 0.96,
        "reason": "voz sintetica detectada",
    }

    with patch.object(session, "evaluate_live_detection", return_value=mock_eval), \
         patch("call_agent.runner.start_call", new_callable=AsyncMock), \
         patch("call_agent.runner.update_call_verdict", new_callable=AsyncMock) as mock_update, \
         patch("call_agent.runner._send_ulaw", new_callable=AsyncMock), \
         patch("call_agent.runner._listen", new_callable=AsyncMock), \
         patch("call_agent.runner._analyze_and_log", new_callable=AsyncMock):

        await runner.run_call_script(mock_ws, "MZtest", session)

        # Debe haber actualizado el veredicto en Mongo a terminated_ai_detected
        mock_update.assert_called_once()
        args = mock_update.call_args
        assert args[1]["is_synthetic"] is True
        assert args[1]["status"] == "terminated_ai_detected"

        # Debe haber cerrado el websocket
        mock_ws.close.assert_called_once()


def test_websocket_media_stream_lifecycle():
    import json
    from fastapi.testclient import TestClient
    import main as main_module

    client = TestClient(main_module.app)
    mock_coll = MagicMock()
    mock_coll.update_one = AsyncMock()
    mock_coll.find_one = AsyncMock(return_value={"status": "in_progress"})

    with patch.object(store, "get_call_logs_collection", return_value=mock_coll), \
         patch("main.run_call_script", new_callable=AsyncMock):
        with client.websocket_connect("/media-stream") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(json.dumps({
                "event": "start",
                "start": {"streamSid": "MZ_integration_test", "callSid": "CA_integration_test"}
            }))
            ws.send_text(json.dumps({
                "event": "media",
                "media": {"payload": "AAAAAAAA"}
            }))
            ws.send_text(json.dumps({"event": "stop"}))

