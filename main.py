import asyncio
import base64
import json
import logging

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from call_agent.runner import run_call_script
from call_agent.session import CallSession
from detector.inference import decode_stereo_wav, predict_call
from detector.schema import DetectionRequest, DetectionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("media-stream")

app = FastAPI()


@app.get("/")
def read_root():
    return {"mensaje": "¡FastAPI funcionando correctamente!"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(payload: DetectionRequest):
    try:
        caller, agent, sample_rate = decode_stereo_wav(payload.audio_b64)
        is_synthetic, confidence = predict_call(caller, agent, sample_rate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el audio: {e}")
    return DetectionResponse(is_synthetic=is_synthetic, confidence=round(confidence, 4))


@app.post("/incoming-call")
async def incoming_call(request: Request):
    host = request.headers.get("host")
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="wss://{host}/media-stream" />'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    script_task: asyncio.Task | None = None
    session = CallSession()

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "connected":
                logger.info("Twilio stream connected")

            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                logger.info("Stream started: %s", stream_sid)
                script_task = asyncio.create_task(run_call_script(websocket, stream_sid, session))

            elif event == "media":
                ulaw_chunk = base64.b64decode(data["media"]["payload"])
                session.ingest_caller_ulaw(ulaw_chunk)

            elif event == "stop":
                logger.info("Stream stopped: %s", stream_sid)
                logger.info("Buffers -> caller: %d bytes (%.1fs), agent: %d bytes (%.1fs)",
                            len(session.caller_pcm), len(session.caller_pcm) / 16000,
                            len(session.agent_pcm), len(session.agent_pcm) / 16000)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", stream_sid)
    finally:
        if script_task and not script_task.done():
            script_task.cancel()
