import asyncio
import json
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from call_agent.runner import run_call_script

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("media-stream")

app = FastAPI()


@app.get("/")
def read_root():
    return {"mensaje": "¡FastAPI funcionando correctamente!"}


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
                script_task = asyncio.create_task(run_call_script(websocket, stream_sid))

            elif event == "media":
                pass  # el audio del caller se procesara en la fase de buffering/mux

            elif event == "stop":
                logger.info("Stream stopped: %s", stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", stream_sid)
    finally:
        if script_task and not script_task.done():
            script_task.cancel()
