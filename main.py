import json
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

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

            elif event == "media":
                # Loopback: reenvía el mismo audio de vuelta para escuchar el eco.
                payload = data["media"]["payload"]
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }))

            elif event == "stop":
                logger.info("Stream stopped: %s", stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", stream_sid)
