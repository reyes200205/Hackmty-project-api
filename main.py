import asyncio
import base64
import json
import logging
from xml.sax.saxutils import escape as xml_escape

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from call_agent.runner import run_call_script
from call_agent.session import CallSession
from customers.auth import verify_password
from customers.deps import get_current_customer
from customers.lookup import find_by_email, find_by_phone
from customers.schema import CustomerOut, LoginRequest, LoginResponse
from customers.tokens import create_access_token
from detector.inference import decode_stereo_wav, predict_call
from detector.schema import DetectionRequest, DetectionResponse
from bank.router import router as bank_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("media-stream")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(bank_router)


@app.get("/")
def read_root():
    return {"mensaje": "¡FastAPI funcionando correctamente!"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(payload: DetectionRequest):
    try:
        caller, agent, sample_rate = decode_stereo_wav(payload.audio_b64)
        is_synthetic, confidence = predict_call(caller, agent, sample_rate)
    except Exception as e:
        logger.exception("Error al procesar el audio en /detect: %s", e)
        raise HTTPException(status_code=400, detail=f"Error al procesar el audio: {e}")
    return DetectionResponse(is_synthetic=is_synthetic, confidence=round(confidence, 4))


@app.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest):
    customer = await find_by_email(payload.email)
    if not customer or not verify_password(payload.password, customer["password_hash"]):
        raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos")

    token = create_access_token(customer["email"], customer["full_name"])
    return LoginResponse(
        access_token=token,
        customer=CustomerOut(
            full_name=customer["full_name"],
            email=customer["email"],
            phone_number=customer["phone_number"],
        ),
    )


@app.get("/auth/me", response_model=CustomerOut)
async def me(customer: dict = Depends(get_current_customer)):
    return CustomerOut(
        full_name=customer["full_name"],
        email=customer["email"],
        phone_number=customer["phone_number"],
    )


@app.post("/incoming-call")
async def incoming_call(request: Request):
    host = request.headers.get("host")
    form = await request.form()
    caller_number = form.get("From")

    customer = await find_by_phone(caller_number) if caller_number else None
    first_name = customer["full_name"].split()[0] if customer else ""

    param_tag = f'<Parameter name="customerName" value="{xml_escape(first_name)}" />' if first_name else ""
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="wss://{host}/media-stream">{param_tag}</Stream>'
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
                session.call_sid = data["start"].get("callSid")
                custom_params = data["start"].get("customParameters", {})
                session.customer_name = custom_params.get("customerName") or None
                logger.info("Stream started: %s (cliente: %s)", stream_sid, session.customer_name or "desconocido")
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
