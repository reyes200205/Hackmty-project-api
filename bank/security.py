import os

from fastapi import HTTPException, Request
from twilio.request_validator import RequestValidator


def _reconstructed_url(request: Request) -> str:
    """ngrok termina TLS y le pasa la peticion a uvicorn como http; Twilio firmo
    la URL https que si llamo, asi que hay que reconstruirla con el esquema real
    (X-Forwarded-Proto) o la validacion de firma siempre fallaria."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    url = f"{scheme}://{host}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
    return url


async def verify_twilio_signature(request: Request) -> dict:
    validator = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])
    signature = request.headers.get("X-Twilio-Signature", "")
    form = await request.form()
    params = dict(form)
    url = _reconstructed_url(request)

    if not validator.validate(url, params, signature):
        raise HTTPException(status_code=403, detail="Firma de Twilio invalida")
    return params
