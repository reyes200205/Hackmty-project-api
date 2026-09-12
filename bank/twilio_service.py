import os

from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    return _client


class TwilioService:
    """Envoltura delgada sobre la API de Twilio para originar llamadas salientes.
    Las credenciales solo se leen de variables de entorno, nunca hardcodeadas."""

    def place_outbound_call(self, to_number: str, voice_webhook_url: str, status_webhook_url: str) -> str:
        call = _get_client().calls.create(
            to=to_number,
            from_=os.environ["TWILIO_FROM_NUMBER"],
            url=voice_webhook_url,
            method="POST",
            status_callback=status_webhook_url,
            status_callback_event=["completed", "no-answer", "busy", "failed", "canceled"],
            status_callback_method="POST",
        )
        return call.sid
