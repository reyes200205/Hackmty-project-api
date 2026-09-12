import logging
import os
from datetime import datetime, timezone

from twilio.twiml.voice_response import Gather, VoiceResponse

from . import repository
from .phrase_match import phrase_matches
from .twilio_service import TwilioService

logger = logging.getLogger("bank")


def _base_url() -> str:
    return os.environ["PUBLIC_BASE_URL"].rstrip("/")


class TransferConfirmationService:
    def __init__(self, twilio: TwilioService | None = None):
        self.twilio = twilio or TwilioService()

    async def start_confirmation(self, transfer: dict, phone_number: str) -> None:
        transfer_id = transfer["transfer_id"]
        token = transfer["confirmation_token"]
        voice_url = f"{_base_url()}/api/transfers/webhooks/voice/{transfer_id}/{token}"
        status_url = f"{_base_url()}/api/transfers/webhooks/status/{transfer_id}/{token}"

        call_sid = self.twilio.place_outbound_call(phone_number, voice_url, status_url)
        logger.info("Llamada de confirmacion iniciada para transferencia %s: call_sid=%s", transfer_id, call_sid)

        applied = await repository.update_transfer_status(
            transfer_id, "pending", "confirmation_pending", {"call_sid": call_sid},
        )
        if not applied:
            logger.warning("No se pudo mover transferencia %s a confirmation_pending (estado ya cambio)", transfer_id)

    def build_gather_twiml(self, transfer: dict) -> str:
        vr = VoiceResponse()
        gather = Gather(
            input="speech",
            language="es-MX",
            action=f"{_base_url()}/api/transfers/webhooks/confirm/{transfer['transfer_id']}/{transfer['confirmation_token']}",
            method="POST",
            speech_timeout="auto",
        )
        gather.say(
            f"Para confirmar la transferencia de {transfer['amount']:.2f} pesos a {transfer['beneficiary_name']}, "
            f"diga: {transfer['confirmation_phrase']}.",
            language="es-MX",
        )
        vr.append(gather)
        vr.say("No se recibio respuesta. La transferencia no sera confirmada.", language="es-MX")
        return str(vr)

    async def handle_confirmation_result(self, transfer_id: str, speech_result: str) -> dict:
        transfer = await repository.get_transfer(transfer_id)
        if transfer is None or transfer["status"] != "confirmation_pending":
            logger.warning("Webhook de confirmacion ignorado para %s (estado actual: %s)",
                            transfer_id, transfer["status"] if transfer else "no existe")
            return {"status": "ignored"}

        if phrase_matches(transfer["confirmation_phrase"], speech_result):
            try:
                new_balance = await repository.atomic_debit_and_log(
                    transfer["customer_email"], transfer["amount"], transfer["concept"], transfer_id,
                )
            except repository.InsufficientFundsError:
                await repository.update_transfer_status(
                    transfer_id, "confirmation_pending", "failed",
                    {"failure_reason": "saldo insuficiente al momento de confirmar"},
                )
                logger.error("Transferencia %s fallo por saldo insuficiente al confirmar", transfer_id)
                return {"status": "failed"}

            await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "completed",
                {"completed_at": datetime.now(timezone.utc), "resulting_balance": new_balance, "heard": speech_result},
            )
            logger.info("Transferencia %s confirmada y completada, nuevo saldo=%.2f", transfer_id, new_balance)
            return {"status": "completed"}

        await repository.update_transfer_status(
            transfer_id, "confirmation_pending", "rejected", {"heard": speech_result},
        )
        logger.info("Transferencia %s rechazada, frase no coincide (escuchado: %r)", transfer_id, speech_result)
        return {"status": "rejected"}

    async def handle_call_status(self, transfer_id: str, call_status: str) -> None:
        if call_status in ("no-answer", "busy", "failed", "canceled"):
            applied = await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "failed", {"failure_reason": f"llamada: {call_status}"},
            )
            if applied:
                logger.info("Transferencia %s marcada como failed (llamada: %s)", transfer_id, call_status)
