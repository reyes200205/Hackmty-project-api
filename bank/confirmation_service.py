import asyncio
import logging
import os
from datetime import datetime, timezone

from twilio.twiml.voice_response import VoiceResponse

from . import repository
from .confirmation_logs import log_confirmation_attempt
from .phrase_generator import generate_liveness_word
from .phrase_match import phrase_matches
from .recording_service import check_response_latency, check_voice_authenticity, download_recording, transcribe_wav
from .twilio_service import TwilioService

logger = logging.getLogger("bank")

MAX_RESULT_POLL_ATTEMPTS = 5


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

        # La palabra de vivacidad se genera y se guarda ANTES de marcar, para
        # garantizar que ya exista en la BD cuando Twilio conteste y pegue al
        # webhook de voz (que ocurre despues de esta llamada, nunca antes).
        # Si se generara despues de marcar, alguien muy rapido podria alcanzar
        # a contestar antes de que quedara guardada.
        liveness_word = generate_liveness_word()
        applied = await repository.update_transfer_status(
            transfer_id, "pending", "confirmation_pending", {"liveness_word": liveness_word},
        )
        if not applied:
            logger.warning("No se pudo mover transferencia %s a confirmation_pending (estado ya cambio)", transfer_id)
            return

        call_sid = self.twilio.place_outbound_call(phone_number, voice_url, status_url)
        logger.info("Llamada de confirmacion iniciada para transferencia %s: call_sid=%s", transfer_id, call_sid)
        await repository.update_transfer_status(
            transfer_id, "confirmation_pending", "confirmation_pending", {"call_sid": call_sid},
        )

    def build_confirmation_twiml(self, transfer: dict) -> str:
        """Pide la frase y GRABA la respuesta (no solo la transcribe) para
        poder analizar despues si la voz suena sintetica.

        Importante: NUNCA se dice la frase de confirmacion en la llamada --
        eso la volveria inutil como segundo factor (cualquiera que solo
        conteste el telefono podria repetirla al vuelo). La frase solo se
        muestra en la app; aqui solo se le pide que la diga.

        La frase SI se ve en la app con minutos de anticipacion a esta llamada
        -- tiempo de sobra para preparar un audio sintetico de antemano si se
        tiene acceso a la app. Por eso se le pide ademas una palabra que no
        existia hasta este momento (liveness_word, ver phrase_generator.py):
        nadie pudo haberla incluido en un audio preparado con anticipacion."""
        vr = VoiceResponse()
        vr.say(
            f"Para confirmar la transferencia de {transfer['amount']:.2f} pesos a {transfer['beneficiary_name']}, "
            "diga la frase de confirmacion que aparece en su aplicacion, seguida de la palabra "
            f"{transfer['liveness_word']}.",
            language="es-MX",
        )
        vr.record(
            max_length=12,
            play_beep=True,
            trim="trim-silence",
            action=f"{_base_url()}/api/transfers/webhooks/confirm/{transfer['transfer_id']}/{transfer['confirmation_token']}",
            method="POST",
        )
        vr.say("No se recibio respuesta. La transferencia no sera confirmada.", language="es-MX")
        return str(vr)

    def build_waiting_twiml(self, transfer: dict) -> str:
        """Se reproduce en cuanto termina la grabacion, mientras el analisis
        (transcripcion + deteccion de voz sintetica) corre en background."""
        vr = VoiceResponse()
        vr.say("Gracias. Estamos validando su solicitud, espere un momento por favor.", language="es-MX")
        vr.pause(length=3)
        vr.redirect(
            f"{_base_url()}/api/transfers/webhooks/result/{transfer['transfer_id']}/{transfer['confirmation_token']}?attempt=1",
            method="POST",
        )
        return str(vr)

    def build_result_twiml(self, transfer: dict, attempt: int) -> str:
        vr = VoiceResponse()
        status = transfer["status"]

        if status == "completed":
            vr.say("Listo, su transferencia ha sido confirmada y procesada exitosamente.", language="es-MX")
        elif status == "rejected":
            reason = transfer.get("failure_reason", "") or ""
            if "sintetic" in reason:
                vr.say(
                    "No pudimos verificar que la voz sea autentica. La transferencia no sera procesada.",
                    language="es-MX",
                )
            elif "tardo" in reason:
                vr.say(
                    "Tardo demasiado en responder. Por seguridad, la transferencia no sera procesada.",
                    language="es-MX",
                )
            elif "palabra de verificacion" in reason:
                vr.say(
                    "No pudimos confirmar la palabra de verificacion. La transferencia no sera procesada.",
                    language="es-MX",
                )
            else:
                vr.say("La frase no coincide. La transferencia no sera procesada.", language="es-MX")
        elif status == "failed":
            vr.say("Ocurrio un error al procesar su solicitud.", language="es-MX")
        elif attempt < MAX_RESULT_POLL_ATTEMPTS:
            vr.pause(length=2)
            vr.redirect(
                f"{_base_url()}/api/transfers/webhooks/result/{transfer['transfer_id']}/{transfer['confirmation_token']}"
                f"?attempt={attempt + 1}",
                method="POST",
            )
        else:
            vr.say(
                "La validacion esta tomando mas tiempo de lo esperado. Le notificaremos el resultado en la aplicacion.",
                language="es-MX",
            )
        return str(vr)

    async def handle_recording(self, transfer_id: str, recording_url: str, call_sid: str) -> None:
        """Corre en background: descarga la grabacion, transcribe, valida la
        frase Y valida que la voz no sea sintetica (reutilizando el
        clasificador acustico real de deteccion de deepfakes), y decide si
        completar la transferencia. Todo queda registrado en el log."""
        transfer = await repository.get_transfer(transfer_id)
        if transfer is None or transfer["status"] != "confirmation_pending":
            logger.warning("Grabacion recibida para transferencia %s en estado inesperado", transfer_id)
            return

        transcript = ""
        try:
            wav_bytes = await download_recording(recording_url)
            transcript = await transcribe_wav(wav_bytes)
            phrase_ok = phrase_matches(transfer["confirmation_phrase"], transcript)
            liveness_ok = phrase_matches(transfer["liveness_word"], transcript)
            is_synthetic, voice_confidence, voice_reasoning = await asyncio.to_thread(
                check_voice_authenticity, wav_bytes,
            )
            too_slow, latency_s = await asyncio.to_thread(check_response_latency, wav_bytes)
            logger.info(
                "Transferencia %s: analisis de voz -> %s; tiempo antes de hablar=%.1fs (%s)",
                transfer_id, voice_reasoning, latency_s, "SOSPECHOSO" if too_slow else "normal",
            )
        except Exception:
            logger.exception("Fallo el analisis de la grabacion para transferencia %s", transfer_id)
            await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "failed",
                {"failure_reason": "error al analizar la grabacion"},
            )
            await log_confirmation_attempt(
                transfer_id, call_sid, recording_url, transcript, False, None, None, "failed", "error de analisis",
            )
            return

        if not phrase_ok:
            await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "rejected",
                {"heard": transcript, "failure_reason": "frase incorrecta"},
            )
            decision, reason = "rejected", "frase incorrecta"
        elif not liveness_ok:
            # La frase se pudo haber sabido con anticipacion (se ve en la app);
            # la palabra de vivacidad no, asi que fallar aqui es la señal mas
            # fuerte de que la respuesta se preparo antes de esta llamada.
            await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "rejected",
                {"heard": transcript, "failure_reason": "no dijo la palabra de verificacion"},
            )
            decision, reason = "rejected", "no dijo la palabra de verificacion"
        elif is_synthetic:
            await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "rejected",
                {"heard": transcript, "failure_reason": "voz sintetica sospechosa", "voice_confidence": voice_confidence},
            )
            decision, reason = "rejected", "voz sintetica sospechosa"
        elif too_slow:
            await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "rejected",
                {"heard": transcript, "failure_reason": "tardo demasiado en responder", "response_latency_s": latency_s},
            )
            decision, reason = "rejected", f"tardo demasiado en responder ({latency_s:.1f}s)"
        else:
            try:
                new_balance = await repository.atomic_debit_and_log(
                    transfer["customer_email"], transfer["amount"], transfer["concept"], transfer_id,
                )
                await repository.update_transfer_status(
                    transfer_id, "confirmation_pending", "completed",
                    {
                        "completed_at": datetime.now(timezone.utc),
                        "resulting_balance": new_balance,
                        "heard": transcript,
                        "voice_confidence": voice_confidence,
                    },
                )
                decision, reason = "completed", None
            except repository.InsufficientFundsError:
                await repository.update_transfer_status(
                    transfer_id, "confirmation_pending", "failed",
                    {"failure_reason": "saldo insuficiente al confirmar"},
                )
                decision, reason = "failed", "saldo insuficiente"

        await log_confirmation_attempt(
            transfer_id, call_sid, recording_url, transcript, phrase_ok, is_synthetic, voice_confidence, decision, reason,
        )
        logger.info(
            "Transferencia %s -> %s (frase_ok=%s, voz_sintetica=%s, confianza=%.2f)",
            transfer_id, decision, phrase_ok, is_synthetic, voice_confidence or 0.0,
        )

    async def handle_call_status(self, transfer_id: str, call_status: str) -> None:
        if call_status in ("no-answer", "busy", "failed", "canceled"):
            applied = await repository.update_transfer_status(
                transfer_id, "confirmation_pending", "failed", {"failure_reason": f"llamada: {call_status}"},
            )
            if applied:
                logger.info("Transferencia %s marcada como failed (llamada: %s)", transfer_id, call_status)
