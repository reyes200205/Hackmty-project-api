from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

from customers.deps import get_current_customer

from . import repository
from .confirmation_service import CONFIRMATION_PHRASE, TransferConfirmationService
from .schema import (
    AccountOut,
    BeneficiaryOut,
    PaginatedTransactions,
    TransactionOut,
    TransferCreateRequest,
    TransferOut,
)
from .security import verify_twilio_signature

router = APIRouter(prefix="/api", tags=["bank"])
confirmation_service = TransferConfirmationService()


@router.get("/account", response_model=AccountOut)
async def get_account(customer: dict = Depends(get_current_customer)):
    account = await repository.get_account_by_email(customer["email"])
    if account is None:
        raise HTTPException(status_code=404, detail="Cuenta no encontrada")
    return AccountOut(**account)


@router.get("/account/transactions", response_model=PaginatedTransactions)
async def get_transactions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    customer: dict = Depends(get_current_customer),
):
    items, total = await repository.list_transactions(customer["email"], limit, offset)
    return PaginatedTransactions(
        items=[TransactionOut(**t) for t in items], total=total, limit=limit, offset=offset,
    )


@router.get("/beneficiaries", response_model=list[BeneficiaryOut])
async def get_beneficiaries(customer: dict = Depends(get_current_customer)):
    items = await repository.list_beneficiaries(customer["email"])
    return [BeneficiaryOut(**b) for b in items]


@router.post("/transfers", response_model=TransferOut, status_code=201)
async def create_transfer(payload: TransferCreateRequest, customer: dict = Depends(get_current_customer)):
    account = await repository.get_account_by_email(customer["email"])
    if account is None:
        raise HTTPException(status_code=404, detail="Cuenta no encontrada")
    if account["balance"] < payload.amount:
        raise HTTPException(status_code=422, detail="Saldo insuficiente")

    beneficiary = await repository.get_beneficiary(customer["email"], payload.beneficiary_id)
    if beneficiary is None:
        raise HTTPException(status_code=404, detail="Beneficiario no encontrado")

    transfer = await repository.create_transfer(
        customer_email=customer["email"],
        beneficiary=beneficiary,
        amount=payload.amount,
        concept=payload.concept,
        confirmation_phrase=CONFIRMATION_PHRASE,
    )

    try:
        await confirmation_service.start_confirmation(transfer, customer["phone_number"])
    except Exception as e:
        await repository.update_transfer_status(
            transfer["transfer_id"], "pending", "failed", {"failure_reason": str(e)},
        )
        raise HTTPException(status_code=502, detail=f"No se pudo iniciar la llamada de confirmacion: {e}")

    transfer["status"] = "confirmation_pending"
    return TransferOut(**transfer)


@router.get("/transfers/{transfer_id}", response_model=TransferOut)
async def get_transfer(transfer_id: str, customer: dict = Depends(get_current_customer)):
    transfer = await repository.get_transfer_owned_by(customer["email"], transfer_id)
    if transfer is None:
        raise HTTPException(status_code=404, detail="Transferencia no encontrada")
    return TransferOut(**transfer)


@router.get("/transfers", response_model=list[TransferOut])
async def list_transfers(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    customer: dict = Depends(get_current_customer),
):
    items = await repository.list_transfers(customer["email"], limit, offset)
    return [TransferOut(**t) for t in items]


@router.post("/transfers/webhooks/voice/{transfer_id}/{token}")
async def voice_webhook(transfer_id: str, token: str, params: dict = Depends(verify_twilio_signature)):
    transfer = await repository.get_transfer(transfer_id)
    if transfer is None or transfer.get("confirmation_token") != token:
        raise HTTPException(status_code=404, detail="not found")
    twiml = confirmation_service.build_gather_twiml(transfer)
    return Response(content=twiml, media_type="application/xml")


@router.post("/transfers/webhooks/confirm/{transfer_id}/{token}")
async def confirm_webhook(transfer_id: str, token: str, params: dict = Depends(verify_twilio_signature)):
    transfer = await repository.get_transfer(transfer_id)
    if transfer is None or transfer.get("confirmation_token") != token:
        raise HTTPException(status_code=404, detail="not found")

    speech_result = params.get("SpeechResult", "")
    result = await confirmation_service.handle_confirmation_result(transfer_id, speech_result)

    message = {
        "completed": "Gracias, su transferencia ha sido confirmada y procesada.",
        "rejected": "La frase no coincide, la transferencia no sera procesada.",
        "failed": "Ocurrio un error al procesar la transferencia.",
        "ignored": "Esta transferencia ya no esta pendiente de confirmacion.",
    }.get(result["status"], "Proceso finalizado.")

    vr = VoiceResponse()
    vr.say(message, language="es-MX")
    return Response(content=str(vr), media_type="application/xml")


@router.post("/transfers/webhooks/status/{transfer_id}/{token}")
async def status_webhook(transfer_id: str, token: str, params: dict = Depends(verify_twilio_signature)):
    transfer = await repository.get_transfer(transfer_id)
    if transfer is None or transfer.get("confirmation_token") != token:
        raise HTTPException(status_code=404, detail="not found")

    await confirmation_service.handle_call_status(transfer_id, params.get("CallStatus", ""))
    return Response(status_code=204)
