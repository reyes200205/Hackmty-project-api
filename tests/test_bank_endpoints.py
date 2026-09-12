from unittest.mock import patch

from bank import repository
from bank.confirmation_service import TransferConfirmationService


async def test_create_transfer_requires_auth(client):
    resp = await client.post("/api/transfers", json={"beneficiary_id": "x", "amount": 10, "concept": "y"})
    assert resp.status_code == 401


async def test_create_transfer_starts_confirmation_pending(client, auth_headers, seeded_customer):
    with patch("bank.twilio_service.TwilioService.place_outbound_call", return_value="CAfake123"):
        resp = await client.post(
            "/api/transfers",
            headers=auth_headers,
            json={"beneficiary_id": "pytest-benef-1", "amount": 100.0, "concept": "prueba pytest"},
        )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "confirmation_pending"

    transfer = await repository.get_transfer(body["transfer_id"])
    assert transfer["call_sid"] == "CAfake123"


async def test_insufficient_balance_rejected_before_calling_twilio(client, auth_headers, seeded_customer):
    with patch("bank.twilio_service.TwilioService.place_outbound_call") as mocked_call:
        resp = await client.post(
            "/api/transfers",
            headers=auth_headers,
            json={"beneficiary_id": "pytest-benef-1", "amount": 999999.0, "concept": "prueba"},
        )
    assert resp.status_code == 422
    mocked_call.assert_not_called()


async def test_unknown_beneficiary_rejected(client, auth_headers, seeded_customer):
    resp = await client.post(
        "/api/transfers",
        headers=auth_headers,
        json={"beneficiary_id": "no-existe", "amount": 10.0, "concept": "prueba"},
    )
    assert resp.status_code == 404


async def test_correct_phrase_completes_transfer(seeded_customer):
    transfer = await repository.create_transfer(
        seeded_customer["email"],
        {"beneficiary_id": "pytest-benef-1", "name": "Beneficiario de Prueba"},
        100.0, "prueba", "CONFIRMO LA TRANSFERENCIA",
    )
    await repository.update_transfer_status(transfer["transfer_id"], "pending", "confirmation_pending")

    result = await TransferConfirmationService().handle_confirmation_result(
        transfer["transfer_id"], "confirmo la transferencia",
    )

    assert result["status"] == "completed"
    account = await repository.get_account_by_email(seeded_customer["email"])
    assert account["balance"] == 900.0


async def test_wrong_phrase_rejects_transfer_without_touching_balance(seeded_customer):
    transfer = await repository.create_transfer(
        seeded_customer["email"],
        {"beneficiary_id": "pytest-benef-1", "name": "Beneficiario de Prueba"},
        100.0, "prueba", "CONFIRMO LA TRANSFERENCIA",
    )
    await repository.update_transfer_status(transfer["transfer_id"], "pending", "confirmation_pending")

    result = await TransferConfirmationService().handle_confirmation_result(
        transfer["transfer_id"], "cancelo la transferencia",
    )

    assert result["status"] == "rejected"
    account = await repository.get_account_by_email(seeded_customer["email"])
    assert account["balance"] == 1000.0


async def test_other_customer_cannot_see_foreign_transfer(client, seeded_customer, other_customer):
    transfer = await repository.create_transfer(
        seeded_customer["email"],
        {"beneficiary_id": "pytest-benef-1", "name": "Beneficiario de Prueba"},
        50.0, "prueba", "CONFIRMO LA TRANSFERENCIA",
    )

    resp = await client.post(
        "/auth/login", json={"email": other_customer["email"], "password": other_customer["password"]},
    )
    other_token = resp.json()["access_token"]

    resp2 = await client.get(
        f"/api/transfers/{transfer['transfer_id']}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp2.status_code == 404
