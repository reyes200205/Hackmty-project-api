from bank import repository


async def test_status_transition_applies_when_expected_matches(seeded_customer):
    transfer = await repository.create_transfer(
        seeded_customer["email"],
        {"beneficiary_id": "x", "name": "Y"},
        100.0, "concepto", "FRASE",
    )
    applied = await repository.update_transfer_status(transfer["transfer_id"], "pending", "confirmation_pending")
    assert applied is True

    fresh = await repository.get_transfer(transfer["transfer_id"])
    assert fresh["status"] == "confirmation_pending"


async def test_status_transition_rejected_when_state_already_changed(seeded_customer):
    """Simula un reintento de webhook de Twilio: la segunda vez no debe re-aplicar."""
    transfer = await repository.create_transfer(
        seeded_customer["email"],
        {"beneficiary_id": "x", "name": "Y"},
        100.0, "concepto", "FRASE",
    )
    first = await repository.update_transfer_status(transfer["transfer_id"], "pending", "confirmation_pending")
    second = await repository.update_transfer_status(transfer["transfer_id"], "pending", "confirmation_pending")

    assert first is True
    assert second is False


async def test_status_transition_fails_for_unknown_transfer():
    applied = await repository.update_transfer_status("no-existe", "pending", "confirmation_pending")
    assert applied is False
