import pytest

from bank import repository


async def test_atomic_debit_reduces_balance_and_logs_transaction(seeded_customer):
    new_balance = await repository.atomic_debit_and_log(
        seeded_customer["email"], 250.0, "prueba", "transfer-xyz",
    )
    assert new_balance == 750.0

    account = await repository.get_account_by_email(seeded_customer["email"])
    assert account["balance"] == 750.0

    items, total = await repository.list_transactions(seeded_customer["email"], 10, 0)
    assert total >= 1
    assert items[0]["related_transfer_id"] == "transfer-xyz"
    assert items[0]["amount"] == -250.0


async def test_atomic_debit_raises_on_insufficient_funds(seeded_customer):
    with pytest.raises(repository.InsufficientFundsError):
        await repository.atomic_debit_and_log(seeded_customer["email"], 999999.0, "prueba", "transfer-abc")

    account = await repository.get_account_by_email(seeded_customer["email"])
    assert account["balance"] == 1000.0  # no debio cambiar
