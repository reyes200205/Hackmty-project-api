from datetime import datetime

from pydantic import BaseModel, Field


class AccountOut(BaseModel):
    account_number: str
    account_type: str
    balance: float
    currency: str
    status: str
    opened_at: datetime


class TransactionOut(BaseModel):
    transaction_id: str
    type: str
    concept: str
    amount: float
    resulting_balance: float
    date: datetime
    status: str
    related_transfer_id: str | None = None


class PaginatedTransactions(BaseModel):
    items: list[TransactionOut]
    total: int
    limit: int
    offset: int


class BeneficiaryOut(BaseModel):
    beneficiary_id: str
    name: str
    account_number: str
    bank_name: str


class TransferCreateRequest(BaseModel):
    beneficiary_id: str
    amount: float = Field(gt=0)
    concept: str = Field(min_length=1, max_length=140)


class TransferOut(BaseModel):
    transfer_id: str
    status: str
    amount: float
    concept: str
    beneficiary_name: str
    created_at: datetime
    updated_at: datetime
    confirmation_phrase: str | None = None
