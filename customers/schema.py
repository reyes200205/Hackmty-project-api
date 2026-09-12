from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class CustomerOut(BaseModel):
    full_name: str
    email: str
    phone_number: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    customer: CustomerOut
