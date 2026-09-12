from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Contrato real del juez (scripts/check_endpoint.py del repo hackmty26).
    call_id, sample_rate y channels no se usan para decodificar (el WAV ya trae
    su propio header), se aceptan solo para no romper si el juez los manda.
    """
    audio_base64: str
    call_id: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class DetectionResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0.0, le=1.0)
