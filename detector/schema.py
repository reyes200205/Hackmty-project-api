from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    audio_b64: str


class DetectionResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0.0, le=1.0)
