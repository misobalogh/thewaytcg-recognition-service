from pydantic import BaseModel, Field
from typing import Optional

class CardMatch(BaseModel):
    name: Optional[str] = None
    text_match_score: float = Field(..., ge=0.0, le=1.0)
    embedding_match_score: float = Field(..., ge=0.0, le=1.0)

class CardRecognitionResponse(BaseModel):
    is_card: bool
    confidence: str = Field(..., pattern="^(high|medium|low|none)$")
    card: CardMatch

    class Config:
        json_schema_extra = {
            "example": {
                "is_card": True,
                "confidence": "high",
                "card": {
                    "name": "Example Card",
                    "text_match_score": 0.87,
                    "embedding_match_score": 0.92
                }
            }
        }
