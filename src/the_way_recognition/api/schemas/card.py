from typing import Literal

from pydantic import BaseModel, Field


class CardMatch(BaseModel):
    id: str | None = None
    name: str | None = None
    sift_match_score: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    inliers: int = Field(default=0, ge=0)
    inlier_ratio: float = Field(default=0.0, ge=0, le=1)
    coverage: float = Field(default=0.0, ge=0, le=1)
    match_margin: float | None = Field(default=None, ge=1, allow_inf_nan=False)


class CardRecognitionResponse(BaseModel):
    is_card: bool
    confidence: Literal["high", "medium", "low", "none"]
    card: CardMatch
