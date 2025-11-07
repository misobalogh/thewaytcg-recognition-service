from pydantic_settings import BaseSettings
# import torch
from functools import lru_cache


class Settings(BaseSettings):
    # Image processing
    MAX_IMAGE_DIM: int = 1000

    # Model settings
    DEVICE: str = "cpu"
    CLIP_MODEL: str = "ViT-B/32"

    # OCR settings
    TESSERACT_LANG: str = "slk"
    TESSERACT_CONFIG: str = "--psm 6"

    # Confidence thresholds
    CONFIDENCE_HIGH: float = 0.75
    CONFIDENCE_MEDIUM: float = 0.5
    CONFIDENCE_LOW: float = 0.3

    # Score weights
    TEXT_WEIGHT: float = 0.4
    EMBED_WEIGHT: float = 0.6
    CONSENSUS_BOOST: float = 0.15   

    # Database
    DATABASE_URL: str = "sqlite:///./cards.db"

    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "The Way Recognition Service"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
