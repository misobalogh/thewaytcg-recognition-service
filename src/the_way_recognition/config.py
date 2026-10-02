from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    REFERENCE_IMAGE_DIR: Path = PROJECT_ROOT / "data/gt/png"
    CARD_METADATA_DIR: Path = PROJECT_ROOT / "data/json"
    MAX_IMAGE_DIM: int = Field(default=900, ge=64)
    OPENCV_THREADS: int = Field(default=1, ge=1)
    SIFT_NFEATURES: int = Field(default=1200, ge=4)
    SIFT_SHORTLIST_SIZE: int = Field(default=10, ge=2)
    SIFT_GLOBAL_CHECKS: int = Field(default=96, ge=1)
    SIFT_RATIO: float = Field(default=0.75, gt=0, lt=1)
    SIFT_MIN_INLIERS: int = Field(default=12, ge=4)
    SIFT_MIN_INLIER_RATIO: float = Field(default=0.45, gt=0, le=1)
    SIFT_MIN_COVERAGE: float = Field(default=0.04, gt=0, le=1)
    SIFT_MIN_MARGIN: float = Field(default=1.25, gt=1)
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "The Way Recognition Service"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
