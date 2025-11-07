from fastapi import Depends
from sqlalchemy.orm import Session
from src.the_way_recognition.db.database import get_db
from src.the_way_recognition.db.repositories.card_repository import CardRepository
from src.the_way_recognition.core.ocr import OCRService
from src.the_way_recognition.core.embeddings import EmbeddingService
from src.the_way_recognition.core.matching import CardMatcher
from functools import lru_cache


# Singleton services
@lru_cache()
def get_ocr_service() -> OCRService:
    return OCRService()


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_card_repository(db: Session = Depends(get_db)) -> CardRepository:
    return CardRepository(db)


def get_card_matcher(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> CardMatcher:
    return CardMatcher(embedding_service)
