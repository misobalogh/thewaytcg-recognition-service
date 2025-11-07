from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from src.the_way_recognition.api.schemas.card import CardRecognitionResponse
from src.the_way_recognition.core.ocr import OCRService
from src.the_way_recognition.core.matching import CardMatcher
from src.the_way_recognition.db.repositories.card_repository import CardRepository
from src.the_way_recognition.utils.image import preprocess_image
from src.the_way_recognition.dependencies import (
    get_ocr_service,
    get_card_matcher,
    get_card_repository
)
from src.the_way_recognition.config import settings

router = APIRouter(prefix=settings.API_V1_PREFIX, tags=["recognition"])

@router.post("/recognize-card", response_model=CardRecognitionResponse)
async def recognize_card(
    file: UploadFile = File(...),
    ocr_service: OCRService = Depends(get_ocr_service),
    card_matcher: CardMatcher = Depends(get_card_matcher),
    card_repo: CardRepository = Depends(get_card_repository)
):
    try:
        image = await preprocess_image(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Extract text using OCR
    ocr_text = ocr_service.extract_text(image)

    # Get all cards from database
    cards = card_repo.get_all()

    if not cards:
        raise HTTPException(status_code=500, detail="No cards found in database")

    # Find best text match
    best_text_card, best_text_score = card_matcher.get_best_text_match(ocr_text, cards)

    # Find best embedding match
    best_emb_card, best_emb_score = card_matcher.get_best_embedding_match(image, cards)

    # Select best overall match
    result = card_matcher.select_best_match(
        best_text_card, best_text_score,
        best_emb_card, best_emb_score
    )

    return CardRecognitionResponse(
        is_card=result.is_card,
        confidence=result.confidence,
        card={
            "name": result.card.name if result.card else None,
            "text_match_score": round(result.text_score, 4),
            "embedding_match_score": round(result.embedding_score, 4),
        }
    )
