from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.the_way_recognition.api.schemas.card import CardMatch, CardRecognitionResponse
from src.the_way_recognition.config import settings
from src.the_way_recognition.core.sift import SIFTRecognitionService
from src.the_way_recognition.dependencies import get_recognition_service

router = APIRouter(prefix=settings.API_V1_PREFIX, tags=["recognition"])


@router.post("/recognize-card", response_model=CardRecognitionResponse)
async def recognize_card(
    file: UploadFile = File(...),
    service: SIFTRecognitionService = Depends(get_recognition_service),
):
    contents = await file.read()
    try:
        result = await run_in_threadpool(service.recognize, contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    candidate = result.candidate
    return CardRecognitionResponse(
        is_card=result.accepted,
        confidence=result.confidence,
        card=CardMatch(
            id=candidate.reference.card_id if result.accepted else None,
            name=candidate.reference.name if result.accepted else None,
            sift_match_score=candidate.score if candidate else 0.0,
            inliers=candidate.inliers if candidate else 0,
            inlier_ratio=candidate.inlier_ratio if candidate else 0.0,
            coverage=candidate.coverage if candidate else 0.0,
            match_margin=result.margin,
        ),
    )
