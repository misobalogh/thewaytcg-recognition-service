from fastapi import HTTPException, Request

from src.the_way_recognition.core.sift import SIFTRecognitionService


def get_recognition_service(request: Request) -> SIFTRecognitionService:
    service = getattr(request.app.state, "recognition_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Recognition index is not ready")
    return service
