from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.the_way_recognition.api.routes import recognition
from src.the_way_recognition.config import settings
from src.the_way_recognition.core.sift import SIFTRecognitionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup rather than serve requests with an incomplete catalog.
    app.state.recognition_service = SIFTRecognitionService(settings)
    logger.info("SIFT index ready: %s cards", len(app.state.recognition_service.references))
    try:
        yield
    finally:
        app.state.recognition_service = None


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(recognition.router)


@app.get("/")
async def root():
    return {"message": "The Way Recognition Service API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
