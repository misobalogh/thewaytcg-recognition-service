from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.the_way_recognition.config import settings
from src.the_way_recognition.api.routes import recognition
from src.the_way_recognition.db.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recognition.router)


@app.get("/")
async def root():
    return {"message": "The Way Recognition Service API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
