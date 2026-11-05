from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.attractions import router as attractions_router
from app.db.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RehlaGuide API",
    description="Bilingual (EN/AR) Egyptian tourism & itinerary planning assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attractions_router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
