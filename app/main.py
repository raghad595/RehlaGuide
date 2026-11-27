import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.analytics import router as analytics_router
from app.api.v1.attractions import router as attractions_router
from app.api.v1.chat import router as chat_router
from app.db.database import init_db
from app.core.config import get_settings
from app.services.vector_store import close_qdrant_client, embed_query, get_qdrant_client


log = logging.getLogger("rehlaguide")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # creates missing tables (e.g. analytics_events); never drops data
    if get_settings().rag_warmup:
        try:  # load the embedding model + open Qdrant now, not on the first user request
            embed_query("warm-up")
            get_qdrant_client()
        except Exception:
            log.exception("RAG warm-up failed; it will be retried on the first chat request")
    yield
    close_qdrant_client()  # release the local Qdrant folder lock


app = FastAPI(
    title="RehlaGuide API",
    description="Bilingual (EN/AR) Egyptian tourism & itinerary planning assistant",
    version="0.2.0",
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
app.include_router(chat_router)
app.include_router(analytics_router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
