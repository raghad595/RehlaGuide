"""Central settings, read from environment variables and the project's .env file.

Every path defaults to the project layout, so nothing needs configuring for local dev.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- storage ---------------------------------------------------------------
    database_path: Path = PROJECT_ROOT / "data" / "rehlaguide.db"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant_db"
    verified_collection: str = "rehlaguide_attractions"
    unverified_collection: str = "rehlaguide_wikidata_unverified"
    embed_model: str = "intfloat/multilingual-e5-small"

    # --- LLM -------------------------------------------------------------------
    # "auto" = Gemini if GEMINI_API_KEY is set (with Groq as backup if GROQ_API_KEY is
    # set), else Groq, else the deterministic grounded fallback. "none" forces the fallback.
    # Model names were checked against both APIs on 2026-11-12; providers retire models,
    # so if you get HTTP 404, list current ones (GET .../v1beta/models, GET .../openai/v1/models).
    llm_provider: Literal["auto", "gemini", "groq", "none"] = "auto"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.5-flash"
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.2

    # --- retrieval -------------------------------------------------------------
    # Cosine similarity cut-offs for multilingual-e5-small (scores are compressed
    # into ~0.75-0.90). Calibrated in Phase 2, see app/services/rag_service.py.
    tier1_min_score: float = 0.84
    tier2_min_score: float = 0.84
    rag_top_k: int = 6
    rag_cache_ttl_seconds: int = 600
    rag_cache_maxsize: int = 512
    rag_warmup: bool = True  # load the embedding model at server startup


@lru_cache
def get_settings() -> Settings:
    return Settings()
