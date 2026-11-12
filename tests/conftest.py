"""Test isolation (runs before any test module imports the app).

- Copies data/rehlaguide.db and data/qdrant_db into a temp folder and points the app at
  the copies, so tests (PATCH status, analytics events, seeded Wikidata points) never
  modify the real data.
- Forces the deterministic LLM fallback so tests never call Gemini/Groq.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REAL_DB = _ROOT / "data" / "rehlaguide.db"
_REAL_QDRANT = _ROOT / "data" / "qdrant_db"

if not _REAL_DB.exists() or not _REAL_QDRANT.exists():
    raise RuntimeError(
        "Run the ingestion first:\n"
        "  .venv\\Scripts\\python.exe data/scripts/ingest_sql.py\n"
        "  .venv\\Scripts\\python.exe data/scripts/ingest_vectors.py"
    )

_TMP = Path(tempfile.mkdtemp(prefix="rehlaguide-tests-"))
shutil.copy2(_REAL_DB, _TMP / "rehlaguide.db")
shutil.copytree(_REAL_QDRANT, _TMP / "qdrant_db", ignore=shutil.ignore_patterns(".lock"))

os.environ["DATABASE_PATH"] = str(_TMP / "rehlaguide.db")
os.environ["QDRANT_PATH"] = str(_TMP / "qdrant_db")
os.environ["LLM_PROVIDER"] = "none"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _cleanup() -> None:
    try:
        from app.services.vector_store import close_qdrant_client

        close_qdrant_client()
    except Exception:
        pass
    shutil.rmtree(_TMP, ignore_errors=True)


atexit.register(_cleanup)
