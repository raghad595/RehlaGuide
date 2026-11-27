"""Phase 1 data verification checks. Each check returns a list of error strings (empty = pass).

Run:  PYTHONPATH=. python data/scripts/verify_phase1_data.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.database import engine
from app.models import Attraction, OpeningHour, TicketPrice
from app.services.vector_store import close_qdrant_client, get_qdrant_client

DATA_DIR = Path(__file__).resolve().parents[1]
CHUNKS_JSON = DATA_DIR / "processed_chunks.json"
QDRANT_PATH = get_settings().qdrant_path
COLLECTION = get_settings().verified_collection

LAT_RANGE = (22.0, 31.8)
LON_RANGE = (24.5, 37.0)
HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
ARABIC_RE = re.compile(r"[؀-ۿ]")
VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
CHUNK_TYPES = {"history_and_significance", "highlights_and_faq", "visitor_guidance_and_access"}


def _load_db():
    with Session(engine) as s:
        return (
            s.exec(select(Attraction)).all(),
            s.exec(select(OpeningHour)).all(),
            s.exec(select(TicketPrice)).all(),
        )


def csv_attraction_count() -> int:
    return len(pd.read_csv(DATA_DIR / "raw_csv" / "attractions.csv", dtype=str))


def check_referential_integrity() -> List[str]:
    attractions, hours, prices = _load_db()
    errors: List[str] = []
    expected = csv_attraction_count()
    if len(attractions) != expected:
        errors.append(f"attractions: {len(attractions)} rows in SQLite, {expected} in CSV")
    ids = {a.attraction_id for a in attractions}
    for h in hours:
        if h.attraction_id not in ids:
            errors.append(f"opening_hours id={h.id}: orphan attraction_id {h.attraction_id}")
    for t in prices:
        if t.attraction_id not in ids:
            errors.append(f"ticket_prices id={t.id}: orphan attraction_id {t.attraction_id}")
    hour_ids = {h.attraction_id for h in hours}
    price_ids = {t.attraction_id for t in prices}
    errors += [f"{aid}: no opening_hours rows" for aid in sorted(ids - hour_ids)]
    errors += [f"{aid}: no ticket_prices rows" for aid in sorted(ids - price_ids)]
    for a in attractions:
        if a.is_open is not True or a.closure_reason is not None:
            errors.append(f"{a.attraction_id}: expected is_open=True / closure_reason=None after ingest")
    return errors


def check_coordinates() -> List[str]:
    attractions, _, _ = _load_db()
    errors = []
    for a in attractions:
        if not (LAT_RANGE[0] <= a.latitude <= LAT_RANGE[1]):
            errors.append(f"{a.attraction_id}: latitude {a.latitude} outside {LAT_RANGE}")
        if not (LON_RANGE[0] <= a.longitude <= LON_RANGE[1]):
            errors.append(f"{a.attraction_id}: longitude {a.longitude} outside {LON_RANGE}")
    return errors


def check_hours_and_prices() -> List[str]:
    _, hours, prices = _load_db()
    errors = []
    primary = Counter(h.attraction_id for h in hours if h.is_primary_session)
    for h in hours:
        tag = f"opening_hours id={h.id} ({h.attraction_id} / {h.area_or_session})"
        closed = h.opening_hour_raw.strip().lower() == "closed"
        unspecified_days = h.opening_days_raw.strip().lower() == "not specified"
        if not closed and not unspecified_days and not h.open_days_list:
            errors.append(f"{tag}: empty open_days_list for '{h.opening_days_raw}'")
        if set(h.open_days_list) - VALID_DAYS:
            errors.append(f"{tag}: invalid day codes {h.open_days_list}")
        for field in ("open_time_24h", "close_time_24h", "last_entry_24h"):
            value = getattr(h, field)
            if value is not None and not HHMM_RE.match(value):
                errors.append(f"{tag}: {field}={value!r} not HH:MM")
        if not closed and (h.open_time_24h is None or h.close_time_24h is None):
            errors.append(f"{tag}: open/close not parsed from '{h.opening_hour_raw}'/'{h.closing_hour_raw}'")
    for aid, n in primary.items():
        if n != 1:
            errors.append(f"{aid}: {n} primary sessions (expected 1)")
    for t in prices:
        for field in ("price_locals_egp", "price_foreigners_egp", "price_children_egp"):
            v = getattr(t, field)
            if v is not None and v < 0:
                errors.append(f"ticket_prices id={t.id}: negative {field}={v}")
        for raw, num in ((t.price_for_locals_raw, t.price_locals_egp),
                         (t.price_for_foreigners_raw, t.price_foreigners_egp)):
            if "EGP" in raw.upper() and not raw.lower().startswith("free") and num is None:
                errors.append(f"ticket_prices id={t.id}: EGP figure in '{raw}' not extracted")
    return errors


def _parity_errors(rows: List[dict], source: str, ids: set) -> List[str]:
    errors = []
    per = Counter((r["attraction_id"], r["language"]) for r in rows)
    for aid in sorted(ids):
        for lang in ("en", "ar"):
            if per[(aid, lang)] != 3:
                errors.append(f"{source}: {aid} [{lang}] has {per[(aid, lang)]} chunks (expected 3)")
    for r in rows:
        if r["chunk_type"] not in CHUNK_TYPES:
            errors.append(f"{source}: unknown chunk_type {r['chunk_type']}")
        if r["language"] == "ar" and not ARABIC_RE.search(r["text"].split("\n\n", 1)[-1]):
            errors.append(f"{source}: Arabic chunk {r['attraction_id']}/{r['chunk_type']} has no Arabic script")
    return errors


def check_chunks_json() -> List[str]:
    if not CHUNKS_JSON.exists():
        return [f"{CHUNKS_JSON} missing - run ingest_vectors.py"]
    attractions, _, _ = _load_db()
    rows = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    return _parity_errors(rows, "processed_chunks.json", {a.attraction_id for a in attractions})


def check_qdrant() -> List[str]:
    if not QDRANT_PATH.exists():
        return [f"{QDRANT_PATH} missing - run ingest_vectors.py"]
    client = get_qdrant_client()  # shared per-process client (local mode allows only one)
    if not client.collection_exists(COLLECTION):
        return [f"Qdrant collection '{COLLECTION}' missing"]
    rows, offset = [], None
    while True:
        points, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True)
        rows += [p.payload for p in points]
        if offset is None:
            break
    attractions, _, _ = _load_db()
    return _parity_errors(rows, "qdrant", {a.attraction_id for a in attractions})


CHECKS: Dict[str, Callable[[], List[str]]] = {
    "Referential integrity": check_referential_integrity,
    "Coordinate bounding box (Egypt)": check_coordinates,
    "Normalized hours & prices": check_hours_and_prices,
    "Chunk parity (processed_chunks.json)": check_chunks_json,
    "Chunk parity (Qdrant)": check_qdrant,
}


def main() -> int:
    failed = 0
    for name, fn in CHECKS.items():
        errors = fn()
        print(f"[{'PASS' if not errors else 'FAIL'}] {name}")
        for e in errors:
            print(f"    - {e}")
        failed += bool(errors)
    close_qdrant_client()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
