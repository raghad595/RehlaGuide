"""Clean the Wikidata snapshot, load it into SQLite (`unverified_attractions`) and index
bilingual profile chunks into Qdrant (`rehlaguide_wikidata_unverified`). Idempotent.

Run (stop the API server first - local Qdrant allows one process per folder):
    $env:PYTHONPATH="."
    .venv\\Scripts\\python.exe data/scripts/fetch_wikidata.py      # once, or to refresh the snapshot
    .venv\\Scripts\\python.exe data/scripts/ingest_wikidata.py     # offline from the snapshot
Options:
    --skip-index     SQLite + chunk JSON only (no embeddings / Qdrant)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlmodel import Session, delete, func, select

from app.db.database import engine, init_db
from app.models import Attraction, UnverifiedAttraction
from app.services import rag_service
from app.services.query_parsing import AliasMatcher
from app.services.rag_service import sanitize_unverified_text
from app.services.vector_store import close_qdrant_client
from data.scripts.ingest_wikidata_template import (
    collection_name, ensure_collection, upsert_payloads, validate_record, _check_unique,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "raw_wikidata"
SNAPSHOT = RAW_DIR / "egypt_wikidata_attractions.json"
CHUNKS_OUT = RAW_DIR / "processed_wikidata_chunks.json"

LAT_RANGE, LON_RANGE = (22.0, 31.8), (24.5, 37.0)
WKT_POINT_RE = re.compile(r"Point\(\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)")
QID_RE = re.compile(r"^Q\d+$")
MODERN_CITY_TYPES = {"big city", "megacity", "metropolis", "national capital", "capital city",
                     "million city", "city with millions of inhabitants", "municipality", "port city"}
DUPLICATE_MAX_KM = 10.0      # name match must also be this close (Tier 1 coords are rounded to ~1 km)
NEAR_VERIFIED_KM = 0.25      # reported only; never used to drop a record on its own

# Wikidata governorate label -> RehlaGuide city label (same vocabulary as the verified dataset).
CITY_EN = {"Faiyum": "Fayoum", "Matruh": "Matrouh", "6th of October": "Giza", "Sinai": "South Sinai"}
CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("museum", ("museum",)),
    ("mosque", ("mosque", "masjid")),
    ("church", ("church", "cathedral", "chapel", "basilica")),
    ("monastery", ("monastery", "convent", "hermitage")),
    ("synagogue", ("synagogue",)),
    ("temple", ("temple",)),
    ("pyramid", ("pyramid",)),
    ("tomb", ("tomb", "mastaba", "mausoleum", "necropolis", "cemetery", "catacomb", "burial")),
    ("fort", ("fort", "citadel", "castle", "fortress", "fortification", "city wall", "city gate")),
    ("palace", ("palace",)),
    ("islamic_monument", ("madrasa", "sabil", "khanqah", "wikala", "caravanserai", "hammam", "minaret")),
    ("archaeological_site", ("archaeological site", "ancient city", "ruins", "tell", "archaeological")),
    ("natural_site", ("protected area", "national park", "nature reserve", "oasis", "mountain", "cave", "wadi")),
]


# --------------------------------------------------------------------------- cleaning helpers
def _s(v) -> Optional[str]:
    if v is None:
        return None
    t = re.sub(r"\s+", " ", str(v)).strip()
    return t or None


def clean_name(v) -> Optional[str]:
    t = _s(v)
    return None if (not t or QID_RE.match(t)) else t


def parse_wkt(wkt: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """WKT is 'Point(<lon> <lat>)' -> returns (lat, lon)."""
    m = WKT_POINT_RE.search(wkt or "")
    if not m:
        return None, None
    lon, lat = float(m.group(1)), float(m.group(2))
    return lat, lon


def in_egypt(lat: Optional[float], lon: Optional[float]) -> bool:
    return lat is not None and lon is not None and LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LON_RANGE[0] <= lon <= LON_RANGE[1]


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def first(v: Optional[str]) -> Optional[str]:
    return v.split(";")[0].strip() if v else None


def normalize_city(gov_en: Optional[str], gov_ar: Optional[str], loc_en: Optional[str], loc_ar: Optional[str]):
    en = first(gov_en)
    if en:
        en = re.sub(r"\s*Governorate$", "", en).strip()
        en = CITY_EN.get(en, en)
    ar = first(gov_ar)
    if ar:
        ar = re.sub(r"^محافظة\s*", "", ar).strip()
    return en or first(loc_en), ar or first(loc_ar)


def categorize(instance_of_en: Optional[str], source_classes: Optional[str]) -> str:
    text = f"{instance_of_en or ''}; {source_classes or ''}".lower()
    for category, words in CATEGORY_RULES:
        if any(w in text for w in words):
            return category
    return "historical_monument"


def strip_operational(text: Optional[str]) -> Optional[str]:
    """Remove any sentence with opening hours / prices (Wikipedia extracts occasionally have them)."""
    return _s(sanitize_unverified_text(text)) if text else None


# --------------------------------------------------------------------------- cleaning
def clean_records(raw: List[dict]) -> Tuple[List[dict], Counter]:
    stats: Counter = Counter(raw=len(raw))
    out = []
    for r in raw:
        rec = {k: _s(v) for k, v in r.items()}
        rec["name_en"], rec["name_ar"] = clean_name(r.get("name_en")), clean_name(r.get("name_ar"))
        if not (rec["name_en"] or rec["name_ar"]):
            stats["dropped_no_name"] += 1
            continue
        for k in ("description_en", "description_ar", "summary_en", "summary_ar"):
            rec[k] = strip_operational(rec.get(k))
        types = {t.strip().lower() for t in (rec.get("instance_of_en") or "").split(";")}
        if types & MODERN_CITY_TYPES:
            # Living cities (Cairo, Alexandria, Asyut...) sometimes also carry "archaeological site";
            # as Tier 2 "monuments" they would hijack city-level questions.
            stats["dropped_modern_city"] += 1
            continue
        substance = ("description_en", "description_ar", "instance_of_en", "instance_of_ar",
                     "architectural_styles_en", "architectural_styles_ar", "located_in_en", "located_in_ar",
                     "summary_en", "summary_ar")
        if not any(rec.get(k) for k in substance):
            stats["dropped_no_substance"] += 1
            continue
        lat, lon = parse_wkt(rec.get("coordinates"))
        if lat is not None and not in_egypt(lat, lon):
            stats["coords_out_of_bounds"] += 1
            lat = lon = None
        if lat is None:
            stats["no_valid_coords"] += 1
        rec["latitude"], rec["longitude"] = lat, lon
        rec["city_en"], rec["city_ar"] = normalize_city(rec.get("governorate_en"), rec.get("governorate_ar"),
                                                        rec.get("located_in_en"), rec.get("located_in_ar"))
        rec["category"] = categorize(rec.get("instance_of_en"), rec.get("source_classes"))
        out.append(rec)
    stats["kept"] = len(out)
    return out, stats


# Words that don't distinguish one site from another ("Karnak Temple Complex" == "Karnak").
_NAME_FILLER = {
    "the", "of", "and", "in", "at", "el", "al", "a", "an", "de", "la", "to", "from", "on",
    "temple", "temples", "complex", "site", "area", "necropolis", "museum", "museums", "monuments",
    "great", "new", "old", "ancient", "international", "national", "archaeological", "pyramid", "pyramids",
    "giza", "cairo", "alexandria", "luxor", "aswan", "egypt", "sinai", "siwa", "oasis", "upper",
    "معبد", "معابد", "مجمع", "منطقه", "موقع", "متحف", "متاحف", "اثار", "الدولي", "القومي", "مقابر",
    "هرم", "اهرامات", "الكبير", "من", "في", "حتي", "الي",
}
_AR_PREFIX_RE = re.compile(r"^(?:و|ب|ف|ل|لل)?(?:ال)?")


_MUSEUM_WORDS = {"museum", "museums", "متحف", "متاحف"}


def _is_filler(token: str, allow_museum: bool) -> bool:
    stripped = _AR_PREFIX_RE.sub("", token)
    if not allow_museum and (token in _MUSEUM_WORDS or stripped in _MUSEUM_WORDS):
        return False
    if token in _NAME_FILLER or len(token) <= 1:
        return True
    return stripped in _NAME_FILLER or ("ال" + stripped) in _NAME_FILLER


def same_site_as_alias(matcher: AliasMatcher, name: Optional[str], categories: Dict[str, str]) -> List[str]:
    """Verified ids whose alias covers the WHOLE name except filler words.
    'Karnak Temple Complex' -> [LXR001]; 'Tomb of Horemheb (Saqqara)' -> [] (a distinct monument);
    'Museum of Siwan Traditions' -> [] ("museum" only counts as filler for verified museums)."""
    if not name:
        return []
    norm, spans = matcher.match_spans(name)
    if not spans:
        return []
    residual = norm
    for start, end, _ in sorted(spans, reverse=True):
        residual = residual[:start] + " " + residual[end:]
    ids = list(dict.fromkeys(aid for _, _, aid in spans))
    return [aid for aid in ids
            if all(_is_filler(t, "museum" in categories[aid].lower()) for t in residual.split())]


def mark_tier1_duplicates(records: List[dict], verified: List[Attraction]) -> Counter:
    """matches_verified_id = verified attraction that this Wikidata item IS (its alias covers the
    whole EN or AR name, using the same matcher the chat uses), confirmed by distance when both
    have coordinates. Items that merely *mention* a verified site (sub-sites, tombs at Saqqara,
    Shali in Siwa) stay in Tier 2: questions naming only them can't be answered by Tier 1."""
    stats: Counter = Counter()
    matcher = AliasMatcher([(a.attraction_id, a.name_en, a.name_ar) for a in verified])
    coords = {a.attraction_id: (a.latitude, a.longitude) for a in verified}
    categories = {a.attraction_id: a.category for a in verified}
    for rec in records:
        rec["matches_verified_id"] = None
        here = (rec["latitude"], rec["longitude"]) if rec["latitude"] is not None else None
        candidates = list(dict.fromkeys(same_site_as_alias(matcher, rec.get("name_en"), categories)
                                        + same_site_as_alias(matcher, rec.get("name_ar"), categories)))
        if not candidates and (matcher.match(rec.get("name_en") or "") or matcher.match(rec.get("name_ar") or "")):
            stats["mentions_verified_site_kept"] += 1
        for aid in candidates:
            if here is None or haversine_km(here, coords[aid]) <= DUPLICATE_MAX_KM:
                rec["matches_verified_id"] = aid
                stats["duplicates_of_tier1"] += 1
                break
        else:
            if candidates:
                stats["name_collision_far_away"] += 1  # same alias, different place -> keep in Tier 2
        if here and not rec["matches_verified_id"] and any(
                haversine_km(here, c) <= NEAR_VERIFIED_KM for c in coords.values()):
            stats["near_verified_but_different_name"] += 1
    return stats


# --------------------------------------------------------------------------- chunks
NOTE_EN = "(Note: Unverified supplementary historical record; verified opening hours and ticket prices are not available.)"
NOTE_AR = "(ملاحظة: سجل تاريخي تكميلي غير موثق؛ مواعيد العمل وأسعار التذاكر الموثقة غير متوفرة.)"


def _join(parts: List[Tuple[str, Optional[str]]], sep: str = " | ") -> str:
    return sep.join(f"{label}{value}" for label, value in parts if value)


def _location(located_in: Optional[str], city: Optional[str]) -> Optional[str]:
    loc = first(located_in)
    if loc and city and city.lower() not in loc.lower():
        return f"{loc}, {city}"
    return loc or city


def build_chunks(rec: dict) -> List[dict]:
    qid = rec["qid"]
    src = f"https://www.wikidata.org/wiki/{qid}"
    base = {
        "attraction_id": f"wikidata_{qid}", "wikidata_qid": qid,
        "name_en": rec.get("name_en"), "name_ar": rec.get("name_ar"),
        "city": rec.get("city_en") or "Egypt", "category": rec["category"],
        "latitude": rec.get("latitude"), "longitude": rec.get("longitude"),
        "image_url": rec.get("image_url"), "source_url": src, "inception": rec.get("inception"),
    }
    chunks = []
    if rec.get("name_en") and any(rec.get(k) for k in ("description_en", "summary_en", "instance_of_en")):
        loc = _location(rec.get("located_in_en"), rec.get("city_en"))
        name = rec["name_en"] + (f" ({rec['name_ar']})" if rec.get("name_ar") else "")
        lines = [
            _join([("Attraction (Unverified - Wikidata): ", name), ("Type: ", rec.get("instance_of_en")),
                   ("Location: ", loc), ("Category: ", rec["category"])]),
            _join([("Description: ", rec.get("description_en"))]),
            _join([("Built/Inception: ", rec.get("inception")),
                   ("Architectural Style: ", rec.get("architectural_styles_en")),
                   ("Heritage: ", rec.get("heritage_en"))]),
            _join([("Details: ", rec.get("summary_en"))]),
            f"Source: {src} {NOTE_EN}",
        ]
        chunks.append({**base, "chunk_id": f"{qid}:en", "language": "en", "text": "\n".join(l for l in lines if l),
                       "description": rec.get("description_en"), "summary": rec.get("summary_en"),
                       "instance_of": rec.get("instance_of_en"), "located_in": loc,
                       "architectural_styles": rec.get("architectural_styles_en"), "heritage": rec.get("heritage_en")})
    if rec.get("name_ar") and any(rec.get(k) for k in ("description_ar", "summary_ar", "instance_of_ar")):
        loc = _location(rec.get("located_in_ar"), rec.get("city_ar"))
        name = rec["name_ar"] + (f" ({rec['name_en']})" if rec.get("name_en") else "")
        lines = [
            _join([("المعلم (غير موثق - ويكي بيانات): ", name), ("النوع: ", rec.get("instance_of_ar")),
                   ("الموقع: ", loc)]),
            _join([("الوصف: ", rec.get("description_ar"))]),
            _join([("تاريخ الإنشاء: ", rec.get("inception")),
                   ("الطراز المعماري: ", rec.get("architectural_styles_ar")),
                   ("التصنيف التراثي: ", rec.get("heritage_ar"))]),
            _join([("تفاصيل: ", rec.get("summary_ar"))]),
            f"المصدر: {src} {NOTE_AR}",
        ]
        chunks.append({**base, "chunk_id": f"{qid}:ar", "language": "ar", "text": "\n".join(l for l in lines if l),
                       "description": rec.get("description_ar"), "summary": rec.get("summary_ar"),
                       "instance_of": rec.get("instance_of_ar"), "located_in": loc,
                       "architectural_styles": rec.get("architectural_styles_ar"), "heritage": rec.get("heritage_ar")})
    # Contract check (forbidden operational keys, required fields, fixed trust flags).
    return [{**validate_record(c), "text": c["text"]} for c in chunks]


# --------------------------------------------------------------------------- SQLite
def to_row(rec: dict) -> UnverifiedAttraction:
    return UnverifiedAttraction(
        qid=rec["qid"], name_en=rec.get("name_en"), name_ar=rec.get("name_ar"),
        description_en=rec.get("description_en"), description_ar=rec.get("description_ar"),
        summary_en=rec.get("summary_en"), summary_ar=rec.get("summary_ar"),
        city_en=rec.get("city_en"), city_ar=rec.get("city_ar"), category=rec["category"],
        architectural_styles_en=rec.get("architectural_styles_en"),
        architectural_styles_ar=rec.get("architectural_styles_ar"),
        inception=rec.get("inception"), heritage_status=rec.get("heritage_en"),
        latitude=rec.get("latitude"), longitude=rec.get("longitude"), image_url=rec.get("image_url"),
        source_url=f"https://www.wikidata.org/wiki/{rec['qid']}",
        wikipedia_en_url=rec.get("wikipedia_en_url"), wikipedia_ar_url=rec.get("wikipedia_ar_url"),
        matches_verified_id=rec.get("matches_verified_id"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the Wikidata snapshot (Tier 2)")
    ap.add_argument("--skip-index", action="store_true", help="SQLite + chunk JSON only")
    args = ap.parse_args()
    if not SNAPSHOT.exists():
        print(f"ERROR: {SNAPSHOT} not found - run data/scripts/fetch_wikidata.py first.")
        return 1
    started = time.monotonic()
    init_db()  # creates unverified_attractions if missing; never touches other tables

    raw = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    records, stats = clean_records(raw)
    with Session(engine) as s:
        verified = s.exec(select(Attraction)).all()
    if not verified:
        print("ERROR: no verified attractions in SQLite - run data/scripts/ingest_sql.py first.")
        return 1
    stats.update(mark_tier1_duplicates(records, verified))

    # SQLite: full refresh of this table only (idempotent).
    with Session(engine) as s:
        s.exec(delete(UnverifiedAttraction))
        s.add_all([to_row(r) for r in records])
        s.commit()
        stored = s.exec(select(func.count()).select_from(UnverifiedAttraction)).one()

    chunks = [c for r in records if not r["matches_verified_id"] for c in build_chunks(r)]
    _check_unique(chunks)
    CHUNKS_OUT.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    by_lang = Counter(c["language"] for c in chunks)

    indexed = None
    if not args.skip_index:
        print(f"Embedding {len(chunks)} chunks into '{collection_name()}' ...")
        try:
            ensure_collection(recreate=True)  # drop stale points from earlier snapshots
            indexed = upsert_payloads(chunks, batch_size=64, progress=True)
        finally:
            close_qdrant_client()
        rag_service.reset_catalog()

    print("\n=== Wikidata Tier 2 ingestion summary ===")
    print(f"  raw records in snapshot        : {stats['raw']}")
    print(f"  dropped (no human-readable name): {stats['dropped_no_name']}")
    print(f"  dropped (no descriptive content): {stats['dropped_no_substance']}")
    print(f"  dropped (modern city, not a site): {stats['dropped_modern_city']}")
    print(f"  cleaned & kept                  : {stats['kept']}")
    print(f"  coordinates outside Egypt -> None: {stats['coords_out_of_bounds']}  "
          f"(records without valid coords: {stats['no_valid_coords']})")
    print(f"  duplicates of Tier 1 (not indexed): {stats['duplicates_of_tier1']}")
    print(f"  mention a verified site but are distinct (kept): {stats['mentions_verified_site_kept']}")
    print(f"  same alias but >{DUPLICATE_MAX_KM:.0f} km away (kept)  : {stats['name_collision_far_away']}")
    print(f"  within {NEAR_VERIFIED_KM * 1000:.0f} m of a verified site, other name (kept): "
          f"{stats['near_verified_but_different_name']}")
    print(f"  SQLite unverified_attractions rows: {stored}")
    print(f"  chunks: {len(chunks)}  (en={by_lang['en']}, ar={by_lang['ar']})  -> {CHUNKS_OUT.name}")
    if indexed is not None:
        print(f"  Qdrant '{collection_name()}' points: {indexed}")
    print(f"  took {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
