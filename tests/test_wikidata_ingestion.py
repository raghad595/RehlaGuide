"""Tier 2 (Wikidata) ingestion + live two-tier routing on the REAL ingested dataset.

Runs on temp copies of data/rehlaguide.db and data/qdrant_db (conftest.py), with the
deterministic LLM fallback (no API quota). Skips with instructions if the Wikidata
pipeline hasn't been run on this machine yet.
"""
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.database import engine
from app.main import app
from app.models import UnverifiedAttraction
from app.services import rag_service, vector_store
from data.scripts.ingest_wikidata import build_chunks, clean_records, parse_wkt

TIME_OR_MONEY = rag_service.OPERATIONAL_NUMBERS_RE
ARABIC = re.compile(r"[؀-ۿ]")
FORBIDDEN_COLUMN_PARTS = ("hour", "opening", "closing", "price", "ticket", "fee", "admission")

# Preferred well-known sites for the live routing checks (first one present in the data wins).
PREFERRED_EN = ["Nilometer", "Ibn Tulun", "Al-Azhar", "Sultan Hassan", "Hanging Church", "Baron Empain",
                "Temple of Edfu", "Dendera", "Abydos", "Bab Zuweila"]
PREFERRED_AR = ["مقياس النيل", "ابن طولون", "الجامع الأزهر", "السلطان حسن", "الكنيسة المعلقة", "قصر البارون",
                "معبد إدفو", "دندرة", "باب زويلة"]


def _tier2_ready() -> bool:
    with Session(engine) as s:
        if not inspect(engine).has_table("unverified_attractions"):
            return False
        if not s.exec(select(UnverifiedAttraction).limit(1)).first():
            return False
    return vector_store.collection_point_count(get_settings().unverified_collection) > 0


pytestmark = pytest.mark.skipif(
    not _tier2_ready(),
    reason="Tier 2 not ingested: run data/scripts/fetch_wikidata.py then data/scripts/ingest_wikidata.py",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh_cache():
    rag_service.invalidate_rag_cache()
    yield


def _all_payloads():
    c = vector_store.get_qdrant_client()
    rows, offset = [], None
    while True:
        pts, offset = c.scroll(get_settings().unverified_collection, limit=512, offset=offset, with_payload=True)
        rows += [p.payload for p in pts]
        if offset is None:
            return rows


def ask(client, message):
    r = client.post("/api/v1/chat", json={"session_id": "pytest-wikidata", "message": message})
    assert r.status_code == 200, r.text
    return r.json()


def _pick(lang: str):
    """A real, indexed (non-duplicate) Wikidata site with a specific name in `lang`."""
    key = "name_en" if lang == "en" else "name_ar"
    indexed = {p["wikidata_qid"]: p for p in _all_payloads() if p["language"] == lang}
    for needle in (PREFERRED_EN if lang == "en" else PREFERRED_AR):
        for p in indexed.values():
            if needle in (p.get(key) or "") and rag_service._is_specific_name(p.get(key)) \
                    and (p.get("description") or p.get("summary")):
                return p
    pytest.fail(f"none of the preferred {lang} sites are indexed; update PREFERRED_{lang.upper()}")


# --------------------------------------------------------------------------- SQLite table
def test_table_has_no_operational_columns():
    cols = [c["name"] for c in inspect(engine).get_columns("unverified_attractions")]
    assert "qid" in cols and "matches_verified_id" in cols
    assert not [c for c in cols if any(p in c.lower() for p in FORBIDDEN_COLUMN_PARTS)], cols


def test_table_populated_and_clean():
    with Session(engine) as s:
        rows = s.exec(select(UnverifiedAttraction)).all()
    assert len(rows) >= 100
    for r in rows:
        assert r.name_en or r.name_ar
        for name in (r.name_en, r.name_ar):
            assert not (name and re.fullmatch(r"Q\d+", name)), r.qid
        assert r.is_verified is False and r.trust_tier == "unverified_historical_only"
        assert r.source_url == f"https://www.wikidata.org/wiki/{r.qid}"


def test_coordinates_inside_egypt():
    with Session(engine) as s:
        rows = s.exec(select(UnverifiedAttraction).where(UnverifiedAttraction.latitude.is_not(None))).all()
    assert rows
    bad = [(r.qid, r.latitude, r.longitude) for r in rows
           if not (22.0 <= r.latitude <= 31.8 and 24.5 <= r.longitude <= 37.0)]
    assert bad == []


def test_tier1_duplicates_detected():
    with Session(engine) as s:
        dupes = s.exec(select(UnverifiedAttraction).where(UnverifiedAttraction.matches_verified_id.is_not(None))).all()
    assert dupes, "expected Wikidata items for verified sites (e.g. Karnak, Abu Simbel) to be linked"
    indexed_qids = {p["wikidata_qid"] for p in _all_payloads()}
    assert not ({d.qid for d in dupes} & indexed_qids), "Tier 1 duplicates must not be indexed in Tier 2"


# --------------------------------------------------------------------------- Qdrant collection
def test_collection_bilingual_and_untrusted():
    payloads = _all_payloads()
    langs = {p["language"] for p in payloads}
    assert langs == {"en", "ar"}
    for p in payloads:
        assert p["is_verified"] is False and p["trust_tier"] == "unverified_historical_only"
        assert p["attraction_id"] == f"wikidata_{p['wikidata_qid']}"
        assert not [k for k in p if any(part in k.lower() for part in FORBIDDEN_COLUMN_PARTS)]
    ar = [p for p in payloads if p["language"] == "ar"]
    assert all(ARABIC.search(p["text"]) for p in ar)


# --------------------------------------------------------------------------- pure helpers
def test_wkt_order_is_lon_then_lat():
    assert parse_wkt("Point(31.2357 30.0444)") == (30.0444, 31.2357)
    assert parse_wkt("garbage") == (None, None)


def test_cleaning_rules():
    raw = [
        {"qid": "Q1", "name_en": "Q1", "name_ar": None, "description_en": "x"},             # QID label -> dropped
        {"qid": "Q2", "name_en": "Lonely Name"},                                              # no substance -> dropped
        {"qid": "Q3", "name_en": "Far Away Site", "instance_of_en": "temple",
         "coordinates": "Point(2.35 48.85)"},                                                 # Paris -> coords None
        {"qid": "Q4", "name_en": "Old Gate", "instance_of_en": "city gate",
         "description_en": "Medieval gate. It is open daily from 9:00 AM and tickets cost EGP 50.",
         "coordinates": "Point(31.26 30.04)", "governorate_en": "Cairo Governorate"},
    ]
    kept, stats = clean_records(raw)
    assert [r["qid"] for r in kept] == ["Q3", "Q4"]
    assert stats["dropped_no_name"] == 1 and stats["dropped_no_substance"] == 1
    assert kept[0]["latitude"] is None and stats["coords_out_of_bounds"] == 1
    gate = kept[1]
    assert gate["city_en"] == "Cairo" and gate["category"] == "fort" and gate["latitude"] == 30.04
    assert "9:00" not in gate["description_en"] and "EGP" not in gate["description_en"]

    gate["matches_verified_id"] = None
    chunks = build_chunks(gate)
    assert [c["language"] for c in chunks] == ["en"]
    text = chunks[0]["text"]
    assert text.startswith("Attraction (Unverified - Wikidata): Old Gate | Type: city gate | Location: Cairo")
    assert "None" not in text and "| |" not in text and "Heritage" not in text  # empty fields omitted


# --------------------------------------------------------------------------- live routing
def test_verified_sites_still_use_tier1(client):
    en = ask(client, "What are the opening hours and ticket prices for Karnak Temple?")
    assert en["data_tier_used"] == "verified"
    assert [a["attraction_id"] for a in en["referenced_attractions"]] == ["LXR001"]
    assert "EGP 600" in en["assistant_message"] and "06:00 AM" in en["assistant_message"]

    ar = ask(client, "كم سعر تذكرة أهرامات الجيزة للأجانب؟")
    assert ar["data_tier_used"] == "verified"
    assert [a["attraction_id"] for a in ar["referenced_attractions"]] == ["GIZ001"]
    assert "700" in ar["assistant_message"]


def _assert_unverified_answer(r, site, notice):
    assert r["data_tier_used"] == "unverified_fallback", r["assistant_message"][:300]
    assert r["referenced_attractions"] == []
    qids = [s["wikidata_qid"] for s in r["supplementary_sources"]]
    assert site["wikidata_qid"] in qids
    msg = r["assistant_message"]
    assert f"https://www.wikidata.org/wiki/{site['wikidata_qid']}" in msg
    assert notice in msg
    assert not TIME_OR_MONEY.search(msg), msg


def test_real_unverified_site_english(client):
    site = _pick("en")
    r = ask(client, f"Tell me about {site['name_en']}. When is it open and how much is the ticket?")
    _assert_unverified_answer(r, site, "opening hours and ticket prices are not available")
    assert "Wikidata" in r["assistant_message"]


def test_real_unverified_site_arabic(client):
    site = _pick("ar")
    r = ask(client, f"حدثني عن {site['name_ar']}، ما هي مواعيد الزيارة وسعر التذكرة؟")
    assert r["detected_language"] == "ar"
    _assert_unverified_answer(r, site, "غير متوفرة")


def test_off_topic_still_none(client):
    for q in ("How do I bake a chocolate cake?", "What is the capital of France?"):
        assert ask(client, q)["data_tier_used"] == "none", q
