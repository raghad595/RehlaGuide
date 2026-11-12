"""Phase 2: two-tier RAG, grounded bilingual chat, cache invalidation, analytics.

Runs against temp copies of the DB and Qdrant (see conftest.py) with the deterministic
LLM fallback, so results are reproducible and nothing real is modified.
"""
import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services import llm_service, rag_service, vector_store
from app.services.query_parsing import detect_language, extract_filters
from app.services.rag_service import RetrievalResult, RetrievedChunk
from data.scripts.ingest_wikidata_template import RecordError, index_wikidata_records, validate_record

ARABIC = re.compile(r"[؀-ۿ]")
TIME_OR_MONEY = rag_service.OPERATIONAL_NUMBERS_RE

NILOMETER_EN = "Tell me about the Nilometer on Roda Island"
NILOMETER_AR = "حدثني عن مقياس النيل بالروضة"

WIKIDATA_SEED = [
    {
        "chunk_id": "QTEST1:en:0", "wikidata_qid": "QTEST1",
        "name_en": "Nilometer on Roda Island", "name_ar": "مقياس النيل بالروضة",
        "language": "en", "city": "Cairo", "category": "nilometer",
        "latitude": 30.0068, "longitude": 31.2250,
        "source_url": "https://www.wikidata.org/wiki/QTEST1",
        "text": ("The Nilometer on Roda Island is a structure built in 861 AD under the Abbasid caliph "
                 "al-Mutawakkil to measure the level of the Nile during the annual flood. Its stone shaft "
                 "and graduated central column were used to forecast harvests and set taxes."),
    },
    {
        "chunk_id": "QTEST1:ar:0", "wikidata_qid": "QTEST1",
        "name_en": "Nilometer on Roda Island", "name_ar": "مقياس النيل بالروضة",
        "language": "ar", "city": "Cairo", "category": "nilometer",
        "latitude": 30.0068, "longitude": 31.2250,
        "source_url": "https://www.wikidata.org/wiki/QTEST1",
        "text": ("مقياس النيل بالروضة منشأة بُنيت عام 861 م في عهد الخليفة العباسي المتوكل لقياس منسوب "
                 "مياه النيل أثناء الفيضان السنوي، وكان يُستخدم للتنبؤ بالمحاصيل وتحديد الضرائب."),
    },
    {   # A Wikidata record for a VERIFIED site: Tier 1 must still win for "Karnak".
        "chunk_id": "QTEST2:en:0", "wikidata_qid": "QTEST2",
        "name_en": "Karnak", "name_ar": "الكرنك",
        "language": "en", "city": "Luxor", "category": "temple complex",
        "source_url": "https://www.wikidata.org/wiki/QTEST2",
        "text": "Karnak is a vast ancient Egyptian temple complex near Luxor dedicated mainly to Amun.",
    },
]


def _rogue_point():
    """A Tier 2 point written WITHOUT the template (simulating a teammate mistake):
    it carries hours/price fields and an operational sentence inside its text."""
    from qdrant_client.models import PointStruct

    text = ("Monument: Bayt al-Suhaymi | City: Cairo | Category: historic house\n\n"
            "Bayt al-Suhaymi is an Ottoman-era merchant house in Islamic Cairo built in 1648, known for its "
            "mashrabiya windows and courtyard garden. It is open daily from 9:00 AM to 5:00 PM and tickets "
            "cost EGP 100.")
    vec = vector_store.embed_passages([text])[0].tolist()
    return PointStruct(id="00000000-0000-0000-0000-000000000123", vector=vec, payload={
        "chunk_id": "QTEST3:en:0", "wikidata_qid": "QTEST3", "name_en": "Bayt al-Suhaymi",
        "language": "en", "city": "Cairo", "category": "historic house",
        "source_url": "https://www.wikidata.org/wiki/QTEST3", "text": text,
        "opening_hours": "9:00 AM - 5:00 PM", "ticket_price": "EGP 100",
    })


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh_rag_cache():
    rag_service.invalidate_rag_cache()
    yield
    rag_service.invalidate_rag_cache()


def _drop_tier2():
    c = vector_store.get_qdrant_client()
    name = get_settings().unverified_collection
    if c.collection_exists(name):
        c.delete_collection(name)


@pytest.fixture
def tier2_empty():
    _drop_tier2()
    rag_service.invalidate_rag_cache()
    yield


@pytest.fixture
def tier2_seeded():
    _drop_tier2()
    index_wikidata_records(WIKIDATA_SEED)
    vector_store.get_qdrant_client().upsert(get_settings().unverified_collection, points=[_rogue_point()])
    rag_service.invalidate_rag_cache()
    yield
    _drop_tier2()


def ask(client, message, session_id="pytest", **extra):
    r = client.post("/api/v1/chat", json={"session_id": session_id, "message": message, **extra})
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- query parsing
@pytest.mark.parametrize("text, lang", [
    ("When does Karnak open?", "en"), ("متى يفتح معبد الكرنك؟", "ar"),
    ("Tell me about أبو الهول please", "en"), ("ما هو GEM؟", "ar"), ("", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_extract_filters_bilingual():
    cat = rag_service._catalog()
    en = extract_filters("museums in Cairo under 2 hours", cat.cities, cat.categories)
    ar = extract_filters("متاحف في القاهرة أقل من ساعتين", cat.cities, cat.categories)
    for f in (en, ar):
        assert f.city_label == "Cairo" and f.cities == ["Cairo"]
        assert "Museum" in f.categories and f.max_duration_minutes == 120
    giza = extract_filters("temples near Giza", cat.cities, cat.categories)
    assert set(giza.cities) == {"Giza", "Giza / Memphis"}


@pytest.mark.parametrize("text, ids", [
    ("Tell me about the Grand Egyptian Museum", ["GIZ003"]),
    ("Egyptian Museum ticket price", ["CAI001"]),
    ("Dahshur pyramids", ["GIZ005"]),
    ("كم سعر تذكرة الأهرامات للأجانب؟", ["GIZ001"]),
    ("هل أستطيع زيارة الكرنك وفيلة في يوم واحد؟", ["LXR001", "ASW001"]),
    ("Is Pompey's Pillar open?", ["ALX004"]),
    ("أقصر مدة زيارة", []),                      # "shortest", not Luxor
    (NILOMETER_EN, []),
])
def test_alias_matching(text, ids):
    assert rag_service._catalog().matcher.match(text) == ids


# --------------------------------------------------------------------------- Tier 1 (verified)
def test_verified_qa_english_hours_and_prices(client):
    r = ask(client, "What are the opening hours and ticket prices for the Egyptian Museum in Tahrir?")
    assert r["data_tier_used"] == "verified" and r["detected_language"] == "en"
    assert [a["attraction_id"] for a in r["referenced_attractions"]] == ["CAI001"]
    museum = r["referenced_attractions"][0]
    assert museum["ticket_prices"][0]["price_foreigners_egp"] == 550.0
    assert museum["opening_hours"][0]["open_time_24h"] == "09:00"
    msg = r["assistant_message"]
    assert "09:00 AM" in msg and "05:00 PM" in msg and "EGP 550" in msg
    assert "egymonuments.gov.eg" in msg and "2026-10-25" in msg  # source + last verified
    assert r["llm_provider"] == "deterministic" and r["supplementary_sources"] == []


def test_verified_qa_arabic(client):
    r = ask(client, "كم سعر تذكرة الأهرامات للأجانب؟")
    assert r["detected_language"] == "ar" and r["data_tier_used"] == "verified"
    assert [a["attraction_id"] for a in r["referenced_attractions"]] == ["GIZ001"]
    msg = r["assistant_message"]
    assert ARABIC.search(msg) and "700" in msg and "أهرامات الجيزة" in msg


def test_verified_semantic_without_name(client):
    r = ask(client, "Where can I see royal mummies?")
    assert r["data_tier_used"] == "verified"
    assert "CAI002" in [a["attraction_id"] for a in r["referenced_attractions"]]


def test_verified_discovery_with_filters(client):
    r = ask(client, "Which museums in Cairo are worth visiting?")
    assert r["data_tier_used"] == "verified"
    refs = r["referenced_attractions"]
    assert refs and all(a["city"] == "Cairo" and "Museum" in a["category"] for a in refs)


def test_preferred_language_overrides_detection(client):
    r = ask(client, "Karnak opening hours", preferred_language="ar")
    assert r["detected_language"] == "ar" and ARABIC.search(r["assistant_message"])


# --------------------------------------------------------------------------- tier "none"
@pytest.mark.parametrize("message, marker", [
    (NILOMETER_EN, "RehlaGuide knowledge base"),
    # Hours/ticket wording lifts an unrelated verified site (Kom Ombo/Philae) over the score
    # cut-off; the unknown proper noun "Roda" must still keep it out of the verified tier.
    (NILOMETER_EN + ". When is it open and how much is the ticket?", "RehlaGuide knowledge base"),
    ("How do I bake a chocolate cake?", "RehlaGuide knowledge base"),
    (NILOMETER_AR, "قاعدة معارف RehlaGuide"),
])
def test_unknown_attraction_without_tier2(client, tier2_empty, message, marker):
    r = ask(client, message)
    assert r["data_tier_used"] == "none"
    assert r["referenced_attractions"] == [] and r["supplementary_sources"] == []
    assert marker in r["assistant_message"]


# --------------------------------------------------------------------------- Tier 2 (unverified)
def test_unverified_fallback_english(client, tier2_seeded):
    r = ask(client, NILOMETER_EN)
    assert r["data_tier_used"] == "unverified_fallback"
    assert r["referenced_attractions"] == []
    assert r["supplementary_sources"][0]["wikidata_qid"] == "QTEST1"
    assert r["supplementary_sources"][0]["is_verified"] is False
    msg = r["assistant_message"]
    assert "Nilometer" in msg and "861" in msg                   # historical info
    assert "unverified" in msg and "Wikidata" in msg             # provenance stated
    assert "opening hours and ticket prices are not available" in msg
    assert not TIME_OR_MONEY.search(msg)


def test_unverified_fallback_arabic(client, tier2_seeded):
    r = ask(client, NILOMETER_AR)
    assert r["detected_language"] == "ar" and r["data_tier_used"] == "unverified_fallback"
    msg = r["assistant_message"]
    assert "مقياس النيل" in msg and "المتوكل" in msg
    assert "غير متوفرة" in msg and "Wikidata" in msg
    assert "المدينة:" not in msg and "| الفئة" not in msg   # embedding header not leaked
    assert not TIME_OR_MONEY.search(msg)


def test_unverified_operational_data_never_leaks(client, tier2_seeded):
    r = ask(client, "Tell me about Bayt al-Suhaymi, what are its opening hours and ticket price?")
    assert r["data_tier_used"] == "unverified_fallback"
    msg = r["assistant_message"]
    assert "Ottoman" in msg                                       # history kept
    assert "9:00" not in msg and "EGP 100" not in msg and "5:00 PM" not in msg
    assert not TIME_OR_MONEY.search(msg)
    result = rag_service.retrieve("Bayt al-Suhaymi", "en", session=None)  # session unused for Tier 2
    assert result.tier == "unverified_fallback"
    for chunk in result.unverified_chunks:
        assert set(chunk.payload) <= set(rag_service.UNVERIFIED_ALLOWED_FIELDS)
        assert "9:00" not in chunk.text and "EGP" not in chunk.text
    assert "opening_hours" not in result.context_blocks()


def test_verified_tier_wins_when_site_is_in_both(client, tier2_seeded):
    r = ask(client, "Karnak opening hours")
    assert r["data_tier_used"] == "verified"
    assert [a["attraction_id"] for a in r["referenced_attractions"]] == ["LXR001"]


def test_template_rejects_operational_fields():
    bad = {**WIKIDATA_SEED[0], "opening_hours": "9-5"}
    with pytest.raises(RecordError):
        validate_record(bad)
    with pytest.raises(RecordError):
        validate_record({**WIKIDATA_SEED[0], "text": ""})
    clean = validate_record(WIKIDATA_SEED[0])
    assert clean["is_verified"] is False and clean["trust_tier"] == "unverified_historical_only"


# --------------------------------------------------------------------------- LLM guardrails
def _result(tier, lang="en", chunks=()):
    return RetrievalResult(query="q", language=lang, tier=tier, match_method="none",
                           filters=extract_filters("q", [], []), unverified_chunks=list(chunks))


def test_guardrails():
    unverified = _result("unverified_fallback", chunks=[RetrievedChunk(0.9, {"name_en": "X", "text": "Old."})])
    assert llm_service.apply_guardrails("It opens at 9:00 AM and costs EGP 50.", unverified) is None
    assert llm_service.apply_guardrails("Tickets are $10.", unverified) is None
    patched = llm_service.apply_guardrails("Built in 861 AD.", unverified)
    assert patched.startswith("Built in 861 AD.") and "not available" in patched
    assert llm_service.apply_guardrails("English only", _result("verified", lang="ar")) is None
    assert llm_service.apply_guardrails("Opens 09:00.", _result("verified")) == "Opens 09:00."


def test_llm_provider_used_and_failure_falls_back(client, monkeypatch):
    monkeypatch.setattr(llm_service, "resolve_provider", lambda: "groq")
    monkeypatch.setattr(llm_service, "_call_provider", lambda p, s, u: "Karnak opens at 06:00 (source: test).")
    r = ask(client, "Karnak opening hours")
    assert r["llm_provider"] == "groq" and r["assistant_message"].startswith("Karnak opens at 06:00")

    def boom(*_):
        raise RuntimeError("network down")

    rag_service.invalidate_rag_cache()
    monkeypatch.setattr(llm_service, "_call_provider", boom)
    r = ask(client, "Karnak opening hours")
    assert r["llm_provider"] == "deterministic" and "Karnak Temple" in r["assistant_message"]


def test_prompt_contains_tagged_blocks(client, tier2_seeded):
    from app.db.database import engine
    from sqlmodel import Session

    with Session(engine) as s:
        verified = rag_service.retrieve("Karnak ticket price", "en", s)
        unverified = rag_service.retrieve(NILOMETER_EN, "en", s)
    vp = llm_service.build_user_prompt(verified)
    assert "<verified_operational_data>" in vp and "<verified_narrative_chunks>" in vp and "LXR001" in vp
    up = llm_service.build_user_prompt(unverified)
    assert '<unverified_supplementary_chunks source="Wikidata">' in up
    assert "<verified_operational_data>" not in up and "<data_tier_used>unverified_fallback" in up


# --------------------------------------------------------------------------- cache
def test_cache_hit_and_status_invalidation(client):
    q = "Is the Egyptian Museum in Tahrir open today?"
    first = ask(client, q, session_id="pytest-a")
    second = ask(client, q, session_id="pytest-b")
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert second["session_id"] == "pytest-b"
    assert second["assistant_message"] == first["assistant_message"]

    other = "Karnak ticket price"
    ask(client, other)  # cache an answer about a DIFFERENT attraction
    try:
        r = client.patch("/api/v1/attractions/CAI001/status",
                         json={"is_open": False, "closure_reason": "Gallery renovation"})
        assert r.status_code == 200
        closed = ask(client, q)
        assert closed["cache_hit"] is False                         # invalidated
        assert closed["referenced_attractions"][0]["is_open"] is False
        assert "CURRENTLY CLOSED" in closed["assistant_message"]
        assert "Gallery renovation" in closed["assistant_message"]
        assert ask(client, other)["cache_hit"] is True               # unrelated entry kept
    finally:
        client.patch("/api/v1/attractions/CAI001/status", json={"is_open": True})
    reopened = ask(client, q)
    assert reopened["cache_hit"] is False and "Status: Open" in reopened["assistant_message"]


def test_arabic_and_english_cached_separately(client):
    ask(client, "Karnak opening hours")
    r = ask(client, "Karnak opening hours", preferred_language="ar")
    assert r["cache_hit"] is False and r["detected_language"] == "ar"


# --------------------------------------------------------------------------- analytics
def test_analytics_summary(client, tier2_empty):
    before = client.get("/api/v1/analytics/summary").json()

    ask(client, "Karnak opening hours", session_id="pytest-analytics")             # verified, miss
    ask(client, "Karnak opening hours", session_id="pytest-analytics")             # verified, hit
    ask(client, "متى يفتح معبد الكرنك؟", session_id="pytest-analytics")             # verified (ar)
    ask(client, "How do I bake a chocolate cake?", session_id="pytest-analytics")  # none
    assert client.get("/api/v1/attractions/LXR001",
                      headers={"X-Session-Id": "pytest-analytics"}).status_code == 200

    after = client.get("/api/v1/analytics/summary").json()
    delta = lambda key, sub: after[key].get(sub, 0) - before[key].get(sub, 0)  # noqa: E731
    assert after["total_queries"] - before["total_queries"] == 5
    assert delta("tier_distribution", "verified") == 4        # 3 chats + 1 lookup
    assert delta("tier_distribution", "none") == 1
    assert delta("language_distribution", "ar") == 1
    assert delta("event_type_distribution", "chat_qa") == 4
    assert delta("event_type_distribution", "attraction_lookup") == 1
    assert 0 < after["cache_hit_rate_pct"] <= 100 and after["average_latency_ms"] >= 0
    assert any(c["city"] == "Luxor" for c in after["top_cities"])
    top = {a["attraction_id"]: a for a in after["top_attractions"]}
    assert "LXR001" in top and top["LXR001"]["name_en"] == "Karnak Temple"

    chat_only = client.get("/api/v1/analytics/summary", params={"event_type": "chat_qa"}).json()
    assert set(chat_only["event_type_distribution"]) == {"chat_qa"}


def test_chat_validation(client):
    assert client.post("/api/v1/chat", json={"session_id": "x", "message": ""}).status_code == 422
    assert client.post("/api/v1/chat", json={"session_id": "x", "message": "hi",
                                             "preferred_language": "fr"}).status_code == 422
