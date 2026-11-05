"""Phase 1 verification suite. Requires ingest_sql.py and ingest_vectors.py to have run.

Run:  PYTHONPATH=. pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from app.api.v1.attractions import clear_cache
from app.main import app
from data.scripts import verify_phase1_data as v
from data.scripts.ingest_sql import parse_days, parse_price, parse_time


# --------------------------------------------------------------------------- parser units
@pytest.mark.parametrize("raw, expected", [
    ("Every day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    ("Daily", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    ("Daily except Friday and Sunday", ["Mon", "Tue", "Wed", "Thu", "Sat"]),
    ("Sunday, Monday, Tuesday, Thursday, Friday", ["Mon", "Tue", "Thu", "Fri", "Sun"]),
    ("Saturday, Wednesday", ["Wed", "Sat"]),
    ("Sunday-Thursday", ["Mon", "Tue", "Wed", "Thu", "Sun"]),
    ("Monday-Thursday, Saturday", ["Mon", "Tue", "Wed", "Thu", "Sat"]),
    ("Thursday-Friday", ["Thu", "Fri"]),
    ("Friday", ["Fri"]),
    ("Not specified", []),
])
def test_parse_days(raw, expected):
    assert parse_days(raw) == expected


@pytest.mark.parametrize("raw, role, expected", [
    ("9:00 AM", "open", "09:00"),
    ("11:30 AM", "open", "11:30"),
    ("5:00 PM", "close", "17:00"),
    ("12:45 PM", "close", "12:45"),
    ("12:00 AM", "open", "00:00"),
    ("Sunrise (approximately 08:00 AM)", "open", "08:00"),
    ("Sunset (approximately 05:00 PM)", "close", "17:00"),
    ("Open 24 hours", "open", "00:00"),
    ("24 Hours", "close", "23:59"),
    ("Sunrise", "open", None),
    ("Closed", "open", None),
    ("Not specified", "close", None),
    ("1 hour before Galleries close", "close", None),
    ("", "open", None),
])
def test_parse_time(raw, role, expected):
    assert parse_time(raw, role=role) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Adult: EGP 60 / Student: EGP 30", 60.0),
    ("Adult: EGP 1,570 / Student: EGP 790", 1570.0),
    ("Adult: EGP 822 / Student: EGP 445.50", 822.0),
    ("EGP 40", 40.0),
    ("EGP 30 for adult student and senior visitors", 30.0),
    ("EGP 790 (Ages 6-12) / Free (Under 6)", 790.0),
    ("Free", 0.0),
    ("Free (Under 6)", 0.0),
    ("Free (Museum area ~EGP 200 / $4 USD)", 0.0),
    ("0", 0.0),
    ("Included", 0.0),
    ("$15 USD", None),
    ("Check official site", None),
    ("Not specified", None),
    ("50% off adult price", None),
    (None, None),
])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


# --------------------------------------------------------------------------- data checks
def test_attraction_count_matches_csv():
    attractions, _, _ = v._load_db()
    assert len(attractions) == v.csv_attraction_count() == 29


def test_referential_integrity():
    assert v.check_referential_integrity() == []


def test_coordinates_inside_egypt():
    assert v.check_coordinates() == []


def test_normalized_hours_and_prices():
    assert v.check_hours_and_prices() == []


def test_chunk_parity_json():
    assert v.check_chunks_json() == []


def test_chunk_parity_qdrant():
    assert v.check_qdrant() == []


# --------------------------------------------------------------------------- API smoke
@pytest.fixture(scope="module")
def client():
    clear_cache()
    with TestClient(app) as c:
        yield c
    clear_cache()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_list_attractions(client):
    r = client.get("/api/v1/attractions")
    assert r.status_code == 200
    assert len(r.json()) == 29


def test_list_attractions_filters(client):
    giza = client.get("/api/v1/attractions", params={"city": "giza"}).json()
    assert {a["attraction_id"] for a in giza} == {"GIZ001", "GIZ002", "GIZ003", "GIZ004", "GIZ005"}
    museums = client.get("/api/v1/attractions", params={"category": "Museum", "max_duration_minutes": 150}).json()
    assert museums and all("museum" in a["category"].lower() and a["visit_duration_minutes"] <= 150 for a in museums)


def test_get_attraction_detail(client):
    r = client.get("/api/v1/attractions/GIZ003")
    assert r.status_code == 200
    body = r.json()
    assert body["name_en"] == "Grand Egyptian Museum (GEM)"
    assert len(body["opening_hours"]) == 4
    assert body["ticket_prices"][0]["price_foreigners_egp"] == 1570.0


def test_get_attraction_404(client):
    assert client.get("/api/v1/attractions/NOPE999").status_code == 404


def test_patch_status_invalidates_cache(client):
    aid = "LXR001"
    assert client.get(f"/api/v1/attractions/{aid}").json()["is_open"] is True  # warm cache
    try:
        r = client.patch(f"/api/v1/attractions/{aid}/status",
                         json={"is_open": False, "closure_reason": "Restoration works"})
        assert r.status_code == 200
        detail = client.get(f"/api/v1/attractions/{aid}").json()
        assert detail["is_open"] is False and detail["closure_reason"] == "Restoration works"
        closed_ids = {a["attraction_id"] for a in client.get("/api/v1/attractions", params={"is_open": False}).json()}
        assert aid in closed_ids
    finally:
        client.patch(f"/api/v1/attractions/{aid}/status", json={"is_open": True})
    restored = client.get(f"/api/v1/attractions/{aid}").json()
    assert restored["is_open"] is True and restored["closure_reason"] is None
