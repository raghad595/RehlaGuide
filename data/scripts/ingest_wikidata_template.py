"""TEMPLATE / CONTRACT for the Tier 2 (unverified) Wikidata collection.

Who:   teammate preparing the extended Egyptian-monuments dataset from Wikidata SPARQL.
What:  write records into the Qdrant collection the chat API already reads as its
       fallback tier. If you follow this file, the API picks your data up with no code
       changes (it checks the collection on every query).

    Collection : "rehlaguide_wikidata_unverified"   (settings.unverified_collection)
    Storage    : ./data/qdrant_db                    (settings.qdrant_path, same folder as Tier 1)
    Embeddings : intfloat/multilingual-e5-small, 384-d, cosine, "passage: " prefix
                 -> ALWAYS go through app.services.vector_store.embed_passages(); never load
                    the model yourself, or the vectors won't be comparable with queries.

REQUIRED payload per point (enforced by validate_record below):
    chunk_id      str   unique and stable, e.g. "Q1234567:en:0"  (point id = uuid5(chunk_id))
    wikidata_qid  str   e.g. "Q1234567"
    name_en       str   English label   (at least one of name_en / name_ar)
    name_ar       str   Arabic label
    language      "en" | "ar"   language of `text` (the API filters by the user's language)
    city          str   governorate/city label, e.g. "Cairo"
    category      str   e.g. "mosque", "temple", "museum" (free text; Wikidata P31 label)
    source_url    str   https://www.wikidata.org/wiki/Q...
    text          str   historical / architectural description to embed (non-empty)
    is_verified   False                       (forced)
    trust_tier    "unverified_historical_only" (forced)
OPTIONAL: latitude, longitude (floats, WGS84), inception (str).

FORBIDDEN: opening hours, visiting times, ticket prices, fees, phone numbers. Wikidata has
no trustworthy operational data for these sites, so keys such as `opening_hours`,
`ticket_price`, `price`, `fee`, `hours` are rejected, and the API additionally strips any
sentence with clock times / money amounts from `text` at query time.

USAGE (stop the API server first: Qdrant local mode allows one process per folder):
    $env:PYTHONPATH="."
    .venv\\Scripts\\python.exe data/scripts/ingest_wikidata_template.py --demo            # validate 2 sample records, write nothing
    .venv\\Scripts\\python.exe data/scripts/ingest_wikidata_template.py --demo --write    # actually index the samples
In your own script:
    from data.scripts.ingest_wikidata_template import index_wikidata_records
    index_wikidata_records(records)                  # upsert (idempotent by chunk_id)
    index_wikidata_records(records, recreate=True)   # wipe the collection first

Text tips for good retrieval: start `text` with the site's name and type, e.g.
"Mosque of Ibn Tulun is a 9th-century mosque in Cairo ...". Long descriptions can be
split into several records (chunk_id suffix :0, :1, ...) with the same qid. Provide an
Arabic record (language="ar", Arabic `text`) whenever Wikidata has an Arabic description,
because Arabic questions search Arabic text first.
"""
from __future__ import annotations

import argparse
import sys
import uuid
import warnings
from typing import Any, Dict, Iterable, List

from app.core.config import get_settings
from app.services.vector_store import close_qdrant_client, embed_passages, get_qdrant_client

COLLECTION = get_settings().unverified_collection  # "rehlaguide_wikidata_unverified"

REQUIRED_FIELDS = ("chunk_id", "wikidata_qid", "language", "city", "category", "source_url", "text")
OPTIONAL_FIELDS = ("name_en", "name_ar", "latitude", "longitude", "inception")
FORBIDDEN_KEY_PARTS = ("hour", "opening", "closing", "price", "ticket", "fee", "admission", "phone")
FIXED_FIELDS = {"is_verified": False, "trust_tier": "unverified_historical_only"}


class RecordError(ValueError):
    pass


def validate_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the clean payload for one record, or raise RecordError."""
    bad_keys = [k for k in rec if any(part in k.lower() for part in FORBIDDEN_KEY_PARTS)]
    if bad_keys:
        raise RecordError(f"{rec.get('chunk_id')}: operational fields are not allowed in Tier 2: {bad_keys}")
    missing = [k for k in REQUIRED_FIELDS if not str(rec.get(k) or "").strip()]
    if missing:
        raise RecordError(f"{rec.get('chunk_id')}: missing required fields {missing}")
    if not (rec.get("name_en") or rec.get("name_ar")):
        raise RecordError(f"{rec['chunk_id']}: needs name_en or name_ar")
    if rec["language"] not in ("en", "ar"):
        raise RecordError(f"{rec['chunk_id']}: language must be 'en' or 'ar'")
    if not str(rec["wikidata_qid"]).startswith("Q"):
        raise RecordError(f"{rec['chunk_id']}: wikidata_qid must look like 'Q12345'")
    payload = {k: rec[k] for k in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS) if rec.get(k) is not None}
    for k in ("latitude", "longitude"):
        if k in payload:
            payload[k] = float(payload[k])
    payload.update(FIXED_FIELDS)
    return payload


def chunk_text_for_embedding(p: Dict[str, Any]) -> str:
    """Same idea as Tier 1's contextual header: name + place + type, then the description."""
    if p["language"] == "ar":
        header = f"الموقع: {p.get('name_ar') or p.get('name_en')} | المدينة: {p['city']} | الفئة: {p['category']}"
    else:
        header = f"Monument: {p.get('name_en') or p.get('name_ar')} | City: {p['city']} | Category: {p['category']}"
    return f"{header}\n\n{p['text']}"


def index_wikidata_records(records: Iterable[Dict[str, Any]], recreate: bool = False,
                           batch_size: int = 256) -> int:
    """Validate, embed and upsert records into the Tier 2 collection. Returns the point count."""
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

    payloads: List[Dict[str, Any]] = [validate_record(r) for r in records]
    ids = [p["chunk_id"] for p in payloads]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RecordError(f"duplicate chunk_id(s): {sorted(dupes)[:10]}")

    client = get_qdrant_client()
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        dim = embed_passages(["dimension probe"]).shape[1]
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # local mode ignores payload indexes
            for f in ("wikidata_qid", "language", "city", "category"):
                client.create_payload_index(COLLECTION, f, PayloadSchemaType.KEYWORD)

    for start in range(0, len(payloads), batch_size):
        batch = payloads[start:start + batch_size]
        vectors = embed_passages([chunk_text_for_embedding(p) for p in batch])
        client.upsert(COLLECTION, points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, p["chunk_id"])), vector=v.tolist(),
                        payload={**p, "text": chunk_text_for_embedding(p)})
            for p, v in zip(batch, vectors)
        ])
    return client.count(COLLECTION, exact=True).count


# --------------------------------------------------------------------------- SPARQL starting point
# Example query (run against https://query.wikidata.org/sparql with an
# "Accept: application/sparql-results+json" header and a descriptive User-Agent):
EXAMPLE_SPARQL = """
SELECT ?site ?siteLabel_en ?siteLabel_ar ?descEn ?descAr ?coord ?cityLabel ?typeLabel WHERE {
  ?site wdt:P17 wd:Q79 ;                      # country: Egypt
        wdt:P31/wdt:P279* wd:Q4989906 ;       # instance of (subclass of) monument
        wdt:P625 ?coord .
  OPTIONAL { ?site rdfs:label ?siteLabel_en FILTER(LANG(?siteLabel_en) = "en") }
  OPTIONAL { ?site rdfs:label ?siteLabel_ar FILTER(LANG(?siteLabel_ar) = "ar") }
  OPTIONAL { ?site schema:description ?descEn FILTER(LANG(?descEn) = "en") }
  OPTIONAL { ?site schema:description ?descAr FILTER(LANG(?descAr) = "ar") }
  OPTIONAL { ?site wdt:P131 ?city . }
  OPTIONAL { ?site wdt:P31 ?type . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 5000
"""

# Demo records use a PLACEHOLDER qid (Q0000001); real records must use the actual Wikidata QID.
DEMO_RECORDS = [
    {
        "chunk_id": "Q0000001:en:0", "wikidata_qid": "Q0000001",
        "name_en": "Mosque of Ibn Tulun", "name_ar": "مسجد أحمد بن طولون",
        "language": "en", "city": "Cairo", "category": "mosque",
        "latitude": 30.0287, "longitude": 31.2495,
        "source_url": "https://www.wikidata.org/wiki/Q0000001",
        "text": ("The Mosque of Ibn Tulun is a 9th-century mosque in Cairo, built by Ahmad ibn Tulun. "
                 "It is one of the oldest mosques in Egypt surviving in its original form and is known "
                 "for its spiral minaret and large open courtyard."),
    },
    {
        "chunk_id": "Q0000001:ar:0", "wikidata_qid": "Q0000001",
        "name_en": "Mosque of Ibn Tulun", "name_ar": "مسجد أحمد بن طولون",
        "language": "ar", "city": "Cairo", "category": "mosque",
        "latitude": 30.0287, "longitude": 31.2495,
        "source_url": "https://www.wikidata.org/wiki/Q0000001",
        "text": ("مسجد أحمد بن طولون مسجد يعود إلى القرن التاسع الميلادي في القاهرة، بناه أحمد بن طولون. "
                 "وهو من أقدم مساجد مصر الباقية على هيئتها الأصلية، ويشتهر بمئذنته الملوية وصحنه الواسع."),
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true", help="use the built-in sample records")
    ap.add_argument("--write", action="store_true", help="actually write to Qdrant (default: validate only)")
    args = ap.parse_args()
    if not args.demo:
        print("Import index_wikidata_records() from your own script, or run with --demo. See the module docstring.")
        return 0
    for r in DEMO_RECORDS:
        validate_record(r)
    print(f"{len(DEMO_RECORDS)} demo records are valid for collection '{COLLECTION}'.")
    if args.write:
        try:
            n = index_wikidata_records(DEMO_RECORDS)
            print(f"Indexed. '{COLLECTION}' now has {n} points.")
        finally:
            close_qdrant_client()
    return 0


if __name__ == "__main__":
    sys.exit(main())
