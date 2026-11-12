"""Two-tier hybrid retrieval.

Tier 1 (verified):   Qdrant `rehlaguide_attractions` + live SQLite join (status, hours, prices).
Tier 2 (unverified): Qdrant `rehlaguide_wikidata_unverified` (historical context ONLY).

Routing (first rule that applies wins):
  1. Query names a verified attraction (alias match)                       -> verified
  2. Query names an unverified Wikidata site (name-token match on hits)    -> unverified_fallback
  3. Discovery query ("which museums in Cairo...") whose city/category
     filters match verified attractions                                    -> verified
  4. Best Tier 1 similarity >= tier1_min_score, and the query names no
     proper noun absent from the verified texts (e.g. "Roda")              -> verified
  5. Best Tier 2 similarity >= tier2_min_score                             -> unverified_fallback
  6. otherwise                                                             -> none

Why names first: multilingual-e5 scores are compressed. In Phase 2 calibration,
monuments NOT in the dataset (Edfu, Ibn Tulun, the Nilometer) scored 0.80-0.84 against
unrelated verified chunks, overlapping unnamed in-domain questions (0.79-0.90). So the
score threshold (0.84) is tuned for precision, and name matching carries recall.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Hashable, List, Literal, Optional, Sequence, Tuple

from cachetools import TTLCache
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import Attraction
from app.schemas import AttractionDetailResponse, UnverifiedSource
from app.services import vector_store
from app.services.attraction_service import load_details
from app.services.query_parsing import (
    AliasMatcher,
    QueryFilters,
    detect_language,
    extract_filters,
    name_tokens_in_query,
    normalize,
)

__all__ = [
    "detect_language", "extract_filters", "retrieve", "RetrievalResult", "invalidate_rag_cache",
    "cache_key", "cache_get", "cache_put", "reset_catalog",
]

Tier = Literal["verified", "unverified_fallback", "none"]

MAX_ATTRACTIONS = 3          # attractions per verified answer (semantic / name)
MAX_DISCOVERY_ATTRACTIONS = 5
SEMANTIC_BAND = 0.03         # also include attractions within this score of the best hit
MAX_UNVERIFIED_SITES = 2

# Payload fields allowed out of the unverified collection. Anything else (opening hours,
# prices, phone numbers, ...) is dropped before it can reach a prompt or an answer.
UNVERIFIED_ALLOWED_FIELDS = (
    "chunk_id", "wikidata_qid", "name_en", "name_ar", "language", "city", "category",
    "latitude", "longitude", "source_url", "text",
)

# Numeric operational facts: clock times and money amounts.
OPERATIONAL_NUMBERS_RE = re.compile(
    r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"
    r"|\b\d{1,2}\s?(?:a\.?m\.?|p\.?m\.?)(?![a-z])"
    r"|\b(?:egp|usd|le|eur)\s?\d|\d[\d,.]*\s?(?:egp|usd|le\b|eur|pounds?|dollars?|euros?)"
    r"|[$€£]\s?\d|\d\s?(?:جنيه|جنيها|دولار|يورو)",
    re.IGNORECASE,
)
# Wording that marks a sentence as operational (used to strip such sentences from Tier 2 text).
OPERATIONAL_WORDS_RE = re.compile(
    r"opening hours|open daily|open from|closing time|ticket|admission|entry fee|entrance fee|"
    r"\bprice|\bfee\b|مواعيد|ساعات العمل|تذكر|تذاكر|رسوم|سعر|أسعار",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+|\n+")

LIST_CUES_RE = re.compile(
    r"\b(which|what are|list|best|top|recommend|suggest|any|options|places)\b"
    r"|ما هي|ما هى|ماهي|ما افضل|افضل|اقترح|ترشح|رشح|اماكن|قائمه|اي ",
)


# --------------------------------------------------------------------------- data classes
@dataclass
class RetrievedChunk:
    score: float
    payload: Dict[str, Any]

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


@dataclass
class RetrievalResult:
    query: str
    language: Literal["en", "ar"]
    tier: Tier
    match_method: Literal["name", "unverified_name", "discovery", "semantic", "none"]
    filters: QueryFilters
    attractions: List[AttractionDetailResponse] = field(default_factory=list)
    verified_chunks: List[RetrievedChunk] = field(default_factory=list)
    unverified_chunks: List[RetrievedChunk] = field(default_factory=list)
    top_score_verified: float = 0.0
    top_score_unverified: float = 0.0
    tier2_available: bool = False

    @property
    def attraction_ids(self) -> List[str]:
        return [a.attraction_id for a in self.attractions]

    @property
    def city(self) -> Optional[str]:
        if self.filters.city_label:
            return self.filters.city_label
        if self.attractions:
            return self.attractions[0].city
        if self.unverified_chunks:
            return self.unverified_chunks[0].payload.get("city")
        return None

    def unverified_sources(self) -> List[UnverifiedSource]:
        seen, out = set(), []
        for c in self.unverified_chunks:
            p = c.payload
            key = p.get("wikidata_qid") or p.get("name_en") or p.get("chunk_id")
            if key in seen:
                continue
            seen.add(key)
            out.append(UnverifiedSource(**{k: p.get(k) for k in UnverifiedSource.model_fields if k != "is_verified"}))
        return out

    # ---- prompt context blocks --------------------------------------------------
    def verified_operational_block(self) -> str:
        return "<verified_operational_data>\n" + "\n\n".join(
            format_operational_record(a) for a in self.attractions
        ) + "\n</verified_operational_data>"

    def verified_chunks_block(self) -> str:
        body = "\n".join(
            f'<chunk attraction_id="{c.payload.get("attraction_id")}" section="{c.payload.get("chunk_type")}" '
            f'score="{c.score:.3f}">\n{c.text}\n</chunk>'
            for c in self.verified_chunks
        )
        return f"<verified_narrative_chunks>\n{body}\n</verified_narrative_chunks>"

    def unverified_block(self) -> str:
        parts = []
        for c in self.unverified_chunks:
            p = c.payload
            coords = (f'{p["latitude"]}, {p["longitude"]}'
                      if p.get("latitude") is not None and p.get("longitude") is not None else "unknown")
            parts.append(
                f'<site wikidata_qid="{p.get("wikidata_qid") or ""}" name_en="{p.get("name_en") or ""}" '
                f'name_ar="{p.get("name_ar") or ""}" city="{p.get("city") or ""}" coordinates="{coords}" '
                f'source_url="{p.get("source_url") or ""}">\n{c.text}\n</site>'
            )
        return ('<unverified_supplementary_chunks source="Wikidata">\n' + "\n".join(parts)
                + "\n</unverified_supplementary_chunks>")

    def context_blocks(self) -> str:
        if self.tier == "verified":
            return self.verified_operational_block() + "\n\n" + self.verified_chunks_block()
        if self.tier == "unverified_fallback":
            return self.unverified_block()
        return "<no_matching_data/>"


def format_operational_record(a: AttractionDetailResponse) -> str:
    lines = [
        f"[{a.attraction_id}] {a.name_en} | {a.name_ar} | City: {a.city} | Category: {a.category}",
        "Status: OPEN" if a.is_open else f"Status: CLOSED — reason: {a.closure_reason or 'not specified'}",
        f"Typical visit duration: {a.visit_duration_minutes} minutes",
        f"Official URL: {a.official_url or 'n/a'}",
        "Opening hours:",
    ]
    for h in a.opening_hours:
        line = (f"  - {h.area_or_session}: days={h.opening_days_raw} | open={h.opening_hour_raw} | "
                f"close={h.closing_hour_raw}")
        if h.last_entry_hour_raw:
            line += f" | last entry={h.last_entry_hour_raw}"
        line += f" | normalized: days={h.open_days_list} {h.open_time_24h or '?'}-{h.close_time_24h or '?'}"
        if h.notes:
            line += f" | notes: {h.notes}"
        line += f" | source={h.source_url or 'n/a'} | last_verified_at={h.last_verified_at or 'n/a'}"
        lines.append(line)
    lines.append("Ticket prices:")
    for t in a.ticket_prices:
        line = (f"  - {t.ticket_type}: locals/Egyptians={t.price_for_locals_raw} | "
                f"foreigners={t.price_for_foreigners_raw}")
        if t.children_ticket_price_raw:
            line += f" | children={t.children_ticket_price_raw}"
        if t.free_entry_cases:
            line += f" | free entry: {t.free_entry_cases}"
        line += f" | source={t.source_url or 'n/a'} | last_verified_at={t.last_verified_at or 'n/a'}"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- catalog
@dataclass(frozen=True)
class _Catalog:
    matcher: AliasMatcher
    cities: Tuple[str, ...]
    categories: Tuple[str, ...]


@lru_cache(maxsize=1)
def _catalog() -> _Catalog:
    from app.db.database import engine

    with Session(engine) as s:
        rows = s.exec(select(Attraction)).all()
    return _Catalog(
        matcher=AliasMatcher([(a.attraction_id, a.name_en, a.name_ar) for a in rows]),
        cities=tuple(sorted({a.city for a in rows})),
        categories=tuple(sorted({a.category for a in rows})),
    )


def reset_catalog() -> None:
    """Call after re-ingesting attractions so new names/cities are picked up."""
    _catalog.cache_clear()
    _verified_vocabulary.cache_clear()


@lru_cache(maxsize=1)
def _verified_vocabulary() -> frozenset:
    """Every normalized word in the verified chunks (both languages) + attraction names."""
    client = vector_store.get_qdrant_client()
    words: set = set()
    offset = None
    while True:
        points, offset = client.scroll(get_settings().verified_collection, limit=256, offset=offset,
                                       with_payload=["text", "name_en", "name_ar"])
        for p in points:
            for k in ("text", "name_en", "name_ar"):
                words.update(normalize((p.payload or {}).get(k) or "").split())
        if offset is None:
            break
    return frozenset(words)


_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z'’]{2,})")
_NOT_ENTITIES = {"I", "Egypt", "Egyptian", "Egyptians", "Arabic", "English", "Islamic", "Ramadan",
                 "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                 "AM", "PM", "EGP", "USD", "UNESCO"}


def unknown_proper_nouns(query: str) -> List[str]:
    """Capitalized, non-sentence-initial English words that never occur in the verified data.

    Such a word means the user is naming something RehlaGuide doesn't cover ("the Nilometer
    on Roda Island"), so a merely *similar* verified chunk must not be used to answer."""
    with vector_store.search_lock:  # first call scrolls Qdrant
        vocab = _verified_vocabulary()
    unknown = []
    for m in _PROPER_NOUN_RE.finditer(query or ""):
        word = m.group(1)
        if word in _NOT_ENTITIES:
            continue
        w = normalize(word)
        if w in vocab or (len(w) > 4 and (w.rstrip("s") in vocab or w + "s" in vocab)):
            continue
        unknown.append(word)
    return unknown


# --------------------------------------------------------------------------- qdrant helpers
def _search(collection: str, vector: List[float], must: Sequence[Any], limit: int) -> List[RetrievedChunk]:
    from qdrant_client.models import Filter

    client = vector_store.get_qdrant_client()
    points = client.query_points(
        collection, query=vector, query_filter=Filter(must=list(must)) if must else None,
        limit=limit, with_payload=True,
    ).points
    return [RetrievedChunk(score=float(p.score), payload=dict(p.payload or {})) for p in points]


def _match(key: str, value: Any):
    from qdrant_client.models import FieldCondition, MatchValue

    return FieldCondition(key=key, match=MatchValue(value=value))


def _match_any(key: str, values: Sequence[Any]):
    from qdrant_client.models import FieldCondition, MatchAny

    return FieldCondition(key=key, match=MatchAny(any=list(values)))


def _lte(key: str, value: float):
    from qdrant_client.models import FieldCondition, Range

    return FieldCondition(key=key, range=Range(lte=value))


def _filter_conditions(f: QueryFilters) -> List[Any]:
    conds = []
    if f.cities:
        conds.append(_match_any("city", f.cities))
    if f.categories:
        conds.append(_match_any("category", f.categories))
    if f.max_duration_minutes:
        conds.append(_lte("visit_duration_minutes", f.max_duration_minutes))
    return conds


def _distinct(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


def sanitize_unverified_text(text: str) -> str:
    """Drop any sentence that states opening hours / prices (Tier 2 is history-only).

    A leading context-header line ("Monument: X | City: Y | ...") is kept as-is, separated
    by a blank line, so answer formatting can still recognise and strip it."""
    text = text or ""
    header = ""
    first, sep, rest = text.partition("\n\n")
    if sep and " | " in first and "\n" not in first:
        header, text = first, rest
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    kept = [s for s in sentences if not (OPERATIONAL_NUMBERS_RE.search(s) or OPERATIONAL_WORDS_RE.search(s))]
    body = " ".join(kept)
    return f"{header}\n\n{body}" if header and body else body


def _sanitize_unverified(chunk: RetrievedChunk) -> RetrievedChunk:
    payload = {k: chunk.payload.get(k) for k in UNVERIFIED_ALLOWED_FIELDS if k in chunk.payload}
    payload["text"] = sanitize_unverified_text(payload.get("text") or "")
    return RetrievedChunk(score=chunk.score, payload=payload)


def _site_key(c: RetrievedChunk) -> str:
    return c.payload.get("wikidata_qid") or c.payload.get("name_en") or c.payload.get("chunk_id") or ""


# --------------------------------------------------------------------------- retrieval
def retrieve(query: str, language: Literal["en", "ar"], session: Session) -> RetrievalResult:
    settings = get_settings()
    cat = _catalog()
    filters = extract_filters(query, cat.cities, cat.categories)
    result = RetrievalResult(query=query, language=language, tier="none", match_method="none", filters=filters)
    lang_cond = _match("language", language)

    with vector_store.search_lock:
        vector = vector_store.embed_query(query)

        # 1. Named verified attraction(s).
        named_ids = cat.matcher.match(query)[:MAX_ATTRACTIONS]
        if named_ids:
            hits = _search(settings.verified_collection, vector,
                           [lang_cond, _match_any("attraction_id", named_ids)], limit=3 * len(named_ids))
            return _finish_verified(result, session, named_ids, hits, "name")

        # Tier 1 semantic search (filtered if the query has filters; unfiltered as fallback).
        conds = _filter_conditions(filters)
        t1_hits = _search(settings.verified_collection, vector, [lang_cond, *conds], limit=30 if conds else settings.rag_top_k)
        filtered_hits = t1_hits if conds else []
        if conds and not t1_hits:
            t1_hits = _search(settings.verified_collection, vector, [lang_cond], limit=settings.rag_top_k)
        result.top_score_verified = t1_hits[0].score if t1_hits else 0.0

        # Tier 2 search (only if the teammate's collection exists and has points).
        t2_hits: List[RetrievedChunk] = []
        result.tier2_available = vector_store.collection_point_count(settings.unverified_collection) > 0
        if result.tier2_available:
            t2_hits = _search(settings.unverified_collection, vector, [lang_cond], limit=10) \
                or _search(settings.unverified_collection, vector, [], limit=10)
        result.top_score_unverified = t2_hits[0].score if t2_hits else 0.0

    # 2. Named unverified site.
    named_t2 = [h for h in t2_hits if any(
        name_tokens_in_query(h.payload.get(k) or "", query) for k in ("name_en", "name_ar"))]
    if named_t2:
        return _finish_unverified(result, named_t2, "unverified_name")

    # 3. Discovery query constrained by city/category that matches verified attractions.
    if filtered_hits and (filters.cities or filters.categories) and LIST_CUES_RE.search(normalize(query)):
        ids = _distinct([h.payload.get("attraction_id") for h in filtered_hits])[:MAX_DISCOVERY_ATTRACTIONS]
        return _finish_verified(result, session, ids, filtered_hits, "discovery")

    # 4. Strong Tier 1 semantic match, unless the user names something we don't cover.
    if (t1_hits and result.top_score_verified >= settings.tier1_min_score
            and not unknown_proper_nouns(query)):
        band = [h for h in t1_hits if h.score >= result.top_score_verified - SEMANTIC_BAND]
        ids = _distinct([h.payload.get("attraction_id") for h in band])[:MAX_ATTRACTIONS]
        return _finish_verified(result, session, ids, t1_hits, "semantic")

    # 5. Strong Tier 2 semantic match.
    if t2_hits and result.top_score_unverified >= settings.tier2_min_score:
        best = result.top_score_unverified
        return _finish_unverified(result, [h for h in t2_hits if h.score >= best - 0.02], "semantic")

    return result  # 6. none


def _finish_verified(result: RetrievalResult, session: Session, ids: List[str],
                     hits: List[RetrievedChunk], method) -> RetrievalResult:
    result.tier = "verified"
    result.match_method = method
    result.attractions = load_details(session, ids)  # live SQLite: status, hours, prices
    wanted = set(ids)
    relevant = [h for h in hits if h.payload.get("attraction_id") in wanted]
    # Best chunk per attraction first, then the remaining chunks by score.
    best_per: Dict[str, RetrievedChunk] = {}
    for h in relevant:
        best_per.setdefault(h.payload["attraction_id"], h)
    firsts = [best_per[i] for i in ids if i in best_per]
    rest = [h for h in relevant if h not in firsts]
    result.verified_chunks = (firsts + rest)[: max(get_settings().rag_top_k, len(firsts))]
    if result.verified_chunks:
        result.top_score_verified = max(result.top_score_verified, result.verified_chunks[0].score)
    return result


def _finish_unverified(result: RetrievalResult, hits: List[RetrievedChunk], method) -> RetrievalResult:
    sites = _distinct([_site_key(h) for h in hits])[:MAX_UNVERIFIED_SITES]
    chunks = [_sanitize_unverified(h) for h in hits if _site_key(h) in sites]
    result.tier = "unverified_fallback"
    result.match_method = method
    result.unverified_chunks = [c for c in chunks if c.text][:4] or chunks[:2]
    return result


# --------------------------------------------------------------------------- answer cache
# Caches finished chat answers keyed by (language, normalized question). Entries remember
# which verified attractions they used, so a status change invalidates only those answers.
_cache: TTLCache = TTLCache(maxsize=get_settings().rag_cache_maxsize, ttl=get_settings().rag_cache_ttl_seconds)
_cache_lock = threading.Lock()


def cache_key(message: str, language: str) -> Tuple[str, str]:
    return (language, normalize(message))


def cache_get(key: Hashable) -> Optional[Any]:
    with _cache_lock:
        entry = _cache.get(key)
    return entry[0] if entry else None


def cache_put(key: Hashable, value: Any, attraction_ids: Sequence[str]) -> None:
    with _cache_lock:
        _cache[key] = (value, frozenset(attraction_ids))


def invalidate_rag_cache(attraction_id: Optional[str] = None) -> int:
    """Drop cached answers that used `attraction_id` (or everything if None). Returns count removed."""
    with _cache_lock:
        if attraction_id is None:
            n = len(_cache)
            _cache.clear()
            return n
        aid = attraction_id.upper()
        stale = [k for k, (_, ids) in list(_cache.items()) if aid in ids]
        for k in stale:
            _cache.pop(k, None)
        return len(stale)
