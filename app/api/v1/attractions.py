import time
from typing import List, Optional

from cachetools import TTLCache
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlmodel import Session, col, select

from app.db.database import get_session
from app.models import Attraction
from app.schemas import AttractionDetailResponse, AttractionStatusUpdate, AttractionSummary
from app.services import analytics_service
from app.services.attraction_service import load_details
from app.services.rag_service import invalidate_rag_cache

router = APIRouter(prefix="/api/v1/attractions", tags=["attractions"])

# attraction_id -> AttractionDetailResponse
_detail_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def clear_cache() -> None:
    _detail_cache.clear()


def _load(session: Session, attraction_id: str) -> AttractionDetailResponse:
    details = load_details(session, [attraction_id])
    if not details:
        raise HTTPException(status_code=404, detail=f"Attraction '{attraction_id}' not found")
    return details[0]


@router.get("", response_model=List[AttractionSummary])
def list_attractions(
    city: Optional[str] = Query(None, description="Case-insensitive substring, e.g. 'Giza'"),
    category: Optional[str] = Query(None, description="Case-insensitive substring, e.g. 'Museum'"),
    max_duration_minutes: Optional[int] = Query(None, ge=0),
    is_open: Optional[bool] = None,
    session: Session = Depends(get_session),
) -> List[Attraction]:
    stmt = select(Attraction)
    if city:
        stmt = stmt.where(col(Attraction.city).ilike(f"%{city.strip()}%"))
    if category:
        stmt = stmt.where(col(Attraction.category).ilike(f"%{category.strip()}%"))
    if max_duration_minutes is not None:
        stmt = stmt.where(Attraction.visit_duration_minutes <= max_duration_minutes)
    if is_open is not None:
        stmt = stmt.where(Attraction.is_open == is_open)
    return list(session.exec(stmt.order_by(Attraction.attraction_id)).all())


@router.get("/{attraction_id}", response_model=AttractionDetailResponse)
def get_attraction(
    attraction_id: str,
    session: Session = Depends(get_session),
    x_session_id: Optional[str] = Header(None, description="Optional client session id for analytics"),
    x_language: Optional[str] = Header(None, description="Optional UI language (en/ar) for analytics"),
) -> AttractionDetailResponse:
    started = time.perf_counter()
    key = attraction_id.upper()
    detail = _detail_cache.get(key)
    cache_hit = detail is not None
    if detail is None:
        detail = _load(session, key)
        _detail_cache[key] = detail
    analytics_service.log_event(
        session_id=x_session_id or "anonymous",
        event_type="attraction_lookup",
        language=x_language if x_language in ("en", "ar") else "en",
        city=detail.city,
        attraction_ids=[detail.attraction_id],
        cache_hit=cache_hit,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return detail


@router.patch("/{attraction_id}/status", response_model=AttractionDetailResponse)
def update_status(
    attraction_id: str,
    payload: AttractionStatusUpdate,
    session: Session = Depends(get_session),
) -> AttractionDetailResponse:
    key = attraction_id.upper()
    attraction = session.get(Attraction, key)
    if attraction is None:
        raise HTTPException(status_code=404, detail=f"Attraction '{attraction_id}' not found")
    attraction.is_open = payload.is_open
    attraction.closure_reason = None if payload.is_open else payload.closure_reason
    session.add(attraction)
    session.commit()
    _detail_cache.pop(key, None)
    invalidate_rag_cache(key)  # chat answers that cited this attraction are now stale
    detail = _load(session, key)
    _detail_cache[key] = detail
    return detail
