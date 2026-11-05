from typing import List, Optional

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from app.db.database import get_session
from app.models import Attraction
from app.schemas import (
    AttractionDetailResponse,
    AttractionStatusUpdate,
    AttractionSummary,
    OpeningHourSchema,
    TicketPriceSchema,
)

router = APIRouter(prefix="/api/v1/attractions", tags=["attractions"])

# attraction_id -> AttractionDetailResponse
_detail_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def clear_cache() -> None:
    _detail_cache.clear()


def _to_detail(a: Attraction) -> AttractionDetailResponse:
    return AttractionDetailResponse(
        attraction_id=a.attraction_id,
        name_en=a.name_en,
        name_ar=a.name_ar,
        city=a.city,
        category=a.category,
        visit_duration_minutes=a.visit_duration_minutes,
        latitude=a.latitude,
        longitude=a.longitude,
        official_url=a.official_url,
        is_open=a.is_open,
        closure_reason=a.closure_reason,
        opening_hours=[OpeningHourSchema.model_validate(h, from_attributes=True) for h in a.opening_hours],
        ticket_prices=[TicketPriceSchema.model_validate(t, from_attributes=True) for t in a.ticket_prices],
    )


def _load(session: Session, attraction_id: str) -> Attraction:
    stmt = (
        select(Attraction)
        .where(Attraction.attraction_id == attraction_id)
        .options(selectinload(Attraction.opening_hours), selectinload(Attraction.ticket_prices))
    )
    attraction = session.exec(stmt).first()
    if attraction is None:
        raise HTTPException(status_code=404, detail=f"Attraction '{attraction_id}' not found")
    return attraction


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
def get_attraction(attraction_id: str, session: Session = Depends(get_session)) -> AttractionDetailResponse:
    key = attraction_id.upper()
    cached = _detail_cache.get(key)
    if cached is not None:
        return cached
    detail = _to_detail(_load(session, key))
    _detail_cache[key] = detail
    return detail


@router.patch("/{attraction_id}/status", response_model=AttractionDetailResponse)
def update_status(
    attraction_id: str,
    payload: AttractionStatusUpdate,
    session: Session = Depends(get_session),
) -> AttractionDetailResponse:
    key = attraction_id.upper()
    attraction = _load(session, key)
    attraction.is_open = payload.is_open
    attraction.closure_reason = None if payload.is_open else payload.closure_reason
    session.add(attraction)
    session.commit()
    _detail_cache.pop(key, None)
    detail = _to_detail(_load(session, key))
    _detail_cache[key] = detail
    return detail
