"""Shared read access to verified attraction records (SQLite)."""
from typing import Dict, List, Sequence

from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from app.models import Attraction
from app.schemas import AttractionDetailResponse, OpeningHourSchema, TicketPriceSchema


def to_detail(a: Attraction) -> AttractionDetailResponse:
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


def load_details(session: Session, attraction_ids: Sequence[str]) -> List[AttractionDetailResponse]:
    """Live SQLite join (status + hours + prices), returned in the order of attraction_ids."""
    if not attraction_ids:
        return []
    stmt = (
        select(Attraction)
        .where(col(Attraction.attraction_id).in_(list(attraction_ids)))
        .options(selectinload(Attraction.opening_hours), selectinload(Attraction.ticket_prices))
    )
    by_id: Dict[str, Attraction] = {a.attraction_id: a for a in session.exec(stmt).all()}
    return [to_detail(by_id[i]) for i in attraction_ids if i in by_id]
