"""Analytics event logging + aggregation (SQLite table `analytics_events`)."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, Optional

from sqlmodel import Session, col, select

from app.db.database import engine
from app.models import AnalyticsEvent, Attraction
from app.schemas import AnalyticsSummaryResponse, AttractionCount, CityCount

log = logging.getLogger("rehlaguide.analytics")


def log_event(
    *,
    session_id: str,
    event_type: str,
    language: str,
    data_tier_used: str = "verified",
    city: Optional[str] = None,
    attraction_ids: Iterable[str] = (),
    cache_hit: bool = False,
    latency_ms: int = 0,
) -> None:
    """Best effort: an analytics failure must never break a user request."""
    try:
        with Session(engine) as s:
            s.add(AnalyticsEvent(
                session_id=session_id, event_type=event_type, language=language,
                data_tier_used=data_tier_used, city=city, attraction_ids_json=list(attraction_ids),
                cache_hit=cache_hit, latency_ms=int(latency_ms),
            ))
            s.commit()
    except Exception:  # pragma: no cover - logged, not raised
        log.exception("failed to log analytics event")


def summarize(session: Session, event_type: Optional[str] = None, top_n: int = 5) -> AnalyticsSummaryResponse:
    stmt = select(AnalyticsEvent)
    if event_type:
        stmt = stmt.where(AnalyticsEvent.event_type == event_type)
    events = session.exec(stmt).all()
    total = len(events)

    attraction_counts = Counter(aid for e in events for aid in (e.attraction_ids_json or []))
    top_ids = [aid for aid, _ in attraction_counts.most_common(top_n)]
    names = {}
    if top_ids:
        rows = session.exec(select(Attraction).where(col(Attraction.attraction_id).in_(top_ids))).all()
        names = {a.attraction_id: a for a in rows}

    return AnalyticsSummaryResponse(
        total_queries=total,
        cache_hit_rate_pct=round(100.0 * sum(e.cache_hit for e in events) / total, 2) if total else 0.0,
        average_latency_ms=round(sum(e.latency_ms for e in events) / total, 2) if total else 0.0,
        language_distribution=dict(Counter(e.language for e in events)),
        tier_distribution=dict(Counter(e.data_tier_used for e in events)),
        event_type_distribution=dict(Counter(e.event_type for e in events)),
        top_cities=[CityCount(city=c, count=n)
                    for c, n in Counter(e.city for e in events if e.city).most_common(top_n)],
        top_attractions=[
            AttractionCount(
                attraction_id=aid,
                name_en=names[aid].name_en if aid in names else None,
                name_ar=names[aid].name_ar if aid in names else None,
                count=attraction_counts[aid],
            )
            for aid in top_ids
        ],
    )
