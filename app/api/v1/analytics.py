from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from app.db.database import get_session
from app.schemas import AnalyticsSummaryResponse
from app.services import analytics_service

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
def analytics_summary(
    event_type: Optional[str] = Query(None, description="e.g. chat_qa, attraction_lookup"),
    top_n: int = Query(5, ge=1, le=50),
    session: Session = Depends(get_session),
) -> AnalyticsSummaryResponse:
    return analytics_service.summarize(session, event_type=event_type, top_n=top_n)
