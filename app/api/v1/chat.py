import time

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db.database import get_session
from app.schemas import ChatRequest, ChatResponse
from app.services import analytics_service, llm_service, rag_service

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    started = time.perf_counter()
    language = req.preferred_language or rag_service.detect_language(req.message)
    key = rag_service.cache_key(req.message, language)

    cached = rag_service.cache_get(key)
    if cached is not None:
        response: ChatResponse = cached.model_copy(update={"session_id": req.session_id, "cache_hit": True})
    else:
        result = rag_service.retrieve(req.message, language, session)
        answer, provider = llm_service.generate_answer(result)
        response = ChatResponse(
            session_id=req.session_id,
            detected_language=language,
            intent="qa_factual",
            assistant_message=answer,
            referenced_attractions=result.attractions,
            data_tier_used=result.tier,
            supplementary_sources=result.unverified_sources(),
            llm_provider=provider,
            cache_hit=False,
            latency_ms=0,
        )
        rag_service.cache_put(key, response, result.attraction_ids)

    response.latency_ms = int((time.perf_counter() - started) * 1000)
    analytics_service.log_event(
        session_id=req.session_id,
        event_type="chat_qa",
        language=language,
        data_tier_used=response.data_tier_used,
        city=_city_of(response),
        attraction_ids=[a.attraction_id for a in response.referenced_attractions],
        cache_hit=response.cache_hit,
        latency_ms=response.latency_ms,
    )
    return response


def _city_of(r: ChatResponse):
    if r.referenced_attractions:
        return r.referenced_attractions[0].city
    if r.supplementary_sources:
        return r.supplementary_sources[0].city
    return None
