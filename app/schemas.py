from datetime import date
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- 1. ATTRACTION DATA CONTRACTS (/api/v1/attractions) ---

class OpeningHourSchema(BaseModel):
    area_or_session: str
    opening_days_raw: str
    opening_hour_raw: str
    closing_hour_raw: str
    last_entry_hour_raw: Optional[str] = None
    open_days_list: List[str]
    open_time_24h: Optional[str] = None
    close_time_24h: Optional[str] = None
    last_entry_24h: Optional[str] = None
    is_primary_session: bool = True
    notes: Optional[str] = None
    source_url: Optional[str] = None
    last_verified_at: Optional[date] = None

class TicketPriceSchema(BaseModel):
    ticket_type: str
    price_for_locals_raw: str
    price_for_foreigners_raw: str
    children_ticket_price_raw: Optional[str] = None
    free_entry_cases: Optional[str] = None
    price_locals_egp: Optional[float] = None
    price_foreigners_egp: Optional[float] = None
    price_children_egp: Optional[float] = None
    source_url: Optional[str] = None
    booking_url: Optional[str] = None
    last_verified_at: Optional[date] = None

class AttractionSummary(BaseModel):
    attraction_id: str
    name_en: str
    name_ar: str
    city: str
    category: str
    visit_duration_minutes: int
    latitude: float
    longitude: float
    is_open: bool = True
    closure_reason: Optional[str] = None

class AttractionStatusUpdate(BaseModel):
    is_open: bool
    closure_reason: Optional[str] = None

class AttractionDetailResponse(BaseModel):
    attraction_id: str
    name_en: str
    name_ar: str
    city: str
    category: str
    visit_duration_minutes: int
    latitude: float
    longitude: float
    official_url: Optional[str] = None
    is_open: bool = True
    closure_reason: Optional[str] = None
    opening_hours: List[OpeningHourSchema]
    ticket_prices: List[TicketPriceSchema]


# --- 2. ITINERARY & SCHEDULING CONTRACTS (/api/v1/itineraries) ---

class ScheduledStop(BaseModel):
    stop_order: int
    attraction_id: str
    name_en: str
    name_ar: str
    category: str
    latitude: float
    longitude: float
    start_time: str = Field(description="24h format HH:MM, e.g., '09:00'")
    end_time: str = Field(description="24h format HH:MM, e.g., '11:30'")
    visit_duration_minutes: int
    transit_from_prev_minutes: int = 0
    applicable_opening_hours: str
    applicable_ticket_price: str
    selection_reason: str = Field(description="Grounded explanation in user's language")
    visitor_tip: Optional[str] = None
    source_url: Optional[str] = None
    last_verified_at: Optional[date] = None

class DayItinerary(BaseModel):
    day_number: int
    day_title: str
    city_or_zone: str
    total_active_minutes: int
    stops: List[ScheduledStop]
    unscheduled_notes: Optional[str] = None

class ItineraryResponse(BaseModel):
    itinerary_id: str
    destination_city: str
    language: Literal["en", "ar"]
    pace: Literal["relaxed", "moderate", "packed"]
    visitor_group: Literal["locals", "foreigners"] = "foreigners"
    assumptions_made: List[str]
    days: List[DayItinerary]
    overflow_suggestions: List[str] = []


# --- 3. CONVERSATIONAL & REPLANNING CONTRACTS (/api/v1/chat) ---

class ChatRequest(BaseModel):
    session_id: str
    message: str
    preferred_language: Optional[Literal["en", "ar"]] = None

class ChatResponse(BaseModel):
    session_id: str
    detected_language: Literal["en", "ar"]
    intent: Literal["qa_factual", "plan_itinerary", "modify_itinerary", "clarification"]
    assistant_message: str
    itinerary: Optional[ItineraryResponse] = None
    referenced_attractions: List[AttractionDetailResponse] = []
    cache_hit: bool = False
    latency_ms: int