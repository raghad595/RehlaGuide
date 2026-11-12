from datetime import date, datetime, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship, Column, JSON

class Attraction(SQLModel, table=True):
    __tablename__ = "attractions"

    attraction_id: str = Field(primary_key=True, index=True)  # From CSV Table 1
    name_en: str = Field(index=True)
    name_ar: str = Field(index=True)
    city: str = Field(index=True)                             # Used for region filtering
    category: str = Field(index=True)                         # e.g., religious, nature, ancient_history
    visit_duration_minutes: int
    latitude: float
    longitude: float
    official_url: Optional[str] = None

    # Added operational state columns for Pillar 1 Dynamic Replanning
    is_open: bool = Field(default=True, index=True)
    closure_reason: Optional[str] = None

    # 1-to-Many Relationships
    opening_hours: List["OpeningHour"] = Relationship(back_populates="attraction")
    ticket_prices: List["TicketPrice"] = Relationship(back_populates="attraction")


class OpeningHour(SQLModel, table=True):
    __tablename__ = "opening_hours"

    id: Optional[int] = Field(default=None, primary_key=True) # From CSV Table 2
    attraction_id: str = Field(foreign_key="attractions.attraction_id", index=True)
    area_or_session: str = Field(default="Main Site")         # From 'Area/Session'
    
    # Raw human-readable strings from CSV (for grounded LLM display)
    opening_days_raw: str                                     # From 'Opening_Days'
    opening_hour_raw: str                                     # From 'Opening_Hour'
    closing_hour_raw: str                                     # From 'Closing_Hour'
    last_entry_hour_raw: Optional[str] = None                 # From 'Last_Entry_Hour'
    
    # Normalized machine-readable fields (for Phase 3 Deterministic Scheduler)
    open_days_list: List[str] = Field(default=[], sa_column=Column(JSON)) # e.g. ["Mon","Tue","Wed","Thu","Sat"]
    open_time_24h: Optional[str] = None                       # e.g. "09:00"
    close_time_24h: Optional[str] = None                      # e.g. "11:30"
    last_entry_24h: Optional[str] = None                      # e.g. "11:00"
    is_primary_session: bool = Field(default=True)            # Marks default slot for scheduler

    notes: Optional[str] = None                               # From 'Notes'
    source_url: Optional[str] = None                          # From 'Source_URL'
    last_verified_at: Optional[date] = None                   # From 'Last_Verified_At'

    attraction: Optional[Attraction] = Relationship(back_populates="opening_hours")


class TicketPrice(SQLModel, table=True):
    __tablename__ = "ticket_prices"

    id: Optional[int] = Field(default=None, primary_key=True) # From CSV Table 3
    attraction_id: str = Field(foreign_key="attractions.attraction_id", index=True)
    ticket_type: str                                          # From 'Ticket_Type'
    
    # Raw strings from CSV (preserves exact notes like "500 EGP / 250 Student")
    price_for_locals_raw: str                                 # From 'Price_for_Locals'
    price_for_foreigners_raw: str                             # From 'Price_for_Foreigners'
    children_ticket_price_raw: Optional[str] = None           # From 'Children_Ticket_Price'
    free_entry_cases: Optional[str] = None                    # From 'Free_Entry_Cases'

    # Normalized numeric EGP floats (for budget filtering & cost estimation)
    price_locals_egp: Optional[float] = None
    price_foreigners_egp: Optional[float] = None
    price_children_egp: Optional[float] = None

    source_url: Optional[str] = None                          # From 'Source_URL'
    booking_url: Optional[str] = None                         # From 'Booking_URL'
    last_verified_at: Optional[date] = None                   # From 'Last_Verified_At'

    attraction: Optional[Attraction] = Relationship(back_populates="ticket_prices")

def _utcnow() -> datetime:
    # Timezone-aware UTC. (datetime.utcnow() is deprecated, and this SQLModel version
    # rejects naive datetimes on insert.)
    return datetime.now(timezone.utc)


class AnalyticsEvent(SQLModel, table=True):
    __tablename__ = "analytics_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utcnow, index=True)
    session_id: str = Field(index=True)
    event_type: str = Field(index=True)      # chat_qa | attraction_lookup | itinerary_plan | itinerary_replan
    language: str = Field(index=True)        # en | ar
    data_tier_used: str = Field(default="verified")  # verified | unverified_fallback | none
    city: Optional[str] = Field(default=None, index=True)
    attraction_ids_json: List[str] = Field(default=[], sa_column=Column(JSON))
    cache_hit: bool = Field(default=False)
    latency_ms: int = Field(default=0)
