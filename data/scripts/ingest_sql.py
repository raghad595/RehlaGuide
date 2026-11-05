"""Load the raw CSVs into SQLite (data/rehlaguide.db), idempotently.

Run:  PYTHONPATH=. python data/scripts/ingest_sql.py
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dateutil import parser as date_parser
from sqlmodel import Session, delete, func, select

from app.db.database import engine, init_db
from app.models import Attraction, OpeningHour, TicketPrice

RAW_CSV_DIR = Path(__file__).resolve().parents[1] / "raw_csv"

WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DAY_ALIASES = {
    "mon": "Mon", "monday": "Mon",
    "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
    "wed": "Wed", "wednesday": "Wed",
    "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
    "fri": "Fri", "friday": "Fri",
    "sat": "Sat", "saturday": "Sat",
    "sun": "Sun", "sunday": "Sun",
}
_NULL_TOKENS = {"", "none", "n/a", "na", "null", "nan"}


# --------------------------------------------------------------------------- helpers
def clean(value) -> Optional[str]:
    """Strip a CSV cell; map empty / 'None' / 'N/A' to None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULL_TOKENS else text


def parse_date(value) -> Optional[date]:
    text = clean(value)
    if not text:
        return None
    try:
        return date_parser.parse(text).date()
    except (ValueError, OverflowError):
        return None


def _day(token: str) -> Optional[str]:
    return _DAY_ALIASES.get(token.strip().strip(".").lower())


def _day_range(start: str, end: str) -> List[str]:
    """Inclusive cyclic range, e.g. Sun-Thu -> Sun, Mon, Tue, Wed, Thu."""
    i, j = WEEK.index(start), WEEK.index(end)
    out = [WEEK[i]]
    while i != j:
        i = (i + 1) % 7
        out.append(WEEK[i])
    return out


def _parse_day_list(text: str) -> set[str]:
    days: set[str] = set()
    for token in re.split(r",|\band\b|&|/", text):
        token = token.strip()
        if not token:
            continue
        if "-" in token or "–" in token:
            start, end = [_day(p) for p in re.split(r"[-–]", token, maxsplit=1)]
            if start and end:
                days.update(_day_range(start, end))
            continue
        day = _day(token)
        if day:
            days.add(day)
    return days


def parse_days(value) -> List[str]:
    """'Every day' / 'Daily except Friday and Sunday' / 'Sunday-Thursday' -> ['Mon', ...]."""
    text = (clean(value) or "").strip()
    low = text.lower()
    if not text or low in {"not specified", "not applicable", "closed"}:
        return []
    m = re.match(r"^(daily|every\s*day|all\s*days?|7\s*days)\b(?:\s*except\s+(.*))?$", low)
    if m:
        excluded = _parse_day_list(m.group(2) or "")
        return [d for d in WEEK if d not in excluded]
    days = _parse_day_list(text)
    return [d for d in WEEK if d in days]


_TIME_WITH_COLON_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp]\.?\s*[Mm]\.?)?")


def is_24_hours(value) -> bool:
    return bool(re.search(r"24\s*(hours|hrs|h)\b", (clean(value) or "").lower()))


def parse_time(value, *, role: str = "open") -> Optional[str]:
    """'9:00 AM' -> '09:00', '5:00 PM' -> '17:00'. Text such as 'Closed' -> None.

    'Open 24 hours' -> '00:00' (open) / '23:59' (close). Embedded times such as
    'Sunrise (approximately 08:00 AM)' are extracted. Never raises.
    """
    text = clean(value)
    if not text:
        return None
    if is_24_hours(text):
        return "00:00" if role == "open" else "23:59"
    m = _TIME_WITH_COLON_RE.search(text)
    if not m:
        # bare "9 AM" style
        m2 = re.search(r"\b(\d{1,2})\s*([AaPp]\.?\s*[Mm]\.?)", text)
        if not m2:
            return None
        hour, minute, meridiem = int(m2.group(1)), 0, m2.group(2)
    else:
        hour, minute, meridiem = int(m.group(1)), int(m.group(2)), m.group(3)
    if meridiem:
        is_pm = meridiem.lower().startswith("p")
        if hour == 12:
            hour = 12 if is_pm else 0
        elif is_pm:
            hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


_EGP_PATTERNS = [
    re.compile(r"(?:EGP|E£|LE)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:EGP|E£|LE|pounds?)\b", re.IGNORECASE),
]


def parse_price(value) -> Optional[float]:
    """Primary EGP amount as float. 'Free'/'0'/'Included' -> 0.0; no EGP figure -> None."""
    text = clean(value)
    if not text:
        return None
    low = text.lower()
    if low.startswith("free") or low.startswith("included") or re.fullmatch(r"0+(\.0+)?", low):
        return 0.0
    for pattern in _EGP_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
    # plain number only, e.g. "250"
    if re.fullmatch(r"[\d,]+(?:\.\d+)?", text):
        return float(text.replace(",", ""))
    return None


# --------------------------------------------------------------------------- loaders
def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_CSV_DIR / name, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build_attractions(df: pd.DataFrame) -> List[Attraction]:
    return [
        Attraction(
            attraction_id=r["attraction_id"].strip(),
            name_en=r["name_en"].strip(),
            name_ar=r["name_ar"].strip(),
            city=r["city"].strip(),
            category=r["category"].strip(),
            visit_duration_minutes=int(r["visit_duration_minutes"]),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            official_url=clean(r["official_url"]),
            is_open=True,
            closure_reason=None,
        )
        for _, r in df.iterrows()
    ]


def build_opening_hours(df: pd.DataFrame) -> List[OpeningHour]:
    rows: List[OpeningHour] = []
    seen_primary: set[str] = set()
    for _, r in df.iterrows():
        aid = r["attraction_id"].strip()
        open_raw = r["Opening_Hour"].strip()
        close_raw = r["Closing_Hour"].strip()
        closed = open_raw.lower() == "closed"
        primary = not closed and aid not in seen_primary
        if primary:
            seen_primary.add(aid)
        rows.append(
            OpeningHour(
                attraction_id=aid,
                area_or_session=clean(r["Area/Session"]) or "Main Site",
                opening_days_raw=r["Opening_Days"].strip(),
                opening_hour_raw=open_raw,
                closing_hour_raw=close_raw,
                last_entry_hour_raw=clean(r["Last_Entry_Hour"]),
                open_days_list=[] if closed else parse_days(r["Opening_Days"]),
                open_time_24h=None if closed else parse_time(open_raw, role="open"),
                close_time_24h=None if closed else parse_time(close_raw, role="close"),
                last_entry_24h=None if closed else parse_time(r["Last_Entry_Hour"], role="close"),
                is_primary_session=primary,
                notes=clean(r["Notes"]),
                source_url=clean(r["Source_URL"]),
                last_verified_at=parse_date(r["Last_Verified_At"]),
            )
        )
    return rows


def build_ticket_prices(df: pd.DataFrame) -> List[TicketPrice]:
    rows: List[TicketPrice] = []
    for _, r in df.iterrows():
        locals_raw = r["Price_for_Locals"].strip()
        foreign_raw = r["Price_for_Foreigners"].strip()
        child_raw = clean(r["Children_Ticket_Price"])
        rows.append(
            TicketPrice(
                attraction_id=r["attraction_id"].strip(),
                ticket_type=r["Ticket_Type"].strip(),
                price_for_locals_raw=locals_raw,
                price_for_foreigners_raw=foreign_raw,
                children_ticket_price_raw=child_raw,
                free_entry_cases=clean(r["Free_Entry_Cases"]),
                price_locals_egp=parse_price(locals_raw),
                price_foreigners_egp=parse_price(foreign_raw),
                price_children_egp=parse_price(child_raw),
                source_url=clean(r["Source_URL"]),
                booking_url=clean(r["Booking_URL"]),
                last_verified_at=parse_date(r["Last_Verified_At"]),
            )
        )
    return rows


def main() -> None:
    init_db()
    attractions = build_attractions(_read("attractions.csv"))
    hours = build_opening_hours(_read("opening_hours.csv"))
    prices = build_ticket_prices(_read("ticket_prices.csv"))

    known = {a.attraction_id for a in attractions}
    orphans = sorted({r.attraction_id for r in [*hours, *prices]} - known)
    if orphans:
        raise SystemExit(f"ERROR: child rows reference unknown attraction_id(s): {orphans}")

    with Session(engine, expire_on_commit=False) as session:
        # Idempotent full refresh: children first (FK order), then parents.
        session.exec(delete(TicketPrice))
        session.exec(delete(OpeningHour))
        session.exec(delete(Attraction))
        session.add_all(attractions)
        session.flush()
        session.add_all(hours)
        session.add_all(prices)
        session.commit()

        counts = {
            m.__tablename__: session.exec(select(func.count()).select_from(m)).one()
            for m in (Attraction, OpeningHour, TicketPrice)
        }

    print("=== SQLite ingestion summary (data/rehlaguide.db) ===")
    for table, n in counts.items():
        print(f"  {table:<15} {n:>4} rows")
    print("--- opening_hours normalization ---")
    print(f"  open_time_24h parsed : {sum(h.open_time_24h is not None for h in hours)}/{len(hours)}")
    print(f"  close_time_24h parsed: {sum(h.close_time_24h is not None for h in hours)}/{len(hours)}")
    print(f"  last_entry_24h parsed: {sum(h.last_entry_24h is not None for h in hours)}/{len(hours)}")
    print(f"  empty open_days_list : {sum(not h.open_days_list for h in hours)} "
          f"(closed or 'Not specified' rows)")
    print("--- ticket_prices normalization ---")
    for field in ("price_locals_egp", "price_foreigners_egp", "price_children_egp"):
        vals = [getattr(p, field) for p in prices]
        print(f"  {field:<21}: {sum(v is not None for v in vals)}/{len(vals)} numeric")


if __name__ == "__main__":
    main()
