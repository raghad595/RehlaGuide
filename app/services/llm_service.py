"""Grounded answer generation: Gemini, Groq, or a deterministic template fallback.

Provider choice (LLM_PROVIDER in .env): "auto" (default) uses Gemini if GEMINI_API_KEY is
set, else Groq if GROQ_API_KEY is set, else the deterministic fallback. "none" forces the
fallback (used by the test suite so it never calls the network).

Guardrails applied to every LLM answer (falls back to the deterministic answer on failure):
  - an Arabic question must get an answer containing Arabic script;
  - an unverified (Wikidata) answer must not contain clock times or money amounts, and
    always ends with the "no verified hours/prices" notice;
  - tier "none" never calls the LLM (nothing to ground on).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import httpx

from app.core.config import get_settings
from app.schemas import AttractionDetailResponse, OpeningHourSchema, TicketPriceSchema
from app.services.query_parsing import ARABIC_RE
from app.services.rag_service import OPERATIONAL_NUMBERS_RE, RetrievalResult, RetrievedChunk

log = logging.getLogger("rehlaguide.llm")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are RehlaGuide, a bilingual (English/Arabic) assistant for tourism in Egypt.

LANGUAGE
- Reply ONLY in the language given in <user_language> ("en" = English, "ar" = Modern Standard Arabic).
  Attraction names may be given in both languages.

GROUNDING
- Use ONLY the information inside the context blocks. Never use outside knowledge for facts,
  numbers, opening hours, prices, dates or closures. If the context does not answer part of
  the question, say so plainly.
- Be concise: short paragraphs or bullet points, no more than ~200 words unless listing several sites.

WHEN <data_tier_used> IS "verified"
- Take opening hours, ticket prices and open/closed status ONLY from <verified_operational_data>.
  Quote prices with their currency exactly as given; do not convert currencies.
- If an attraction's Status is CLOSED, say so first and give the closure reason.
- Use <verified_narrative_chunks> for history, highlights, visitor tips and accessibility.
- End with the source URL(s) and the last_verified_at date for the hours/prices you cited.

WHEN <data_tier_used> IS "unverified_fallback"
- Share only the historical / cultural / architectural information in
  <unverified_supplementary_chunks>.
- State clearly that this information comes from RehlaGuide's supplementary, unverified
  catalog (Wikidata) and that the site is not one of RehlaGuide's verified attractions.
- NEVER state, estimate or guess opening hours, visiting times, ticket prices or fees for
  these sites, even if asked directly. Explain that verified opening hours and ticket prices
  are not available for this site and that only historical context is available for
  unverified sites; suggest checking official sources before visiting.

WHEN <data_tier_used> IS "none"
- Politely say the attraction is not in the RehlaGuide knowledge base.
"""

DAYS_AR = {"Mon": "الاثنين", "Tue": "الثلاثاء", "Wed": "الأربعاء", "Thu": "الخميس",
           "Fri": "الجمعة", "Sat": "السبت", "Sun": "الأحد"}

UNVERIFIED_NOTICE = {
    "en": ("Note: verified opening hours and ticket prices are not available for this site. "
           "Only historical context is available for unverified sites, so please check official "
           "sources before visiting."),
    "ar": ("ملاحظة: مواعيد الزيارة وأسعار التذاكر المُتحقق منها غير متوفرة لهذا الموقع، "
           "وتتوفر المعلومات التاريخية فقط للمواقع غير المُتحقق منها، لذا يُرجى مراجعة المصادر "
           "الرسمية قبل الزيارة."),
}
# Phrases proving the notice is present in an LLM answer.
_NOTICE_MARKERS = {
    "en": re.compile(r"not available|unavailable|no verified|not verified", re.IGNORECASE),
    "ar": re.compile(r"غير متوفر|غير متاح|لا تتوفر|لا يتوفر"),
}


# --------------------------------------------------------------------------- public API
def resolve_provider() -> str:
    s = get_settings()
    if s.llm_provider == "none":
        return "deterministic"
    if s.llm_provider == "gemini":
        return "gemini" if s.gemini_api_key else "deterministic"
    if s.llm_provider == "groq":
        return "groq" if s.groq_api_key else "deterministic"
    if s.gemini_api_key:
        return "gemini"
    if s.groq_api_key:
        return "groq"
    return "deterministic"


def build_user_prompt(result: RetrievalResult) -> str:
    return (
        f"<user_language>{result.language}</user_language>\n"
        f"<data_tier_used>{result.tier}</data_tier_used>\n\n"
        f"{result.context_blocks()}\n\n"
        f"<question>\n{result.query}\n</question>"
    )


def provider_chain() -> List[str]:
    """Providers to try in order. In "auto" mode Groq backs up Gemini (e.g. on a 503 overload)."""
    primary = resolve_provider()
    if primary == "deterministic":
        return []
    chain = [primary]
    s = get_settings()
    if s.llm_provider == "auto" and primary == "gemini" and s.groq_api_key:
        chain.append("groq")
    return chain


def generate_answer(result: RetrievalResult) -> Tuple[str, str]:
    """Returns (answer_text, provider_used)."""
    fallback = deterministic_answer(result)
    if result.tier == "none":
        return fallback, "deterministic"
    prompt = build_user_prompt(result)
    for provider in provider_chain():
        try:
            text = _call_provider(provider, SYSTEM_PROMPT, prompt).strip()
        except Exception as exc:  # network, quota, overload, malformed response...
            log.warning("LLM provider %s failed (%s: %s)", provider, type(exc).__name__, exc)
            continue
        checked = apply_guardrails(text, result)
        if checked is not None:
            return checked, provider
        log.warning("LLM answer from %s failed guardrails", provider)
    return fallback, "deterministic"


def apply_guardrails(text: str, result: RetrievalResult) -> Optional[str]:
    """Validated/patched answer, or None if it must be replaced by the deterministic one."""
    if not text:
        return None
    if result.language == "ar" and not ARABIC_RE.search(text):
        return None
    if result.tier == "unverified_fallback":
        if OPERATIONAL_NUMBERS_RE.search(text):
            return None  # an hour or price leaked into an unverified answer
        if not _NOTICE_MARKERS[result.language].search(text):
            text = f"{text}\n\n{UNVERIFIED_NOTICE[result.language]}"
    return text


# --------------------------------------------------------------------------- providers
def _call_provider(provider: str, system: str, user: str) -> str:
    s = get_settings()
    with httpx.Client(timeout=s.llm_timeout_seconds) as client:
        if provider == "gemini":
            r = client.post(
                GEMINI_URL.format(model=s.gemini_model),
                headers={"x-goog-api-key": s.gemini_api_key or ""},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": s.llm_temperature},
                },
            )
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        if provider == "groq":
            r = client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {s.groq_api_key}"},
                json={
                    "model": s.groq_model,
                    "temperature": s.llm_temperature,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    raise ValueError(f"unknown provider {provider!r}")


# --------------------------------------------------------------------------- deterministic answers
def deterministic_answer(result: RetrievalResult) -> str:
    if result.tier == "verified" and result.attractions:
        return _verified_answer(result)
    if result.tier == "unverified_fallback" and result.unverified_chunks:
        return _unverified_answer(result)
    return _none_answer(result.language)


def _excerpt(chunk: Optional[RetrievedChunk], max_chars: int = 380) -> str:
    if chunk is None:
        return ""
    text = chunk.text
    if text.startswith(("Attraction:", "المعلم:", "Monument:", "الموقع:")) and "\n\n" in text:
        text = text.split("\n\n", 1)[1]  # drop the injected context header line
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*\*\*[^*]+\*\*\s*:?\s*", ln)]
    flat = re.sub(r"\*\*|^\s*[*\-]\s+", "", "\n".join(lines), flags=re.MULTILINE)
    flat = re.sub(r"\s+", " ", flat).strip()
    if len(flat) <= max_chars:
        return flat
    cut = flat[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("؟ "), cut.rfind("! "))
    return (cut[: end + 1] if end > 80 else cut.rsplit(" ", 1)[0] + "…").strip()


def _days(h: OpeningHourSchema, lang: str) -> str:
    if lang == "en":
        return h.opening_days_raw
    if len(h.open_days_list) == 7:
        return "يوميًا"
    if not h.open_days_list:
        return "الأيام غير محددة"
    return "، ".join(DAYS_AR[d] for d in h.open_days_list)


def _hours_line(h: OpeningHourSchema, lang: str) -> str:
    closed = h.opening_hour_raw.strip().lower() == "closed"
    if lang == "en":
        if closed:
            return f"{h.area_or_session}: closed on {h.opening_days_raw}"
        s = f"{h.area_or_session}: {h.opening_days_raw}, {h.opening_hour_raw} – {h.closing_hour_raw}"
        if h.last_entry_24h:
            s += f" (last entry {h.last_entry_hour_raw})"
        return s
    if closed:
        return f"{h.area_or_session}: مغلق ({_days_from_raw(h)})"
    if h.open_time_24h == "00:00" and h.close_time_24h == "23:59":
        span = "مفتوح على مدار 24 ساعة"
    elif h.open_time_24h and h.close_time_24h:
        span = f"من {h.open_time_24h} إلى {h.close_time_24h}"
    else:
        span = f"{h.opening_hour_raw} – {h.closing_hour_raw}"
    s = f"{h.area_or_session}: {_days(h, 'ar')}، {span}"
    if h.last_entry_24h:
        s += f" (آخر دخول {h.last_entry_24h})"
    return s


_DAY_NAMES_AR = {"monday": "الاثنين", "tuesday": "الثلاثاء", "wednesday": "الأربعاء", "thursday": "الخميس",
                 "friday": "الجمعة", "saturday": "السبت", "sunday": "الأحد", "every day": "يوميًا"}


def _days_from_raw(h: OpeningHourSchema) -> str:
    """Translate day names inside a raw string (closed rows have an empty open_days_list)."""
    s = h.opening_days_raw
    for en, ar in _DAY_NAMES_AR.items():
        s = re.sub(en, ar, s, flags=re.IGNORECASE)
    return s


def _price(value: Optional[float], raw: Optional[str], lang: str) -> str:
    if value is None:
        return raw or ("not specified" if lang == "en" else "غير محدد")
    if value == 0:
        return "free" if lang == "en" else "مجانًا"
    amount = f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
    return f"EGP {amount}" if lang == "en" else f"{amount} جنيه"


def _ticket_line(t: TicketPriceSchema, lang: str) -> str:
    if lang == "en":
        s = f"{t.ticket_type} — Egyptians: {t.price_for_locals_raw}; foreigners: {t.price_for_foreigners_raw}"
        if t.children_ticket_price_raw:
            s += f"; children: {t.children_ticket_price_raw}"
        return s
    s = (f"{t.ticket_type} — للمصريين: {_price(t.price_locals_egp, t.price_for_locals_raw, 'ar')} | "
         f"للأجانب: {_price(t.price_foreigners_egp, t.price_for_foreigners_raw, 'ar')}")
    if t.children_ticket_price_raw:
        s += f" | للأطفال: {_price(t.price_children_egp, t.children_ticket_price_raw, 'ar')}"
    return s


def _attraction_card(a: AttractionDetailResponse, chunk: Optional[RetrievedChunk], lang: str) -> str:
    max_rows = 4
    hours = a.opening_hours[:max_rows]
    tickets = a.ticket_prices[:3]
    verified_dates = sorted({str(x.last_verified_at) for x in [*a.opening_hours, *a.ticket_prices] if x.last_verified_at})
    sources = list(dict.fromkeys(
        [*(h.source_url for h in hours if h.source_url), *(t.source_url for t in tickets if t.source_url)]
    ))[:2] or ([a.official_url] if a.official_url else [])
    about = _excerpt(chunk)
    if lang == "en":
        status = "Open" if a.is_open else f"CURRENTLY CLOSED — {a.closure_reason or 'reason not specified'}"
        out = [f"**{a.name_en}** ({a.city} · {a.category}) — Status: {status}"]
        out += [f"- Opening hours: {_hours_line(h, 'en')}" for h in hours]
        if len(a.opening_hours) > max_rows:
            out.append(f"- …plus {len(a.opening_hours) - max_rows} more sessions/areas")
        out += [f"- Tickets: {_ticket_line(t, 'en')}" for t in tickets]
        out.append(f"- Suggested visit: about {a.visit_duration_minutes} minutes")
        if about:
            out.append(f"- About: {about}")
        out.append(f"Source: {', '.join(sources) or 'n/a'}"
                   + (f" (last verified {verified_dates[-1]})" if verified_dates else ""))
        return "\n".join(out)
    status = "مفتوح" if a.is_open else f"مغلق حاليًا — {a.closure_reason or 'السبب غير محدد'}"
    out = [f"**{a.name_ar}** ({a.name_en}) — الحالة: {status}"]
    out += [f"- مواعيد الزيارة: {_hours_line(h, 'ar')}" for h in hours]
    if len(a.opening_hours) > max_rows:
        out.append(f"- …بالإضافة إلى {len(a.opening_hours) - max_rows} فترات/أقسام أخرى")
    out += [f"- التذاكر: {_ticket_line(t, 'ar')}" for t in tickets]
    out.append(f"- مدة الزيارة المقترحة: حوالي {a.visit_duration_minutes} دقيقة")
    if about:
        out.append(f"- نبذة: {about}")
    out.append(f"المصدر: {'، '.join(sources) or 'غير متوفر'}"
               + (f" (آخر تحقق: {verified_dates[-1]})" if verified_dates else ""))
    return "\n".join(out)


def _verified_answer(result: RetrievalResult) -> str:
    lang = result.language
    best_chunk = {}
    for c in result.verified_chunks:
        best_chunk.setdefault(c.payload.get("attraction_id"), c)
    intro = ("Here is what RehlaGuide's verified data says:" if lang == "en"
             else "إليك المعلومات من بيانات RehlaGuide المُتحقق منها:")
    cards = [_attraction_card(a, best_chunk.get(a.attraction_id), lang) for a in result.attractions]
    return intro + "\n\n" + "\n\n".join(cards)


def _unverified_answer(result: RetrievalResult) -> str:
    lang = result.language
    by_site: dict = {}
    for c in result.unverified_chunks:
        key = c.payload.get("wikidata_qid") or c.payload.get("name_en")
        by_site.setdefault(key, c)
    blocks: List[str] = []
    for c in by_site.values():
        p = c.payload
        name = (p.get("name_ar") if lang == "ar" else p.get("name_en")) or p.get("name_en") or p.get("name_ar") or "?"
        lines = [f"**{name}**", _excerpt(c, 600)]
        loc = p.get("city") or ""
        if p.get("latitude") is not None and p.get("longitude") is not None:
            loc = f"{loc} ({p['latitude']}, {p['longitude']})".strip()
        if loc:
            lines.append(("Location: " if lang == "en" else "الموقع: ") + loc)
        if p.get("source_url"):
            lines.append(("Source: " if lang == "en" else "المصدر: ") + p["source_url"])
        blocks.append("\n".join(x for x in lines if x))
    intro = (
        "This site is not one of RehlaGuide's verified attractions. The historical context below comes "
        "from RehlaGuide's supplementary, unverified catalog (Wikidata):"
        if lang == "en" else
        "هذا الموقع ليس ضمن المعالم المُتحقق منها في RehlaGuide. المعلومات التاريخية التالية مأخوذة من "
        "الكتالوج التكميلي غير المُتحقق منه (ويكي بيانات - Wikidata):"
    )
    return intro + "\n\n" + "\n\n".join(blocks) + "\n\n" + UNVERIFIED_NOTICE[lang]


def _none_answer(lang: str) -> str:
    if lang == "en":
        return ("Sorry, I couldn't find that attraction in the RehlaGuide knowledge base. RehlaGuide "
                "currently covers verified attractions in Cairo, Giza, Alexandria, Luxor, Aswan, South "
                "Sinai, Fayoum, Siwa (Matrouh) and the Red Sea. Try asking about one of them, for example "
                "the Pyramids of Giza, Karnak Temple or Abu Simbel.")
    return ("عذرًا، لم أجد هذا المعلم في قاعدة معارف RehlaGuide. تغطي RehlaGuide حاليًا معالم مُتحققًا منها "
            "في القاهرة والجيزة والإسكندرية والأقصر وأسوان وجنوب سيناء والفيوم وسيوة (مطروح) والبحر الأحمر. "
            "جرّب السؤال عن أحدها، مثل أهرامات الجيزة أو معبد الكرنك أو معابد أبو سمبل.")
