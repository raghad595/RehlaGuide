"""Deterministic, bilingual (EN/AR) query understanding.

- detect_language: Arabic-script vs Latin-script letter counts.
- normalize: one normalized string used for ALL matching (so match spans are comparable).
- extract_filters: city / category / max visit duration.
- AliasMatcher: finds verified attractions named in the query (names, parts of names,
  common short forms, attraction_id), longest-match-wins so that
  "Grand Egyptian Museum" does not also match "Egyptian Museum".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
_AR_LETTER = r"ء-ي"
_DIACRITICS_RE = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alef, tatweel
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def detect_language(text: str) -> Literal["en", "ar"]:
    """'ar' when Arabic letters are at least as frequent as Latin letters."""
    ar = len(ARABIC_RE.findall(text or ""))
    lat = len(LATIN_RE.findall(text or ""))
    return "ar" if ar > 0 and ar >= lat else "en"


def normalize(text: str) -> str:
    t = (text or "").translate(_AR_DIGITS).lower()
    t = _DIACRITICS_RE.sub("", t)
    t = re.sub("[أإآٱ]", "ا", t).replace("ة", "ه").replace("ى", "ي")
    t = t.replace("’", "'").replace("`", "'").replace("'", "")
    t = re.sub(r"[-–—_/|]", " ", t)
    t = re.sub(r"(?<=\d)\.(?=\d)", "\x00", t)  # protect decimal points ("1.5 hours")
    t = re.sub(r"[^\w\s\x00]", " ", t)          # drop punctuation (keeps letters/digits of all scripts)
    t = t.replace("\x00", ".")
    return re.sub(r"\s+", " ", t).strip()


def _is_arabic(s: str) -> bool:
    return bool(ARABIC_RE.search(s))


def alias_pattern(alias: str) -> re.Pattern:
    """Regex for a *normalized* alias.

    English: word boundaries. Arabic: no Arabic letter before/after, and allow the
    attached prefixes و ف ب ك ل + optional ال (e.g. "الأقصر" matches "بالأقصر", "للأقصر").
    """
    alias = normalize(alias)
    if not _is_arabic(alias):
        return re.compile(rf"\b{re.escape(alias)}\b")
    first, *rest = alias.split(" ")
    has_article = first.startswith("ال") and len(first) > 3
    core = first[2:] if has_article else first
    body = re.escape(" ".join([core, *rest]))
    # If the alias carries the article, require it ("الأقصر"=Luxor but bare "أقصر"="shortest").
    prefix = r"(?:لل|[وفبكل]?ال)" if has_article else r"(?:لل|[وفبكل]?(?:ال)?)"
    return re.compile(rf"(?<![{_AR_LETTER}]){prefix}{body}(?![{_AR_LETTER}])")


# --------------------------------------------------------------------------- filters
CITY_ALIASES: Dict[str, List[str]] = {
    # label -> aliases. A label selects every DB city that contains it (Giza -> "Giza / Memphis" too).
    "Giza": ["giza", "الجيزة", "جيزة"],
    "Memphis": ["memphis", "منف"],
    "Cairo": ["cairo", "القاهرة"],
    "Alexandria": ["alexandria", "alex", "الإسكندرية", "اسكندرية"],
    "Luxor": ["luxor", "الأقصر"],
    "Aswan": ["aswan", "أسوان"],
    "Sinai": ["sinai", "south sinai", "sharm", "sharm el sheikh", "dahab", "سيناء", "جنوب سيناء", "شرم الشيخ", "شرم"],
    "Fayoum": ["fayoum", "faiyum", "fayyum", "الفيوم"],
    "Matrouh": ["matrouh", "marsa matrouh", "مطروح", "مرسى مطروح"],
    "Red Sea": ["red sea", "hurghada", "البحر الأحمر", "الغردقة"],
}

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    # stem (matched case-insensitively inside DB category names) -> query keywords
    "museum": ["museum", "museums", "متحف", "متاحف"],
    "archaeolog": ["temple", "temples", "tomb", "tombs", "archaeological", "ruins", "ancient site",
                   "ancient sites", "معبد", "معابد", "مقبرة", "مقابر", "أثري", "أثرية", "آثار"],
    "fort": ["citadel", "citadels", "fort", "fortress", "castle", "قلعة", "قلاع"],
    "national park": ["national park", "national parks", "nature reserve", "protected area", "محمية", "محميات"],
    "religious": ["monastery", "church", "religious", "دير", "كنيسة", "ديني", "دينية"],
    "oasis": ["oasis", "oases", "واحة", "واحات"],
    "resort": ["resort", "resorts", "beach", "beaches", "منتجع", "شاطئ", "شواطئ"],
    "library": ["library", "libraries", "مكتبة", "مكتبات"],
    "landmark": ["landmark", "landmarks", "bridge", "كوبري", "جسر"],
}

_WORD_NUM = {"one": 1, "an": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_EN_LIMIT = r"(?:under|less than|at most|max(?:imum)?|within|up to|no more than|shorter than|<=?)"
_EN_DURATION_RE = re.compile(
    rf"{_EN_LIMIT}\s*(\d+(?:\.\d+)?|one|two|three|four|five|six|an|a)\s*(hours?|hrs?|h|minutes?|mins?)\b"
)
_AR_LIMIT = r"(?:اقل من|خلال|في حدود|حتي|لا تزيد عن|لا يزيد عن|لا تتجاوز|لا يتجاوز|بحد اقصي)"
_AR_DURATION_RE = re.compile(rf"{_AR_LIMIT}\s*(\d+(?:\.\d+)?)?\s*(ساعتين|ساعات|ساعه|دقيقه|دقائق|دقايق)")


@dataclass
class QueryFilters:
    city_label: Optional[str] = None
    cities: List[str] = field(default_factory=list)       # exact DB city values
    category_stems: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)   # exact DB category values
    max_duration_minutes: Optional[int] = None

    @property
    def any(self) -> bool:
        return bool(self.cities or self.categories or self.max_duration_minutes)


def _first_match(norm: str, aliases: Iterable[str]) -> Optional[re.Match]:
    best = None
    for a in aliases:
        m = alias_pattern(a).search(norm)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def extract_max_duration(norm: str) -> Optional[int]:
    m = _EN_DURATION_RE.search(norm)
    if m:
        raw, unit = m.group(1), m.group(2)
        value = float(_WORD_NUM.get(raw, raw) if not raw.replace(".", "").isdigit() else raw)
        return int(value * 60) if unit.startswith("h") else int(value)
    m = _AR_DURATION_RE.search(norm)
    if m:
        num, unit = m.group(1), m.group(2)
        if unit == "ساعتين":
            return 120
        value = float(num) if num else 1.0
        return int(value * 60) if unit.startswith("ساع") else int(value)
    return None


def extract_filters(text: str, db_cities: Sequence[str], db_categories: Sequence[str]) -> QueryFilters:
    norm = normalize(text)
    f = QueryFilters()

    # City: earliest mentioned city wins.
    found: List[Tuple[int, str]] = []
    for label, aliases in CITY_ALIASES.items():
        m = _first_match(norm, aliases)
        if m:
            found.append((m.start(), label))
    if found:
        f.city_label = sorted(found)[0][1]
        f.cities = [c for c in db_cities if f.city_label.lower() in c.lower()]

    for stem, keywords in CATEGORY_KEYWORDS.items():
        if _first_match(norm, keywords):
            f.category_stems.append(stem)
    f.categories = sorted({c for c in db_categories for s in f.category_stems if s in c.lower()})

    f.max_duration_minutes = extract_max_duration(norm)
    return f


# --------------------------------------------------------------------------- name matching
# Short forms people actually type. Full names / name parts / ids are added from SQLite.
# "generic" aliases are dropped whenever a specific alias also matched
# (e.g. "Dahshur pyramids" -> Dahshur only, not also the Giza pyramids).
MANUAL_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "GIZ001": {"specific": ["giza pyramids", "pyramids of giza", "great pyramid", "khufu", "أهرامات الجيزة", "الهرم الأكبر", "هرم خوفو"],
               "generic": ["pyramids", "الأهرامات", "الأهرام"]},
    "GIZ002": {"specific": ["sphinx", "great sphinx", "أبو الهول"]},
    "GIZ003": {"specific": ["grand egyptian museum", "gem", "المتحف المصري الكبير", "المتحف الكبير"]},
    "GIZ004": {"specific": ["saqqara", "sakkara", "step pyramid", "djoser", "zoser", "سقارة", "هرم زوسر", "زوسر"]},
    "GIZ005": {"specific": ["dahshur", "dahshour", "bent pyramid", "red pyramid", "دهشور"]},
    "CAI001": {"specific": ["egyptian museum", "tahrir museum", "museum of egyptian antiquities",
                            "المتحف المصري", "متحف التحرير"]},
    "CAI002": {"specific": ["nmec", "national museum of egyptian civilization", "civilization museum",
                            "museum of civilization", "متحف الحضارة", "المتحف القومي للحضارة"]},
    "CAI003": {"specific": ["citadel of saladin", "saladin citadel", "saladin", "cairo citadel", "salah el din citadel",
                            "mohammed ali mosque", "muhammad ali mosque", "alabaster mosque",
                            "قلعة صلاح الدين", "صلاح الدين", "مسجد محمد علي", "قلعة القاهرة"]},
    "ALX001": {"specific": ["bibliotheca alexandrina", "bibliotheca", "library of alexandria", "alexandria library",
                            "مكتبة الإسكندرية"]},
    "ALX002": {"specific": ["qaitbay", "qaitbey", "quaitbay", "kait bey", "قايتباي", "قلعة قايتباي"]},
    "ALX003": {"specific": ["kom el shoqafa", "kom el shuqafa", "kom al shuqafa", "catacombs", "كوم الشقافة"]},
    "ALX004": {"specific": ["pompeys pillar", "pompey pillar", "pompey", "serapeum", "عمود السواري", "السرابيوم"]},
    "ALX005": {"specific": ["stanley bridge", "stanley", "كوبري ستانلي", "ستانلي"]},
    "SSN001": {"specific": ["saint catherine", "st catherine", "st. catherine", "mount sinai", "jabal musa",
                            "mount moses", "burning bush", "سانت كاترين", "جبل موسى", "دير سانت كاترين"]},
    "SSN002": {"specific": ["ras mohammed", "ras muhammad", "ras mohamed", "رأس محمد"]},
    "FYM001": {"specific": ["wadi el hitan", "wadi al hitan", "whale valley", "valley of the whales", "وادي الحيتان"]},
    "FYM002": {"specific": ["wadi el rayan", "wadi al rayan", "wadi rayan", "وادي الريان"]},
    "MTR001": {"specific": ["siwa", "سيوة"]},
    "RED001": {"specific": ["el gouna", "gouna", "الجونة"]},
    "LXR001": {"specific": ["karnak", "الكرنك"]},
    "LXR002": {"specific": ["luxor temple", "temple of luxor", "معبد الأقصر"]},
    "LXR003": {"specific": ["hatshepsut", "deir el bahari", "deir al bahari", "حتشبسوت", "الدير البحري"]},
    "LXR004": {"specific": ["valley of the kings", "وادي الملوك"]},
    "LXR005": {"specific": ["medinet habu", "habu", "مدينة هابو", "هابو"]},
    "ASW001": {"specific": ["philae", "فيلة"]},
    "ASW002": {"specific": ["abu simbel", "أبو سمبل"]},
    "ASW003": {"specific": ["kom ombo", "كوم أمبو"]},
    "ASW004": {"specific": ["nubian museum", "nubia museum", "متحف النوبة"]},
    "ASW005": {"specific": ["unfinished obelisk", "المسلة الناقصة"]},
}


def _name_parts(name: str) -> List[str]:
    """'Mohammed Ali Mosque - Citadel of Saladin' -> [full, 'Mohammed Ali Mosque', 'Citadel of Saladin'];
    'Wadi El-Hitan (Whale Valley)' -> [full, 'Wadi El-Hitan', 'Whale Valley']."""
    parts = [name]
    inner = re.findall(r"\(([^)]+)\)", name)
    outer = re.sub(r"\([^)]*\)", " ", name)
    parts += [p.strip() for p in re.split(r"\s[-–]\s", outer) if p.strip()]
    parts += [p.strip() for p in inner if p.strip()]
    return list(dict.fromkeys(parts))


@dataclass(frozen=True)
class _Alias:
    attraction_id: str
    pattern: re.Pattern
    generic: bool


class AliasMatcher:
    def __init__(self, attractions: Iterable[Tuple[str, str, str]]):
        """attractions: (attraction_id, name_en, name_ar)."""
        aliases: List[_Alias] = []
        for aid, name_en, name_ar in attractions:
            specific = [aid, *_name_parts(name_en), *_name_parts(name_ar)]
            specific += MANUAL_ALIASES.get(aid, {}).get("specific", [])
            generic = MANUAL_ALIASES.get(aid, {}).get("generic", [])
            for a in dict.fromkeys(specific):
                if len(normalize(a)) >= 3:
                    aliases.append(_Alias(aid, alias_pattern(a), False))
            for a in generic:
                aliases.append(_Alias(aid, alias_pattern(a), True))
        self._aliases = aliases

    def match(self, text: str) -> List[str]:
        """Attraction ids named in text, in order of first mention."""
        norm = normalize(text)
        hits: List[Tuple[int, int, str, bool]] = []
        for al in self._aliases:
            for m in al.pattern.finditer(norm):
                hits.append((m.start(), m.end(), al.attraction_id, al.generic))
        # Longest span first; drop spans overlapping an already-accepted span.
        hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
        accepted: List[Tuple[int, int, str, bool]] = []
        for h in hits:
            if all(h[1] <= a[0] or h[0] >= a[1] for a in accepted):
                accepted.append(h)
        if any(not h[3] for h in accepted):
            accepted = [h for h in accepted if not h[3]]
        ordered = sorted(accepted, key=lambda h: h[0])
        return list(dict.fromkeys(h[2] for h in ordered))


_STOP = {"the", "of", "and", "in", "on", "at", "el", "al", "a", "an", "de", "la"}


def name_tokens_in_query(name: str, query: str) -> bool:
    """True if every significant token of `name` appears in `query` (order-free).
    Used for Tier 2 names: 'Mosque of Ibn Tulun' matches 'ibn tulun mosque opening hours'."""
    name_n, q_n = normalize(name), normalize(query)
    if not name_n:
        return False
    if _is_arabic(name_n):
        return bool(alias_pattern(name_n).search(q_n))
    tokens = [t for t in name_n.split() if t not in _STOP and len(t) >= 3]
    q_tokens = set(q_n.split())
    return bool(tokens) and all(t in q_tokens for t in tokens)
