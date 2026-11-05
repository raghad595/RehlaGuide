"""Structure-aware bilingual Markdown chunker + Qdrant indexer.

Run (after ingest_sql.py):  PYTHONPATH=. python data/scripts/ingest_vectors.py
Options:  --skip-index   only build data/processed_chunks.json (no embeddings)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from sqlmodel import Session, select

from app.db.database import engine
from app.models import Attraction
from rag.chunk_documents import parse_attraction_sections

DATA_DIR = Path(__file__).resolve().parents[1]
RAW_MD_DIR = DATA_DIR / "raw_md"
CHUNKS_JSON = DATA_DIR / "processed_chunks.json"
QDRANT_PATH = DATA_DIR / "qdrant_db"
COLLECTION = "rehlaguide_attractions"
EMBED_MODEL = "intfloat/multilingual-e5-small"
EXPECTED_CHUNKS_PER_LANG = 3

# Canonical sub-section keys, with EN + AR aliases (normalized form).
SECTION_ALIASES: Dict[str, List[str]] = {
    "overview": ["overview", "نبذة عامة"],
    "history": ["historical background", "history", "الخلفية التاريخية"],
    "culture": ["cultural significance", "الأهمية الثقافية"],
    "highlights": ["main highlights", "highlights", "أبرز المعالم"],
    "guidance": ["visitor guidance", "visitor tips", "إرشادات الزوار"],
    "accessibility": ["accessibility", "سهولة الوصول"],
    "faq": ["common questions", "faq", "frequently asked questions", "أسئلة شائعة"],
    "sources": ["sources", "references", "المصادر"],
}
ALIAS_TO_KEY = {alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases}

CHUNK_GROUPS: Dict[str, List[str]] = {
    "history_and_significance": ["overview", "history", "culture"],
    "highlights_and_faq": ["highlights", "faq"],
    "visitor_guidance_and_access": ["guidance", "accessibility", "sources"],
}

HEADER_TEMPLATES = {
    "en": "Attraction: {name_en} | City: {city} | Category: {category} | Section: {chunk_type}",
    "ar": "المعلم: {name_ar} | المدينة: {city} | الفئة: {category} | القسم: {chunk_type}",
}

PART_RE = re.compile(r"^##\s*Part\s*(\d)\b.*$", re.IGNORECASE | re.MULTILINE)
BOLD_HEADING_RE = re.compile(r"^\s*\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")


@dataclass
class Chunk:
    chunk_id: str
    attraction_id: str
    name_en: str
    name_ar: str
    language: str
    chunk_type: str
    city: str
    category: str
    visit_duration_minutes: int
    source_file: str
    text: str


def normalize_heading(text: str) -> str:
    text = text.replace("‏", "").replace("‎", "")
    text = re.sub(r"\s+", " ", text).strip().rstrip(":：").strip().lower()
    return text


def split_languages(markdown: str) -> Dict[str, str]:
    """Return {'en': ..., 'ar': ...} using '## Part 1' / '## Part 2' headings."""
    markers = list(PART_RE.finditer(markdown))
    parts: Dict[str, str] = {}
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
        lang = {"1": "en", "2": "ar"}.get(m.group(1))
        if lang:
            parts[lang] = markdown[m.end():end]
    return parts


def split_subsections(block: str) -> Dict[str, str]:
    """Map canonical section key -> body text (including its bold heading line)."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in block.splitlines():
        m = BOLD_HEADING_RE.match(line)
        key = ALIAS_TO_KEY.get(normalize_heading(m.group(1))) if m else None
        if key:
            current = key
            sections[current] = [f"**{m.group(1).strip()}**"]
            continue
        if current:
            sections[current].append(line)
    return {k: re.sub(r"\n{3,}", "\n\n", "\n".join(v)).strip() for k, v in sections.items()}


def build_chunks(attractions: Dict[str, Attraction]) -> tuple[List[Chunk], List[str]]:
    chunks: List[Chunk] = []
    errors: List[str] = []
    for md_path in sorted(RAW_MD_DIR.glob("*.md")):
        markdown = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        parts = split_languages(markdown)
        for lang in ("en", "ar"):
            if lang not in parts:
                errors.append(f"{md_path.name}: missing '## Part {1 if lang == 'en' else 2}' section")
        for lang, section_text in parts.items():
            for sec in parse_attraction_sections(section_text):
                attraction = attractions.get(sec.attraction_id or "")
                if attraction is None:
                    errors.append(
                        f'{md_path.name} [{lang}] unmatched heading: "### {sec.title}" '
                        f"(attraction_id={sec.attraction_id!r} not in SQLite)"
                    )
                    continue
                subs = split_subsections(sec.content)
                missing = [k for k in SECTION_ALIASES if k not in subs]
                if missing:
                    errors.append(f"{md_path.name} [{lang}] {attraction.attraction_id}: missing sub-sections {missing}")
                for chunk_type, keys in CHUNK_GROUPS.items():
                    body = "\n\n".join(subs[k] for k in keys if k in subs)
                    if not body:
                        continue
                    header = HEADER_TEMPLATES[lang].format(
                        name_en=attraction.name_en, name_ar=attraction.name_ar,
                        city=attraction.city, category=attraction.category, chunk_type=chunk_type,
                    )
                    chunks.append(Chunk(
                        chunk_id=f"{attraction.attraction_id}:{lang}:{chunk_type}",
                        attraction_id=attraction.attraction_id,
                        name_en=attraction.name_en,
                        name_ar=attraction.name_ar,
                        language=lang,
                        chunk_type=chunk_type,
                        city=attraction.city,
                        category=attraction.category,
                        visit_duration_minutes=attraction.visit_duration_minutes,
                        source_file=md_path.name,
                        text=f"{header}\n\n{body}",
                    ))

    # Every SQLite attraction must have exactly 3 chunks per language.
    per = Counter((c.attraction_id, c.language) for c in chunks)
    for aid in attractions:
        for lang in ("en", "ar"):
            if per[(aid, lang)] != EXPECTED_CHUNKS_PER_LANG:
                errors.append(f"{aid} [{lang}]: {per[(aid, lang)]} chunks (expected {EXPECTED_CHUNKS_PER_LANG})")
    dupes = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    if dupes:
        errors.append(f"duplicate chunk ids: {dupes}")
    return chunks, errors


def index_chunks(chunks: List[Chunk]) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    # e5 models expect a "passage: " prefix for documents ("query: " for queries).
    vectors = model.encode(
        [f"passage: {c.text}" for c in chunks],
        batch_size=16, normalize_embeddings=True, show_progress_bar=True,
    )

    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        for field in ("attraction_id", "language", "city", "chunk_type", "category"):
            client.create_payload_index(COLLECTION, field, PayloadSchemaType.KEYWORD)
        client.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=vec.tolist(),
                    payload=asdict(c),
                )
                for c, vec in zip(chunks, vectors)
            ],
        )
        return client.count(COLLECTION, exact=True).count
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true", help="only write processed_chunks.json")
    args = ap.parse_args()

    with Session(engine) as session:
        attractions = {a.attraction_id: a for a in session.exec(select(Attraction)).all()}
    if not attractions:
        print("ERROR: no attractions in SQLite - run data/scripts/ingest_sql.py first.")
        return 1

    chunks, errors = build_chunks(attractions)
    CHUNKS_JSON.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Chunking diagnostic report ===")
    by_file: Dict[str, Counter] = defaultdict(Counter)
    for c in chunks:
        by_file[c.source_file][c.language] += 1
    print(f"  {'file':<48} {'en':>4} {'ar':>4}")
    for name in sorted(by_file):
        print(f"  {name:<48} {by_file[name]['en']:>4} {by_file[name]['ar']:>4}")
    lang_totals = Counter(c.language for c in chunks)
    expected = len(attractions) * 2 * EXPECTED_CHUNKS_PER_LANG
    print(f"  TOTAL chunks: {len(chunks)} (en={lang_totals['en']}, ar={lang_totals['ar']}); "
          f"expected {expected} = {len(attractions)} attractions x 2 languages x 3")
    print(f"  Wrote {CHUNKS_JSON.relative_to(DATA_DIR.parent)}")

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  ERROR {e}")
        return 1

    if not args.skip_index:
        n = index_chunks(chunks)
        print(f"  Qdrant collection '{COLLECTION}' at {QDRANT_PATH.relative_to(DATA_DIR.parent)}: {n} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
