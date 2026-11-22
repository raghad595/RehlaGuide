"""Fetch Egyptian monuments/museums/sites from Wikidata (Tier 2, unverified) into a local snapshot.

Run:
    $env:PYTHONPATH="."
    .venv\\Scripts\\python.exe data/scripts/fetch_wikidata.py                    # Wikidata only (~5 min)
    .venv\\Scripts\\python.exe data/scripts/fetch_wikidata.py --enrich-wiki 150  # + Wikipedia lead summaries

Output (re-used offline by ingest_wikidata.py):
    data/raw_wikidata/egypt_wikidata_attractions.json
    data/raw_wikidata/egypt_wikidata_attractions.csv     (utf-8-sig, opens cleanly in Excel)

Design (avoids the 60 s query timeout):
  Stage 1  one ID-only query per class: ?item wdt:P31/wdt:P279* <class> ; wdt:P17 wd:Q79 (LIMIT 1000).
           Falls back to direct wdt:P31 if the subclass walk keeps timing out.
  Stage 2  batched detail queries (VALUES ?item {...}); every multi-valued property gets its own
           small query (combining them in one query multiplies rows and is what times out).
  Stage 3  optional Wikipedia REST lead extracts for the most notable items (cached on disk).

By design this NEVER requests opening hours, ticket prices or fees (Wikidata has no reliable
operational data for these sites; Tier 2 is history-only).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote, urlparse

import httpx

log = logging.getLogger("fetch_wikidata")

OUT_DIR = Path(__file__).resolve().parents[1] / "raw_wikidata"
JSON_PATH = OUT_DIR / "egypt_wikidata_attractions.json"
CSV_PATH = OUT_DIR / "egypt_wikidata_attractions.csv"
WIKI_CACHE_PATH = OUT_DIR / "wiki_summary_cache.json"

SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "RehlaGuideCapstoneBot/1.0 (https://github.com/raghad595/RehlaGuide; educational-capstone)"
REQUEST_DELAY_S = 1.5          # politeness delay between SPARQL requests
WIKI_DELAY_S = 0.25            # politeness delay between Wikipedia REST requests
MAX_RETRIES = 5
RETRY_STATUSES = {429, 500, 502, 503, 504}

CLASSES = {  # core classes: every Egyptian item (incl. subclasses)
    "Q33506": "museum",
    "Q4989906": "monument",
    "Q839954": "archaeological site",
    "Q570116": "tourist attraction",
}
# Extra classes: Islamic/Coptic Cairo etc. are typed as mosque/church, which the core classes
# never reach (no Nilometer, Ibn Tulun, Al-Azhar...). Wikidata holds ~65,000 Egyptian mosques
# (bulk import), so these are limited to NOTABLE items: >= MIN_SITELINKS Wikipedia articles.
EXTRA_CLASSES = {
    "Q32815": "mosque",
    "Q16970": "church building",
    "Q16560": "palace",
    "Q1340071": "nilometer",
    "Q162875": "mausoleum",
    "Q34627": "synagogue",
    "Q57821": "fortification",
}
MIN_SITELINKS = 1
ALL_CLASSES = {**CLASSES, **EXTRA_CLASSES}
EGYPT = "Q79"
GOVERNORATE_CLASS = "Q204910"   # governorate of Egypt
PER_CLASS_LIMIT = 1000           # page size; full pages are followed with OFFSET
MAX_PAGES = 3
DETAIL_BATCH = 150

FIELDS = [
    "qid", "name_en", "name_ar", "description_en", "description_ar", "coordinates",
    "architectural_styles_en", "architectural_styles_ar", "located_in_en", "located_in_ar",
    "governorate_en", "governorate_ar", "instance_of_en", "instance_of_ar", "inception",
    "heritage_en", "heritage_ar", "image_url", "wikipedia_en_url", "wikipedia_ar_url",
    "summary_en", "summary_ar", "source_classes", "fetched_at",
]
MULTI_VALUED = {  # output field stem -> triple pattern binding ?v for ?item
    "architectural_styles": "?item wdt:P149 ?v .",
    "located_in": "?item wdt:P131 ?v .",
    "instance_of": "?item wdt:P31 ?v .",
    "heritage": "?item wdt:P1435 ?v .",
    # Walk up the administrative hierarchy to the governorate (normalized city).
    "governorate": f"?item wdt:P131+ ?v . ?v wdt:P31 wd:{GOVERNORATE_CLASS} .",
}
QID_RE = re.compile(r"^Q\d+$")


# --------------------------------------------------------------------------- HTTP
class SparqlClient:
    def __init__(self) -> None:
        self.http = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
            timeout=httpx.Timeout(75.0, connect=15.0),
        )
        self._last = 0.0

    def query(self, sparql: str) -> List[dict]:
        delay = 2.0
        for attempt in range(1, MAX_RETRIES + 1):
            wait = REQUEST_DELAY_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.http.post(SPARQL_URL, data={"query": sparql, "format": "json"})
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                log.warning("SPARQL %s (attempt %d/%d); retrying in %.0fs", type(exc).__name__, attempt, MAX_RETRIES, delay)
            else:
                if r.status_code == 200:
                    return r.json()["results"]["bindings"]
                if r.status_code not in RETRY_STATUSES:
                    raise RuntimeError(f"SPARQL HTTP {r.status_code}: {r.text[:300]}")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                log.warning("SPARQL HTTP %d (attempt %d/%d); retrying in %.0fs", r.status_code, attempt, MAX_RETRIES, delay)
            time.sleep(delay)
            delay *= 2
        raise RuntimeError(f"SPARQL failed after {MAX_RETRIES} attempts")

    def close(self) -> None:
        self.http.close()


def _val(b: dict, key: str) -> Optional[str]:
    v = b.get(key, {}).get("value")
    return v.strip() if isinstance(v, str) and v.strip() else None


def _qid(uri: Optional[str]) -> Optional[str]:
    return uri.rsplit("/", 1)[-1] if uri else None


def _chunks(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- stage 1
def fetch_ids(client: SparqlClient, cls: str) -> List[str]:
    """IDs for one class. Pages with OFFSET while pages come back full (no ORDER BY: it would
    time out; Blazegraph returns a stable order for an identical query, and we de-duplicate)."""
    notable = (f"?item wikibase:sitelinks ?links . FILTER(?links >= {MIN_SITELINKS})"
               if cls in EXTRA_CLASSES else "")
    for path in ("wdt:P31/wdt:P279*", "wdt:P31"):
        ids: List[str] = []
        try:
            for page in range(MAX_PAGES):
                q = (f"SELECT DISTINCT ?item WHERE {{ ?item {path} wd:{cls} ; wdt:P17 wd:{EGYPT} . {notable} }} "
                     f"LIMIT {PER_CLASS_LIMIT} OFFSET {page * PER_CLASS_LIMIT}")
                batch = [i for i in (_qid(_val(b, "item")) for b in client.query(q)) if i]
                ids += batch
                if len(batch) < PER_CLASS_LIMIT:
                    break
            log.info("  %-20s %-18s -> %d items%s", ALL_CLASSES[cls], path, len(ids),
                     " (notable only)" if notable else "")
            return list(dict.fromkeys(ids))
        except RuntimeError as exc:
            if ids:  # keep the pages we got rather than restarting with a narrower path
                log.warning("  %s: later page failed (%s); keeping %d items", ALL_CLASSES[cls], exc, len(ids))
                return list(dict.fromkeys(ids))
            log.warning("  %s via %s failed (%s); trying a narrower query", ALL_CLASSES[cls], path, exc)
    return []


# --------------------------------------------------------------------------- stage 2
def _values(ids: List[str]) -> str:
    return "VALUES ?item { " + " ".join(f"wd:{q}" for q in ids) + " }"


def _format_inception(value: Optional[str], precision: Optional[str]) -> Optional[str]:
    """Wikidata time ('-2560-01-01T00:00:00Z', precision 9=year, 8=decade, 7=century, 6=millennium)."""
    if not value:
        return None
    m = re.match(r"^([+-]?)(\d+)-", value)
    if not m:
        return None
    year = int(m.group(2))
    bc = m.group(1) == "-"
    era = " BC" if bc else ""
    p = int(precision) if precision and precision.isdigit() else 9
    if p >= 9:
        return f"{year}{era}"
    if p == 8:
        return f"{year // 10 * 10}s{era}"
    if p == 7:
        c = (year - 1) // 100 + 1 if not bc else (year + 99) // 100
        suffix = "th" if 10 <= c % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(c % 10, "th")
        return f"{c}{suffix} century{era}"
    if p == 6:
        k = (year - 1) // 1000 + 1 if not bc else (year + 999) // 1000
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(k, "th")
        return f"{k}{suffix} millennium{era}"
    return f"{year}{era}"


def fetch_core(client: SparqlClient, ids: List[str]) -> Dict[str, dict]:
    q = f"""
    SELECT ?item (SAMPLE(?en) AS ?name_en) (SAMPLE(?ar) AS ?name_ar)
           (SAMPLE(?den) AS ?description_en) (SAMPLE(?dar) AS ?description_ar)
           (SAMPLE(?coord) AS ?coordinates) (SAMPLE(?img) AS ?image_url)
           (SAMPLE(?inc) AS ?inception_value) (SAMPLE(?prec) AS ?inception_precision)
           (SAMPLE(?wen) AS ?wikipedia_en_url) (SAMPLE(?war) AS ?wikipedia_ar_url)
    WHERE {{
      {_values(ids)}
      OPTIONAL {{ ?item rdfs:label ?en FILTER(LANG(?en) = "en") }}
      OPTIONAL {{ ?item rdfs:label ?ar FILTER(LANG(?ar) = "ar") }}
      OPTIONAL {{ ?item schema:description ?den FILTER(LANG(?den) = "en") }}
      OPTIONAL {{ ?item schema:description ?dar FILTER(LANG(?dar) = "ar") }}
      OPTIONAL {{ ?item wdt:P625 ?coord }}
      OPTIONAL {{ ?item wdt:P18 ?img }}
      OPTIONAL {{ ?item p:P571/psv:P571 [ wikibase:timeValue ?inc ; wikibase:timePrecision ?prec ] }}
      OPTIONAL {{ ?wen schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> }}
      OPTIONAL {{ ?war schema:about ?item ; schema:isPartOf <https://ar.wikipedia.org/> }}
    }} GROUP BY ?item"""
    out = {}
    for b in client.query(q):
        qid = _qid(_val(b, "item"))
        rec = {k: _val(b, k) for k in ("name_en", "name_ar", "description_en", "description_ar", "coordinates",
                                         "image_url", "wikipedia_en_url", "wikipedia_ar_url")}
        rec["inception"] = _format_inception(_val(b, "inception_value"), _val(b, "inception_precision"))
        out[qid] = rec
    return out


def fetch_multi(client: SparqlClient, ids: List[str], stem: str, pattern: str) -> Dict[str, Dict[str, List[str]]]:
    q = f"""
    SELECT ?item ?v ?en ?ar WHERE {{
      {_values(ids)}
      {pattern}
      OPTIONAL {{ ?v rdfs:label ?en FILTER(LANG(?en) = "en") }}
      OPTIONAL {{ ?v rdfs:label ?ar FILTER(LANG(?ar) = "ar") }}
    }}"""
    out: Dict[str, Dict[str, List[str]]] = {}
    for b in client.query(q):
        qid = _qid(_val(b, "item"))
        d = out.setdefault(qid, {"en": [], "ar": []})
        for lang in ("en", "ar"):
            label = _val(b, lang)
            if label and not QID_RE.match(label) and label not in d[lang]:
                d[lang].append(label)
    return out


# --------------------------------------------------------------------------- stage 3 (Wikipedia)
def _wiki_title(url: str) -> Optional[str]:
    path = urlparse(url).path
    return path.split("/wiki/", 1)[1] if "/wiki/" in path else None


def _lead_sentences(text: str, max_sentences: int = 3, max_chars: int = 700) -> str:
    parts = re.split(r"(?<=[.!?؟])\s+", (text or "").strip())
    out = ""
    for p in parts[:max_sentences]:
        if len(out) + len(p) > max_chars and out:
            break
        out = f"{out} {p}".strip()
    return out


def notability(rec: dict) -> int:
    return (3 * bool(rec.get("name_en") and rec.get("name_ar")) + 2 * bool(rec.get("image_url"))
            + 2 * bool(rec.get("wikipedia_en_url")) + 2 * bool(rec.get("wikipedia_ar_url"))
            + 2 * bool(rec.get("heritage_en")) + bool(rec.get("description_en")) + bool(rec.get("inception")))


def enrich_with_wikipedia(records: List[dict], top_n: int) -> int:
    cache: Dict[str, Optional[str]] = {}
    if WIKI_CACHE_PATH.exists():
        cache = json.loads(WIKI_CACHE_PATH.read_text(encoding="utf-8"))
    candidates = sorted((r for r in records if r.get("wikipedia_en_url") or r.get("wikipedia_ar_url")),
                        key=notability, reverse=True)[:top_n]
    filled = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20.0, follow_redirects=True) as http:
        for i, rec in enumerate(candidates, 1):
            for lang in ("en", "ar"):
                url = rec.get(f"wikipedia_{lang}_url")
                title = _wiki_title(url) if url else None
                if not title:
                    continue
                key = f"{lang}:{unquote(title)}"
                if key not in cache:
                    time.sleep(WIKI_DELAY_S)
                    try:
                        r = http.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}")
                        cache[key] = r.json().get("extract") if r.status_code == 200 else None
                    except (httpx.HTTPError, ValueError) as exc:
                        log.warning("Wikipedia %s failed: %s", key, exc)
                        continue  # not cached: retried next run
                if cache.get(key):
                    rec[f"summary_{lang}"] = _lead_sentences(cache[key])
                    filled += 1
            if i % 25 == 0:
                log.info("  Wikipedia: %d/%d items", i, len(candidates))
                WIKI_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    WIKI_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return filled


# --------------------------------------------------------------------------- main
def fetch_all() -> List[dict]:
    client = SparqlClient()
    try:
        log.info("Stage 1: item IDs per class (Egypt; pages of %d)", PER_CLASS_LIMIT)
        classes_of: Dict[str, List[str]] = {}
        for cls, label in ALL_CLASSES.items():
            for qid in fetch_ids(client, cls):
                classes_of.setdefault(qid, []).append(label)
        ids = sorted(classes_of, key=lambda q: int(q[1:]))
        log.info("  merged & de-duplicated: %d unique items", len(ids))

        log.info("Stage 2: details in batches of %d", DETAIL_BATCH)
        records: Dict[str, dict] = {q: {"qid": q} for q in ids}
        batches = list(_chunks(ids, DETAIL_BATCH))
        for n, batch in enumerate(batches, 1):
            for qid, core in fetch_core(client, batch).items():
                records[qid].update(core)
            for stem, pattern in MULTI_VALUED.items():
                for qid, labels in fetch_multi(client, batch, stem, pattern).items():
                    records[qid][f"{stem}_en"] = "; ".join(labels["en"]) or None
                    records[qid][f"{stem}_ar"] = "; ".join(labels["ar"]) or None
            log.info("  batch %d/%d done", n, len(batches))
    finally:
        client.close()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for qid in ids:
        rec = records[qid]
        for k in ("name_en", "name_ar"):  # Wikidata sometimes uses the bare QID as a label
            if rec.get(k) and QID_RE.match(rec[k]):
                rec[k] = None
        rec["source_classes"] = "; ".join(classes_of[qid])
        rec["fetched_at"] = now
        rows.append({f: rec.get(f) for f in FIELDS})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the Tier 2 Wikidata snapshot")
    ap.add_argument("--enrich-wiki", type=int, default=0, metavar="N",
                    help="fetch Wikipedia lead summaries for the N most notable items (default 0)")
    ap.add_argument("--from-snapshot", action="store_true",
                    help="skip Wikidata; re-use the saved snapshot (e.g. to add --enrich-wiki later)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.from_snapshot:
        rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        log.info("Loaded %d items from %s", len(rows), JSON_PATH.name)
    else:
        rows = fetch_all()

    if args.enrich_wiki > 0:
        log.info("Stage 3: Wikipedia lead summaries for top %d notable items", args.enrich_wiki)
        log.info("  summaries added: %d", enrich_with_wikipedia(rows, args.enrich_wiki))

    JSON_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    has = lambda f: sum(1 for r in rows if r.get(f))  # noqa: E731
    print("\n=== Wikidata snapshot ===")
    print(f"  items: {len(rows)}   (took {time.monotonic() - started:.0f}s)")
    for f in ("name_en", "name_ar", "description_en", "description_ar", "coordinates", "governorate_en",
              "inception", "heritage_en", "image_url", "wikipedia_en_url", "wikipedia_ar_url",
              "summary_en", "summary_ar"):
        print(f"  with {f:<18} {has(f):>5}")
    print(f"  wrote {JSON_PATH.relative_to(OUT_DIR.parents[1])} and {CSV_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
