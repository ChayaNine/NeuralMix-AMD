"""Scrape audio engineering articles from Sound on Sound and iZotope Learn.

Rule-based section extraction — no external LLM required.
Each heading + content block becomes one Q&A training pair.

Target: ~400 pairs, categories: eq, compression, reverb_delay, concepts.

Usage:
    python scrape_audio_web.py [--output data/raw/web_pairs.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from dataset import TrainingPair

HEADERS = {"User-Agent": "NeuralMix-scraper/1.0"}
DELAY = 0.8

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

IZOTOPE_ARTICLES = [
    "https://www.izotope.com/en/learn/audio-compression-basics.html",
    "https://www.izotope.com/en/learn/how-to-eq-vocals.html",
    "https://www.izotope.com/en/learn/how-to-use-a-compressor.html",
    "https://www.izotope.com/en/learn/mixing-vocals.html",
    "https://www.izotope.com/en/learn/how-to-use-reverb-in-a-mix.html",
    "https://www.izotope.com/en/learn/how-to-mix-drums.html",
    "https://www.izotope.com/en/learn/how-to-mix-bass.html",
    "https://www.izotope.com/en/learn/how-to-mix-guitars.html",
    "https://www.izotope.com/en/learn/understanding-eq-frequency-ranges.html",
    "https://www.izotope.com/en/learn/how-to-sidechain-compress.html",
    "https://www.izotope.com/en/learn/gain-staging.html",
    "https://www.izotope.com/en/learn/what-is-limiting.html",
    "https://www.izotope.com/en/learn/mastering-for-streaming.html",
    "https://www.izotope.com/en/learn/parallel-compression.html",
    "https://www.izotope.com/en/learn/attack-and-release.html",
]

IZOTOPE_INDEX_URLS = [
    "https://www.izotope.com/en/learn/",
    "https://www.izotope.com/en/learn/mixing/",
    "https://www.izotope.com/en/learn/mastering/",
]

WAVES_ARTICLES = [
    "https://www.waves.com/how-to-eq-vocals",
    "https://www.waves.com/how-to-compress-vocals",
    "https://www.waves.com/mixing-drums",
    "https://www.waves.com/how-to-use-reverb",
    "https://www.waves.com/how-to-mix-bass",
]

MAX_PAGES_PER_SOURCE = 40

RELEVANCE_KEYWORDS = {
    "eq", "equaliz", "compress", "reverb", "delay", "frequency", "threshold",
    "attack", "release", "ratio", "db", "gain", "mix", "stem", "vocal", "drum",
    "bass", "guitar", "kick", "snare", "dynamics", "limiter", "saturati",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        if resp.status_code == 200:
            return resp.text
        print(f"  HTTP {resp.status_code}: {url}")
    except requests.RequestException as e:
        print(f"  Error fetching {url}: {e}")
    return None


def is_relevant(soup: BeautifulSoup) -> bool:
    text = soup.get_text().lower()
    return sum(1 for kw in RELEVANCE_KEYWORDS if kw in text) >= 4


def clean_text(element: Tag) -> str:
    parts: list[str] = []
    for child in element.descendants:
        if not hasattr(child, "name"):
            t = str(child).strip()
            if t:
                parts.append(t)
        elif child.name in ("code", "tt"):
            parts.append(f"`{child.get_text(strip=True)}`")
        elif child.name == "pre":
            parts.append(f"\n```\n{child.get_text().strip()}\n```")
    result = " ".join(parts)
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def infer_category(heading: str, content: str) -> str:
    combined = (heading + " " + content).lower()
    if any(w in combined for w in ["eq", "equaliz", "frequenc", "shelf", "high-pass", "low-pass", "notch", "boost", "cut"]):
        return "eq"
    if any(w in combined for w in ["compress", "ratio", "threshold", "attack", "release", "knee", "parallel"]):
        return "compression"
    if any(w in combined for w in ["reverb", "delay", "echo", "room", "plate", "hall", "pre-delay", "decay"]):
        return "reverb_delay"
    if any(w in combined for w in ["limit", "ceiling", "lufs", "loudness", "master", "clip"]):
        return "limiting"
    if any(w in combined for w in ["chain", "channel strip", "stem", "vocal chain", "drum chain"]):
        return "stem_chain"
    return "concepts"


def heading_to_question(heading: str, category: str) -> str:
    h = heading.strip().rstrip(".")
    if h.endswith("?"):
        return h
    h_lower = h.lower()
    if h_lower.startswith("how to") or h_lower.startswith("how do"):
        return f"{h}?"
    if h_lower.startswith("what is") or h_lower.startswith("what are"):
        return f"{h}?"
    if h_lower.startswith("when to") or h_lower.startswith("when should"):
        return f"{h}?"
    prefix_map = {
        "eq": f"How do I use {h.lower()}?",
        "compression": f"How do I apply {h.lower()}?",
        "reverb_delay": f"How do I set up {h.lower()}?",
        "limiting": f"How do I use {h.lower()} correctly?",
    }
    return prefix_map.get(category, f"What is {h}?")


def extract_pairs_from_article(html: str, url: str) -> Iterator[TrainingPair]:
    soup = BeautifulSoup(html, "lxml")

    if not is_relevant(soup):
        return

    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside", "form"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("div", class_=re.compile(r"entry-content|post-content|article-body|article__body|content__body"))
        or soup.find("div", class_=re.compile(r"learn-content|tutorial-content|blog-content"))
        or soup.find("main")
        or soup.find("div", class_="content")
    )
    if not main:
        return

    current_heading: str | None = None
    current_parts: list[str] = []

    def flush() -> TrainingPair | None:
        if not current_heading or not current_parts:
            return None
        content = "\n\n".join(p for p in current_parts if p.strip())
        if len(content) < 80:
            return None
        category = infer_category(current_heading, content)
        question = heading_to_question(current_heading, category)
        return TrainingPair(
            instruction=question,
            input="",
            output=content + f"\n\nSource: {url}",
            source="plugin_docs",
            category=category,  # type: ignore[arg-type]
            language="en",
            verified=False,
        )

    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "ul", "ol", "blockquote"]):
        if el.name in ("h1", "h2", "h3", "h4"):
            pair = flush()
            if pair:
                yield pair
            current_heading = el.get_text(strip=True)
            current_parts = []
        else:
            text = clean_text(el)
            if text:
                current_parts.append(text)

    pair = flush()
    if pair:
        yield pair


# ---------------------------------------------------------------------------
# Index crawlers
# ---------------------------------------------------------------------------

def collect_izotope_urls() -> list[str]:
    seen: set[str] = set(IZOTOPE_ARTICLES)
    urls: list[str] = list(IZOTOPE_ARTICLES)
    base = "https://www.izotope.com"

    for index_url in IZOTOPE_INDEX_URLS:
        html = fetch(index_url)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base, href)
            parsed = urlparse(full)
            if (
                parsed.netloc == "www.izotope.com"
                and "/learn/" in parsed.path
                and full not in seen
                and not full.endswith("/learn/")
                and len(parsed.path.strip("/").split("/")) >= 3
            ):
                seen.add(full)
                urls.append(full)
                if len(urls) >= MAX_PAGES_PER_SOURCE:
                    return urls
        time.sleep(DELAY)

    return urls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape audio web articles into training pairs")
    parser.add_argument("--output", default="data/raw/web_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Collecting iZotope article URLs...")
    urls = collect_izotope_urls()
    urls += WAVES_ARTICLES
    print(f"Total URLs to process: {len(urls)}")

    total = 0
    with output.open("w", encoding="utf-8") as f:
        for url in urls:
            print(f"Scraping: {url}")
            html = fetch(url)
            if not html:
                continue

            page_count = 0
            for pair in extract_pairs_from_article(html, url):
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                page_count += 1
                total += 1
                if args.limit and total >= args.limit:
                    break

            if page_count:
                print(f"  → {page_count} pairs")
            if args.limit and total >= args.limit:
                break
            time.sleep(DELAY)

    print(f"\nTotal pairs written: {total}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
