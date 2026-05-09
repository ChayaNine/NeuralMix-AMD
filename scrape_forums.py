"""Scrape audio engineering Q&A from Reddit r/audioengineering.

Uses Reddit JSON API (no auth required for public posts, 60 req/min).
Fetches top posts from the subreddit + wiki/FAQ content.

Target: ~200 pairs, categories: eq, compression, concepts, analysis.

Usage:
    python scrape_forums.py [--output data/raw/forum_pairs.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterator

import requests

from dataset import TrainingPair

HEADERS = {
    "User-Agent": "NeuralMix-scraper/1.0 (audio engineering dataset)",
}
DELAY = 2.0  # Reddit asks for 1 req/2s for unauthenticated
MIN_SCORE = 50   # only include posts with enough upvotes
MIN_ANSWER_CHARS = 150

SUBREDDIT = "audioengineering"

# Top-level wiki pages to scrape for structured Q&A content.
WIKI_PAGES = [
    f"https://www.reddit.com/r/{SUBREDDIT}/wiki/index.json",
    f"https://www.reddit.com/r/{SUBREDDIT}/wiki/faq.json",
    f"https://www.reddit.com/r/{SUBREDDIT}/wiki/getting_started.json",
]

# Flairs or keywords that indicate a technical question worth training on.
USEFUL_FLAIRS = {"Question", "Tips and Techniques", "Discussion", "Resource"}
USEFUL_KEYWORDS = {
    "eq", "equaliz", "compress", "reverb", "delay", "frequency", "threshold",
    "attack", "release", "ratio", "db", "gain", "mix", "vocal", "drum",
    "bass", "guitar", "kick", "snare", "dynamics", "limiter", "sidechain",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            print("  Rate limited — waiting 60s...")
            time.sleep(60)
        else:
            print(f"  HTTP {resp.status_code}: {url}")
    except requests.RequestException as e:
        print(f"  Error: {e}")
    return None


def is_relevant(text: str) -> bool:
    text_lower = text.lower()
    return sum(1 for kw in USEFUL_KEYWORDS if kw in text_lower) >= 2


def clean_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def infer_category(title: str, body: str) -> str:
    combined = (title + " " + body).lower()
    if any(w in combined for w in ["eq", "equaliz", "frequenc", "shelf", "notch", "boost", "cut at"]):
        return "eq"
    if any(w in combined for w in ["compress", "ratio", "threshold", "attack", "release", "parallel"]):
        return "compression"
    if any(w in combined for w in ["reverb", "delay", "echo", "room", "plate", "pre-delay"]):
        return "reverb_delay"
    if any(w in combined for w in ["limit", "ceiling", "lufs", "loudness", "master"]):
        return "limiting"
    if any(w in combined for w in ["mud", "harsh", "cut through", "volume", "rms", "gain stag"]):
        return "analysis"
    return "concepts"


# ---------------------------------------------------------------------------
# Reddit post fetching
# ---------------------------------------------------------------------------

def fetch_top_posts(subreddit: str, limit: int = 100, time_filter: str = "all") -> list[dict]:
    posts = []
    after = None

    while len(posts) < limit:
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t={time_filter}"
        if after:
            url += f"&after={after}"

        data = fetch_json(url)
        if not data:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            if (
                post.get("score", 0) >= MIN_SCORE
                and not post.get("is_self") is False  # only text posts
                and post.get("selftext", "")
                and is_relevant(post.get("title", "") + " " + post.get("selftext", ""))
            ):
                posts.append(post)

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(DELAY)
        print(f"  Fetched {len(posts)} relevant posts so far...")

        if len(posts) >= limit:
            break

    return posts[:limit]


def fetch_comments(permalink: str) -> list[dict]:
    url = f"https://www.reddit.com{permalink}.json?limit=20&sort=top"
    data = fetch_json(url)
    if not data or len(data) < 2:
        return []
    time.sleep(DELAY)

    comments = []
    for child in data[1].get("data", {}).get("children", []):
        c = child.get("data", {})
        body = c.get("body", "")
        if (
            len(body) >= MIN_ANSWER_CHARS
            and c.get("score", 0) >= 5
            and body != "[deleted]"
            and body != "[removed]"
        ):
            comments.append(c)

    return sorted(comments, key=lambda c: c.get("score", 0), reverse=True)


def post_to_pair(post: dict, comment: dict) -> TrainingPair | None:
    title = post.get("title", "").strip()
    body = clean_markdown(post.get("selftext", ""))
    answer = clean_markdown(comment.get("body", ""))

    if len(title) < 15 or len(answer) < MIN_ANSWER_CHARS:
        return None

    instruction = title if title.endswith("?") else title.rstrip(".") + "?"
    input_text = body[:800] if body and len(body) > 20 else ""

    post_url = f"https://www.reddit.com{post.get('permalink', '')}"
    output = answer + f"\n\nSource: {post_url}"

    category = infer_category(title, body + answer)

    return TrainingPair(
        instruction=instruction,
        input=input_text,
        output=output,
        source="forums",
        category=category,  # type: ignore[arg-type]
        language="en",
        verified=False,
    )


# ---------------------------------------------------------------------------
# Reddit wiki scraping
# ---------------------------------------------------------------------------

def scrape_wiki_page(url: str) -> Iterator[TrainingPair]:
    data = fetch_json(url)
    if not data:
        return

    content = data.get("data", {}).get("content_md", "")
    if not content:
        return

    # Split wiki content by headings into Q&A pairs.
    sections = re.split(r"\n#{1,3} ", content)
    for section in sections[1:]:
        lines = section.strip().split("\n", 1)
        if len(lines) < 2:
            continue
        heading = lines[0].strip().rstrip("#").strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if len(body) < 80 or not is_relevant(heading + " " + body):
            continue

        category = infer_category(heading, body)
        question = heading if heading.endswith("?") else f"What is {heading.lower()}?"

        yield TrainingPair(
            instruction=question,
            input="",
            output=clean_markdown(body) + f"\n\nSource: r/audioengineering wiki",
            source="forums",
            category=category,  # type: ignore[arg-type]
            language="en",
            verified=False,
        )
    time.sleep(DELAY)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape r/audioengineering into training pairs")
    parser.add_argument("--output", default="data/raw/forum_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with output.open("w", encoding="utf-8") as f:
        # Wiki pages first (high quality, no rate limit concern).
        print("Scraping r/audioengineering wiki...")
        for wiki_url in WIKI_PAGES:
            for pair in scrape_wiki_page(wiki_url):
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                total += 1

        print(f"  {total} wiki pairs")

        # Top posts from the subreddit.
        print(f"\nFetching top posts from r/{SUBREDDIT}...")
        posts = fetch_top_posts(SUBREDDIT, limit=min(args.limit, 150))
        print(f"  {len(posts)} relevant posts found")

        for post in posts:
            if total >= args.limit:
                break
            comments = fetch_comments(post.get("permalink", ""))
            if not comments:
                continue
            pair = post_to_pair(post, comments[0])
            if not pair:
                continue
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            total += 1

    print(f"\nTotal pairs written: {total}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
