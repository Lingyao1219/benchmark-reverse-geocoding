#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_wordcloud_all.py — Aggregate reasoning text from dataset*_*.jsonl and generate a word cloud (PDF)

Dependencies:
    pip install wordcloud matplotlib
"""

import json, ast, re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# ------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------
FILES = [
    "dataset1_claude-3-5-haiku-20241022.jsonl",
    "dataset2_claude-3-5-haiku-20241022.jsonl",
    "dataset3_claude-3-5-haiku-20241022.jsonl",
]
OUTPUT_PDF = "geoloc_reasoning_wordcloud_claude-3.5-haiku.pdf"
MAX_WORDS = 200
WIDTH, HEIGHT = 2400, 1200
BACKGROUND = "white"

# Custom stopwords (lowercase only)
STOPWORDS_EXTRA = {
    "along", "across", "around", "near", "toward",
    "front", "side", "back", "rear",
    "appears", "appear", "seems", "seem",
    "suggest", "suggests", "indicates",
    "likely", "maybe", "perhaps", "could",
    "typical", "common", "generic", "characteristic",
    "iconic", "distinctive", "clear", "clearly",
    "visible", "readable", "unmistakable",
    "large", "small", "big", "wide", "narrow",
    "high", "low", "mid", "mixed",
    "image", "photo", "picture", "scene", "view",
    "area", "region", "location", "place", "setting",
    "structure", "structures", "building", "buildings",
    "architecture", "style",
    "urban", "suburban", "rural", "outdoor", "indoor",
    "central", "downtown", "global", "globally"
}
# ------------------------------------------------------------------

STOPWORDS_ALL = STOPWORDS.union(STOPWORDS_EXTRA)
TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list[str]:
    return [
        t for t in TOKEN_RE.findall(text.lower())
        if t not in STOPWORDS_ALL and len(t) > 2
    ]

def bigramize(tokens: list[str]) -> list[str]:
    return [
        f"{a}_{b}"
        for a, b in zip(tokens, tokens[1:])
        if a not in STOPWORDS_ALL and b not in STOPWORDS_ALL
    ]

def reasoning_tokens(raw_json: str | None, location_info: dict | None) -> Counter:
    counter = Counter()

    def _add(node):
        if isinstance(node, str):
            unigrams = tokenize(node)
            counter.update(unigrams)
            counter.update(bigramize(unigrams))
        elif isinstance(node, dict):
            for v in node.values():
                _add(v)
        elif isinstance(node, list):
            for v in node:
                _add(v)

    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except Exception:
            try:
                parsed = ast.literal_eval(raw_json)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            _add(parsed.get("reasoning", {}))

    if isinstance(location_info, dict):
        _add(location_info.get("reasoning", {}))

    return counter


def main() -> None:
    token_counts = Counter()

    for file_path in FILES:
        path = Path(file_path)
        if not path.is_file():
            continue

        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_rsp = rec.get("raw_response")
                loc_inf = rec.get("location_info")
                token_counts.update(reasoning_tokens(raw_rsp, loc_inf))

    if not token_counts:
        raise RuntimeError("No reasoning text extracted. Check data files and format.")

    wc = WordCloud(
        width=WIDTH,
        height=HEIGHT,
        background_color=BACKGROUND,
        max_words=MAX_WORDS,
        colormap="coolwarm",
        collocations=False
    ).generate_from_frequencies(token_counts)

    plt.figure(figsize=(WIDTH / 100, HEIGHT / 100))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, format="pdf", dpi=300)
    plt.close()

    top_n = 50
    csv_path = "bigram_top50.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["bigram", "frequency"])
        for bigram, freq in token_counts.most_common(top_n):
            writer.writerow([bigram.replace("_", " "), freq])


if __name__ == "__main__":
    main()
