#!/usr/bin/env python3
"""Scan data/distill/failed.jsonl for tokens that look like Hindi
numerals the parser doesn't recognize.

Heuristic: for every token in every failed row's input, if it's not in
any known vocabulary (Hindi or English numerals, unit words, common
English / Hindi connector words) but is a close lexical match to an
existing Hindi numeral, surface it as a candidate variant.

Output is sorted by frequency. The script does NOT modify the parser;
it's a read-only audit so we can choose which variants to add by hand.

Usage:
    python scripts/audit_unrecognized_numerals.py
    python scripts/audit_unrecognized_numerals.py --min-count 3 --top 50
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amount_parser import _EN_NUMERAL, _EN_UNIT_WORD, _HI_NUMERAL, _HI_UNIT_WORD  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAILED = REPO_ROOT / "data" / "distill" / "failed.jsonl"

# Connector / filler words that show up in money inputs but aren't numerals.
# Curated, not exhaustive — anything not in here just gets weaker noise filtering.
_NOISE_TOKENS = {
    # Hindi connectors / common words
    "ka", "ki", "ke", "kya", "hai", "mein", "se", "ko", "aur", "ya",
    "rupay", "rupaya", "rupaye", "rupye", "rs", "rupees", "rupee",
    "paisa", "paise", "paisay",
    "do", "le", "liya", "liye", "kiya", "diya", "diye", "tha", "thi", "the",
    "kar", "kara", "kare", "karna", "karne", "kiya", "raha", "rahi", "rahe",
    "wala", "wali", "wale", "waala", "waali", "waale",
    "yeh", "woh", "ye", "wo", "ji", "na", "haan", "han", "nahi", "nahin",
    "phir", "abhi", "kal", "aaj", "kabhi", "bhi", "to", "toh", "bas",
    "thoda", "thodi", "zyada", "kam",
    "mera", "meri", "mere", "tera", "teri", "tere", "uska", "uski", "uske",
    "apna", "apni", "apne", "humara", "hamari", "hamare",
    "main", "tum", "aap", "vo", "ham", "hum",
    # English connectors
    "the", "a", "an", "of", "for", "on", "in", "at", "to", "and", "or",
    "with", "from", "i", "we", "you", "he", "she", "it", "they",
    "is", "was", "were", "am", "are", "be", "been", "being",
    "spent", "bought", "paid", "got", "had", "have", "has",
    "rupees", "rupee", "rs", "inr", "dollars", "dollar", "usd",
    "expense", "spending", "money", "cash",
    # Common items appearing in inputs (just to reduce noise — not exhaustive)
    "chai", "coffee", "tea", "milk", "bread", "egg", "eggs",
    "dosa", "idli", "samosa", "biryani", "pizza", "burger",
    "phone", "laptop", "shirt", "kurta", "jeans", "shoes", "watch",
    "car", "bike", "auto", "taxi", "uber", "ola", "petrol", "diesel",
    "movie", "ticket", "ticket", "show",
    "today", "yesterday", "tomorrow", "morning", "evening", "night",
    "lunch", "dinner", "breakfast", "snack", "snacks",
}

_KNOWN_VOCAB = (
    set(_HI_NUMERAL) | set(_HI_UNIT_WORD)
    | set(_EN_NUMERAL) | set(_EN_UNIT_WORD)
    | _NOISE_TOKENS
)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed", type=Path, default=DEFAULT_FAILED)
    p.add_argument("--min-count", type=int, default=2,
                   help="Only report tokens appearing >= this many times")
    p.add_argument("--top", type=int, default=80,
                   help="Show this many top results")
    p.add_argument("--similarity", type=float, default=0.72,
                   help="difflib SequenceMatcher ratio threshold (0..1) "
                        "for considering a token to be a numeral variant")
    p.add_argument("--show-contexts", type=int, default=2,
                   help="Per token, print up to N example input snippets")
    p.add_argument("--out", type=Path, default=None,
                   help="If set, write results to this file as UTF-8 "
                        "(avoids Windows console encoding issues).")
    args = p.parse_args()

    if not args.failed.exists():
        raise SystemExit(f"Failed file not found: {args.failed}")

    hi_keys = list(_HI_NUMERAL.keys())
    counts: Counter = Counter()
    contexts: dict[str, list[str]] = defaultdict(list)

    with args.failed.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = row.get("input")
            if not isinstance(inp, str):
                continue
            for m in _TOKEN_RE.finditer(inp):
                tok = m.group(0).lower()
                if len(tok) < 3:
                    continue
                if tok in _KNOWN_VOCAB:
                    continue
                counts[tok] += 1
                if len(contexts[tok]) < args.show_contexts:
                    contexts[tok].append(inp)

    # For each unknown token, find its closest Hindi numeral neighbor.
    suggestions: list[tuple[int, str, str, int, float]] = []  # (count, tok, nearest, value, ratio)
    for tok, n in counts.items():
        if n < args.min_count:
            continue
        # difflib gives 0..1 similarity ratio. Use get_close_matches for top-1.
        matches = difflib.get_close_matches(tok, hi_keys, n=1, cutoff=args.similarity)
        if not matches:
            continue
        nearest = matches[0]
        ratio = difflib.SequenceMatcher(None, tok, nearest).ratio()
        suggestions.append((n, tok, nearest, _HI_NUMERAL[nearest], ratio))

    suggestions.sort(key=lambda r: (-r[0], -r[4]))
    suggestions = suggestions[: args.top]

    if not suggestions:
        print(f"No unrecognized numeral-like tokens found (min-count={args.min_count}).")
        return 0

    lines = []
    lines.append(f"Top {len(suggestions)} unrecognized tokens (min-count={args.min_count}, "
                 f"similarity>={args.similarity}):")
    lines.append("")
    lines.append(f"{'count':>5}  {'token':25s}  {'nearest':20s}  value  ratio  example")
    lines.append("-" * 110)
    for n, tok, nearest, value, ratio in suggestions:
        ex = contexts[tok][0] if contexts[tok] else ""
        if len(ex) > 45:
            ex = ex[:42] + "..."
        lines.append(f"{n:>5}  {tok:25s}  {nearest:20s}  {value:>4}   {ratio:.2f}   {ex!r}")
    output = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {len(suggestions)} suggestions to {args.out}")
    else:
        try:
            print(output)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
