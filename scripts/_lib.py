"""Shared constants and helpers used across pipeline stages.

Anything that has to stay in lockstep between training and evaluation —
the system prompt, the schema, the category list — lives here so a
change can't silently desync.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests
from jsonschema import Draft202012Validator


CATEGORIES = [
    "Food", "Drinks", "Groceries", "Transport", "Shopping",
    "Entertainment", "Bills", "Health", "Education", "Personal",
    "Gifts", "Income", "Other",
]
TYPES = ["expense", "income"]
CURRENCIES = ["INR", "USD"]


SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["transactions"],
    "additionalProperties": False,
    "properties": {
        "transactions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["amount", "currency", "item", "category", "type"],
                "additionalProperties": False,
                "properties": {
                    "amount": {"type": "number"},
                    "currency": {"type": "string", "enum": CURRENCIES},
                    "item": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "type": {"type": "string", "enum": TYPES},
                },
            },
        }
    },
}

_VALIDATOR = Draft202012Validator(SCHEMA)


def schema_errors(obj: Any) -> list[str]:
    """Return a list of human-readable schema-validation error strings."""
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in _VALIDATOR.iter_errors(obj)
    ]


def is_schema_valid(obj: Any) -> bool:
    return _VALIDATOR.is_valid(obj)


# System prompt used at training time AND at inference time. They MUST match.
SYSTEM_PROMPT = """You convert voice-transcribed transaction descriptions into structured JSON.

Output ONLY a JSON object with this schema, no other text:
{"transactions":[{"amount":<number>,"currency":"INR"|"USD","item":"<lowercase singular noun phrase>","category":"<enum>","type":"expense"|"income"}]}

Categories: Food, Drinks, Groceries, Transport, Shopping, Entertainment, Bills, Health, Education, Personal, Gifts, Income, Other.

Rules:
- Currency defaults to INR. Use USD only when the input explicitly says "dollars" or contains "$".
- Amounts: "k" = ×1000, "hazaar" = ×1000, "sau" = ×100, "lakh" = ×100000. Convert number-words ("five hundred") to digits.
- type is "expense" by default; "income" only for explicit salary, cashback, refund, gift received, payment received.
- For disfluencies and corrections ("500 wait no 600"), output the CORRECTED amount only.
- For ambiguous items ("that thing", "stuff"), use item "unspecified" and category "Other".
- Item field: lowercase singular noun phrase ("uber ride", "beer", "chai" — not "Beers" or "Uber").
- Multi-transaction inputs become multiple array entries in spoken order.
- Category heuristics: uber/ola/auto/petrol/bus/metro → Transport; beer/wine/chai/coffee/juice → Drinks; rent/electricity/wifi/recharge/gas → Bills; movie/netflix/concert → Entertainment; doctor/medicine/hospital → Health."""


SAMPLE_INPUTS: list[str] = [
    "500 rs on beer 50 rs on candy",
    "do sau rupay ka chai",
    "got 5000 salary today",
    "1.5k for shoes from myntra",
    "300 lunch 50 chai 200 uber back home",
    "500 on beer wait no 600 on beer",
    "umm paid 300 for that thing yesterday",
]


def build_messages(input_str: str, output_obj: dict | None = None) -> list[dict[str, str]]:
    """Build a chat-template message list. Assistant turn is included if
    ``output_obj`` is provided (training); omitted for inference prompts."""
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_str},
    ]
    if output_obj is not None:
        msgs.append({
            "role": "assistant",
            "content": json.dumps(output_obj, ensure_ascii=False, separators=(",", ":")),
        })
    return msgs


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Diversity focuses — used by both Stage 1 (full pair generation) and
# Stage 5 (input-only generation) to rotate the prompt's "focus" line.
# Order is fixed so a re-run hits the same focuses in the same order.
# ---------------------------------------------------------------------------

BATCH_FOCUSES: list[str] = [
    "Focus on clean formal English single-transaction inputs. Vary items widely.",
    "Focus on casual shorthand single-transaction inputs like '500 rs beer' or '200 uber'.",
    "Focus on Hinglish-heavy inputs using sau, hazaar, rupay, ka, ke liye. Mostly single transactions.",
    "Focus on numbers-as-words inputs like 'five hundred for petrol' or 'two thousand on medicines'.",
    "Focus on abbreviated amounts: k suffix, /- suffix, decimals like 1.5k/2.5k, and the rupee symbol.",
    "Focus on multi-transaction inputs with exactly 2 transactions per input.",
    "Focus on multi-transaction inputs with 3 to 5 transactions per input, mixing styles.",
    "Focus on disfluency and corrections: 'wait no', 'umm', 'I mean', stutters, restarts.",
    "Focus on ambiguous inputs that map to item 'unspecified' and category 'Other'.",
    "Focus on mixed Hinglish and English in the same multi-transaction string.",
    "Focus on Food category items: meals, snacks, street food, sweets, restaurants.",
    "Focus on Drinks category: chai, coffee, beer, wine, juice, smoothies, mocktails.",
    "Focus on Groceries: sabzi, milk, supermarket runs, kirana store purchases.",
    "Focus on Transport: uber, ola, auto, petrol, bus, metro, train, parking.",
    "Focus on Shopping: clothes, electronics, accessories, online orders (myntra, amazon, flipkart).",
    "Focus on Entertainment: movies, netflix, concerts, gaming, amusement parks.",
    "Focus on Bills: rent, electricity, wifi, mobile recharge, gas, DTH, maintenance.",
    "Focus on Health: doctor visits, medicine, hospital, lab tests, dentist, gym.",
    "Focus on Education: course fees, books, tuition, exam fees, online courses.",
    "Focus on Personal and Gifts: haircut, salon, gifts to family or friends, donations.",
    "Focus on Income cases: salary, cashback, refunds, freelance payment, gift received.",
    "Focus on very small amounts in the range 10 to 100 rupees.",
    "Focus on very large amounts in lakh range and big-ticket purchases (50000 and above).",
    "Focus on decimal and odd amounts like '1.5k', '2499', 'rs 99', '12500'.",
    "Focus on edge cases: occasional USD mentions, ambiguous transcripts, low-confidence inputs.",
]


# ---------------------------------------------------------------------------
# DeepSeek HTTP client (OpenAI-compatible).
# ---------------------------------------------------------------------------


def call_deepseek(
    prompt: str,
    *,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    max_tokens: int = 8000,
    temperature: float = 1.0,
    timeout: int = 300,
) -> str:
    """Single-shot non-streaming chat completion. Returns the assistant message."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# JSON extraction from model output. Tries direct parse, fence-stripping,
# then a balanced-brace scan that respects string contents.
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str | None:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else None


def _extract_braces(text: str) -> str | None:
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return None


def extract_json(text: str) -> dict | None:
    """Robustly pull a JSON object out of model output."""
    if not text:
        return None
    for cand in (text.strip(), _strip_fence(text), _extract_braces(text)):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# Input-line cleaning helpers (shared between legacy phase_inputs and the
# multi-provider _parse_input_lines). Migrated here in Slice 2 so both
# call sites use the same fixed regex.
# ---------------------------------------------------------------------------

# Strips leading bullets ("- ", "* ") and numbering ("1. ", "2) ").
# Deliberately narrow: must NOT match digit-led natural content like "500 beer".
# The previous regex r"^[\s\-\*\d]+[.)\s]+" had this bug.
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)")


def clean_input_line(line: str) -> str | None:
    """Strip leading bullets/numbering and surrounding whitespace.

    Strips:
      "- 500 beer"   -> "500 beer"
      "* 500 beer"   -> "500 beer"
      "1. 500 beer"  -> "500 beer"
      "2) 500 beer"  -> "500 beer"

    Does NOT strip digit-led natural content:
      "500 beer"     -> "500 beer"
      "1 lakh rent"  -> "1 lakh rent"

    Returns None for blank, fenced, or out-of-range inputs.
    """
    s = line.strip()
    if not s:
        return None
    if s.startswith("```") or s.lower().startswith("output:") or s.lower().startswith("example"):
        return None
    s = _LINE_PREFIX_RE.sub("", s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if len(s) < 3 or len(s) > 300:
        return None
    return s


def normalize_input(text: str) -> str:
    """Canonical dedupe key. Lower-cased, whitespace-collapsed."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Re-exports — placed at the bottom of _lib.py so amount_parser/validator can
# `from _lib import is_schema_valid, schema_errors` lazily without a circular
# load. Existing imports of _lib symbols are untouched.
# ---------------------------------------------------------------------------

from amount_parser import AmountCandidate, parse_amounts  # noqa: E402
from validator import (  # noqa: E402
    ValidationError, ValidationResult,
    validate_example, serialize_validation_result,
)
