"""Deterministic amount parser. Text in -> list of AmountCandidate.

This module knows nothing about the JSON schema, transaction categories,
or model outputs. It is pure stdlib (re, dataclasses, math) so it can be
imported by training-time tooling AND embedded in runtime validators.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Literal


Status = Literal["active", "superseded"]
Source = Literal[
    "decimal_k", "k_suffix",
    "digits_lakh", "digits_hazaar", "digits_sau",
    "currency_symbol", "slash_dash",
    "digits",
    "number_word_en", "number_word_hi",
]
CurrencyHint = Literal["INR", "USD"]


@dataclass(frozen=True)
class AmountCandidate:
    value: float
    raw: str
    span: tuple[int, int]
    status: Status
    source: Source
    currency_hint: CurrencyHint | None = None


_DECIMAL_K_RE = re.compile(r"(\d+\.\d+)\s*[kK]\b")
_INT_K_RE     = re.compile(r"(?<!\.)\b(\d+)\s*[kK]\b")

_DIGITS_LAKH_RE   = re.compile(r"\b(\d+(?:\.\d+)?)\s*lakhs?\b", re.IGNORECASE)
_DIGITS_HAZAAR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hazaar|hazar|thousand)\b", re.IGNORECASE)
_DIGITS_SAU_RE    = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:sau|hundred)\b", re.IGNORECASE)

_CURRENCY_PREFIX_RE = re.compile(
    r"""
    (?P<sym>\$|₹|Rs\.?|rs\.?)      # currency symbol
    \s*
    (?P<num>\d+(?:[.,]\d+)*)       # number, allow commas/decimals
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SLASH_DASH_RE = re.compile(r"\b(\d+(?:[.,]\d+)*)\s*/-")

_PLAIN_DIGITS_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\b")

# ---------------------------------------------------------------------------
# English / Hindi number-word + unit  (lowest-priority matchers, priority 8-9)
# ---------------------------------------------------------------------------
_EN_NUMERAL = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
# Hindi numerals: spelling variants (chaar/char, chhe/che) reflect common
# voice-transcript transliterations of the same word — both intentional.
_HI_NUMERAL = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4,
    "paanch": 5, "chhe": 6, "che": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10,
}

_EN_UNIT_WORD = {"hundred": 100, "thousand": 1000, "lakh": 100000}
# Both `hazaar` and `hazar` are accepted transliterations.
_HI_UNIT_WORD = {"sau": 100, "hazaar": 1000, "hazar": 1000, "lakh": 100000}

_EN_SIMPLE_RE = re.compile(
    r"\b(" + "|".join(_EN_NUMERAL) + r")\s+(" + "|".join(_EN_UNIT_WORD) + r")\b",
    re.IGNORECASE,
)
# Compound: "twenty five hundred" = (20 + 5) * 100 = 2500
_EN_COMPOUND_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+"
    r"(one|two|three|four|five|six|seven|eight|nine)\s+(hundred)\b",
    re.IGNORECASE,
)
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_HI_UNIT_RE = re.compile(
    r"\b(" + "|".join(sorted(_HI_NUMERAL, key=len, reverse=True))
    + r")\s+(" + "|".join(sorted(_HI_UNIT_WORD, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_REJECT_KEYWORDS = (
    "phone", "mobile", "otp", "pin",
    "account", "upi id", "order id",
)
_KEYWORD_WINDOW_CHARS = 30  # roughly ~5 tokens
_MAX_PLAIN_DIGIT_LEN = 6  # reject 7+ consecutive digits (phone, OTP, account numbers)


def _match_decimal_k(text: str) -> list[AmountCandidate]:
    out = []
    for m in _DECIMAL_K_RE.finditer(text):
        out.append(AmountCandidate(
            value=float(m.group(1)) * 1000.0,
            raw=m.group(0),
            span=(m.start(), m.end()),
            status="active",
            source="decimal_k",
        ))
    return out


def _match_int_k(text: str) -> list[AmountCandidate]:
    out = []
    for m in _INT_K_RE.finditer(text):
        out.append(AmountCandidate(
            value=float(m.group(1)) * 1000.0,
            raw=m.group(0),
            span=(m.start(), m.end()),
            status="active",
            source="k_suffix",
        ))
    return out


def _match_digits_lakh(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=float(m.group(1)) * 100000.0,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="digits_lakh",
        )
        for m in _DIGITS_LAKH_RE.finditer(text)
    ]


def _match_digits_hazaar(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=float(m.group(1)) * 1000.0,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="digits_hazaar",
        )
        for m in _DIGITS_HAZAAR_RE.finditer(text)
    ]


def _match_digits_sau(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=float(m.group(1)) * 100.0,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="digits_sau",
        )
        for m in _DIGITS_SAU_RE.finditer(text)
    ]


def _to_float(numstr: str) -> float:
    return float(numstr.replace(",", ""))


def _match_currency_prefix(text: str) -> list[AmountCandidate]:
    out = []
    for m in _CURRENCY_PREFIX_RE.finditer(text):
        sym = m.group("sym").lower()
        currency: CurrencyHint = "USD" if sym == "$" else "INR"
        out.append(AmountCandidate(
            value=_to_float(m.group("num")),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="currency_symbol",
            currency_hint=currency,
        ))
    return out


def _match_slash_dash(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=_to_float(m.group(1)),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="slash_dash",
            currency_hint="INR",
        )
        for m in _SLASH_DASH_RE.finditer(text)
    ]


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    """True if the half-open span [s, e) intersects any (cs, ce) in claimed."""
    s, e = span
    return any(not (e <= cs or s >= ce) for cs, ce in claimed)


def _blocked_spans(text: str) -> list[tuple[int, int]]:
    """Pre-compute time/date pattern spans so any digit overlapping them is rejected."""
    spans: list[tuple[int, int]] = []
    for rx in (_TIME_RE, _DATE_RE):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _preceded_by_keyword(text: str, start: int) -> bool:
    lookback_start = max(0, start - _KEYWORD_WINDOW_CHARS)
    snippet = text[lookback_start:start].lower()
    return any(kw in snippet for kw in _REJECT_KEYWORDS)


def _match_plain_digits(
    text: str, blocked: list[tuple[int, int]],
) -> list[AmountCandidate]:
    out = []
    for m in _PLAIN_DIGITS_RE.finditer(text):
        raw = m.group(0)
        digits_only = raw.replace(",", "")
        if len(digits_only) > _MAX_PLAIN_DIGIT_LEN:
            continue
        if _overlaps((m.start(), m.end()), blocked):
            continue
        if _preceded_by_keyword(text, m.start()):
            continue
        out.append(AmountCandidate(
            value=_to_float(raw),
            raw=raw, span=(m.start(), m.end()),
            status="active", source="digits",
        ))
    return out


def _match_number_word_en(text: str) -> list[AmountCandidate]:
    out: list[AmountCandidate] = []
    # Compound FIRST so "twenty five hundred" claims its span before the
    # simple matcher could grab "five hundred" inside it.
    for m in _EN_COMPOUND_RE.finditer(text):
        tens = _TENS[m.group(1).lower()]
        ones = _EN_NUMERAL[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float((tens + ones) * 100),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))
    for m in _EN_SIMPLE_RE.finditer(text):
        num = _EN_NUMERAL[m.group(1).lower()]
        unit = _EN_UNIT_WORD[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float(num * unit),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))
    return out


def _match_number_word_hi(text: str) -> list[AmountCandidate]:
    out: list[AmountCandidate] = []
    for m in _HI_UNIT_RE.finditer(text):
        num = _HI_NUMERAL[m.group(1).lower()]
        unit = _HI_UNIT_WORD[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float(num * unit),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_hi",
        ))
    return out


# ---------------------------------------------------------------------------
# Supersession pass  (§3.6)
# ---------------------------------------------------------------------------

# Multi-token compound markers MUST be matched before bare-token forms
# to avoid spurious bare-'no' matches inside 'wait no'.
_COMPOUND_MARKER_PHRASES = (
    "wait no", "wait actually", "wait sorry", "wait correction",
    "wait,", "i mean", "scratch that", "umm no", "uhh no",
)
_BARE_MARKER_WORDS = (
    "actually", "nope", "nahi", "nahin", "galat", "correction:",
)
_BARE_NO = re.compile(r"\bno\b", re.IGNORECASE)


def _find_correction_marker_spans(text: str) -> list[tuple[int, int]]:
    """Return spans where a correction marker appears, in input order."""
    spans: list[tuple[int, int]] = []
    lower = text.lower()
    # Compound phrases first
    for phrase in _COMPOUND_MARKER_PHRASES:
        start = 0
        while True:
            i = lower.find(phrase, start)
            if i == -1:
                break
            spans.append((i, i + len(phrase)))
            start = i + len(phrase)
    # Bare markers — \b boundary
    for word in _BARE_MARKER_WORDS:
        for m in re.finditer(rf"\b{re.escape(word)}\b", lower):
            spans.append((m.start(), m.end()))
    # Bare 'no' — but only if NOT already inside a compound marker span.
    for m in _BARE_NO.finditer(lower):
        s, e = m.start(), m.end()
        if any(cs <= s and e <= ce for cs, ce in spans):
            continue
        spans.append((s, e))

    spans.sort()
    return spans


_TOKEN_RE = re.compile(r"\S+")
_WINDOW_TOKENS = 6


def _token_index_at_char(text: str, char_pos: int) -> int:
    """Return the index of the first token whose start >= char_pos.

    Used to measure 'distance in tokens' between a candidate and a marker.
    """
    idx = 0
    for i, m in enumerate(_TOKEN_RE.finditer(text)):
        if m.start() >= char_pos:
            return i
        idx = i + 1
    return idx


def _apply_supersession(
    text: str, candidates: list[AmountCandidate],
) -> list[AmountCandidate]:
    if len(candidates) < 2:
        return candidates
    marker_spans = _find_correction_marker_spans(text)
    if not marker_spans:
        return candidates

    # Build token-index lookups.
    cand_token_idx = [_token_index_at_char(text, c.span[0]) for c in candidates]
    marker_token_idx = [_token_index_at_char(text, s) for s, _ in marker_spans]

    superseded: set[int] = set()
    for mtok in marker_token_idx:
        # Find nearest active candidate strictly before this marker, within window.
        prev_i = -1
        for i, ctok in enumerate(cand_token_idx):
            if ctok < mtok and (mtok - ctok) <= _WINDOW_TOKENS and i not in superseded:
                prev_i = i  # keep the latest qualifying one
        if prev_i == -1:
            continue
        # Find a replacement candidate AFTER the marker within window.
        next_i = -1
        for j, ctok in enumerate(cand_token_idx):
            if ctok > mtok and (ctok - mtok) <= _WINDOW_TOKENS and j not in superseded:
                next_i = j
                break
        if next_i == -1:
            continue
        superseded.add(prev_i)

    return [
        dataclasses.replace(c, status="superseded") if i in superseded else c
        for i, c in enumerate(candidates)
    ]


def parse_amounts(text: str) -> list[AmountCandidate]:
    """Return amount candidates in input order. Spans never overlap.

    Implemented incrementally — see subsequent tasks for each priority
    band of matchers.
    """
    if not text:
        return []
    claimed: list[tuple[int, int]] = []
    blocked = _blocked_spans(text)
    out: list[AmountCandidate] = []

    higher_priority_bands = (
        _match_decimal_k, _match_int_k,
        _match_digits_lakh, _match_digits_hazaar, _match_digits_sau,
        _match_currency_prefix, _match_slash_dash,
    )
    for band in higher_priority_bands:
        for cand in band(text):
            if _overlaps(cand.span, claimed):
                continue
            out.append(cand)
            claimed.append(cand.span)

    for cand in _match_plain_digits(text, blocked):
        if _overlaps(cand.span, claimed):
            continue
        out.append(cand)
        claimed.append(cand.span)

    # English / Hindi number-word + unit — lowest priority. Compound and
    # simple English matchers can produce overlapping spans, so we rely
    # on `claimed` to keep only the first hit.
    for band in (_match_number_word_en, _match_number_word_hi):
        for cand in band(text):
            if _overlaps(cand.span, claimed):
                continue
            out.append(cand)
            claimed.append(cand.span)

    out.sort(key=lambda c: c.span[0])
    out = _apply_supersession(text, out)
    return out
