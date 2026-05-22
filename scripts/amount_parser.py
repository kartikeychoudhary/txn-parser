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
    "bare_unit_word",
    "numeral_currency",
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

_PLAIN_DIGITS_RE = re.compile(
    r"\b("
    # Comma-grouped: US (\d{1,3},\d{3}+) OR Indian (\d{1,3},\d{2,3}+, irregular)
    # with optional decimal fraction.
    r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?"
    r"|"
    # Plain integer with optional decimal fraction.
    r"\d+(?:\.\d+)?"
    r")\b"
)

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
    # ── 1–10  (+ common transliteration variants from voice/chat) ──────────
    "ek": 1,    "aik": 1,   "ik": 1,     "eik": 1,
    "do": 2,    "doh": 2,
    "teen": 3,  "tin": 3,   "tiin": 3,
    "char": 4,  "chaar": 4,
    "paanch": 5, "panch": 5,
    "chhe": 6,  "che": 6,   "chhah": 6,  "chheh": 6,  "cheh": 6,  "chah": 6,  "chh": 6,
    "saat": 7,  "sat": 7,
    "aath": 8,  "aat": 8,   "ath": 8,
    "nau": 9,   "nao": 9,
    "das": 10,  "duss": 10, "dus": 10,
    # ── 11–19 ──────────────────────────────────────────────────────────────
    "gyarah": 11,  "gyara": 11,   "gyarha": 11,   "gyaarah": 11,  "gyaara": 11,
    "barah": 12,   "bara": 12,    "baara": 12,    "barha": 12,    "baarah": 12,
    "terah": 13,   "tera": 13,    "terha": 13,    "teraah": 13,
    "chaudah": 14, "chauda": 14,  "chaudha": 14,  "chaudaah": 14, "chaudaha": 14,
    "pandrah": 15, "pandra": 15,  "paandrah": 15, "paandra": 15,  "pandraah": 15,
    "solah": 16,   "sola": 16,    "solaah": 16,
    "satrah": 17,  "satra": 17,   "sattrah": 17,  "saatrah": 17,
    "atharah": 18, "athra": 18,   "athara": 18,   "aatharah": 18, "athaarah": 18, "attharah": 18,
    "unnees": 19,  "unnis": 19,   "unees": 19,    "uneess": 19,   "unniss": 19,
    # ── tens 20–90  (+ variants) ────────────────────────────────────────────
    "bees": 20,    "bis": 20,     "bais": 20,
    "tees": 30,    "tis": 30,
    "chalis": 40,  "chalees": 40, "chaalees": 40, "chaalis": 40, "chaliis": 40,
    "pachaas": 50, "pachas": 50,  "pachhas": 50,  "pacchas": 50,
    "saath": 60,   "sath": 60,    "saatth": 60,
    "sattar": 70,  "satar": 70,
    "assi": 80,    "aassi": 80,
    "nabbe": 90,   "navve": 90,   "nabbey": 90,
    # ── 21–29 ──────────────────────────────────────────────────────────────
    "ikees": 21,     "ikkees": 21,    "ikkis": 21,
    "baaees": 22,    "baees": 22,     "baais": 22,
    "teis": 23,      "teyees": 23,
    "chaubees": 24,  "chaubis": 24,
    "pachchees": 25, "pachees": 25,   "pachchis": 25, "pacchis": 25, "pachis": 25,
    "chhabbees": 26, "chhabbis": 26,
    "sattaees": 27,  "sataees": 27,   "sattais": 27,
    "atthaees": 28,  "athaees": 28,   "atthais": 28,
    "untees": 29,    "unatees": 29,
    # ── 31–39 ──────────────────────────────────────────────────────────────
    "ikattees": 31,  "ikatees": 31,
    "battees": 32,   "batees": 32,    "battis": 32,
    "taintees": 33,  "tentees": 33,
    "chautees": 34,
    "paintees": 35,  "panchtees": 35, "paintis": 35,
    "chhattees": 36, "chhatees": 36,  "chhattis": 36, "chattis": 36,
    "saintees": 37,  "saittees": 37,
    "artees": 38,
    "untalees": 39,  "unchalees": 39,
    # ── 41–49 ──────────────────────────────────────────────────────────────
    "iktaalees": 41, "iktalees": 41,
    "bayalees": 42,  "byalees": 42,
    "tiyalees": 43,  "titalees": 43,
    "chavaalees": 44,
    "paintaalees": 45, "panchtaalees": 45,
    "chhiyaalees": 46, "chhiyalees": 46,
    "saintaalees": 47, "saitaalees": 47,
    "artaalees": 48,
    "unchaas": 49,   "unpachaas": 49, "unchase": 49,
    # ── 51–59 ──────────────────────────────────────────────────────────────
    "ikyaavan": 51,  "ikkyavan": 51,
    "baavan": 52,    "bavan": 52,     "bawan": 52,
    "tirpan": 53,    "tripan": 53,
    "chauvan": 54,
    "pachpan": 55,
    "chhappan": 56,  "chappan": 56,
    "sattavan": 57,  "satavan": 57,
    "athavan": 58,
    "unsath": 59,    "unsaath": 59,
    # ── 61–69 ──────────────────────────────────────────────────────────────
    "iksath": 61,    "eksath": 61,
    "barsath": 62,   "baasath": 62,
    "tirsath": 63,   "tresath": 63,
    "chausath": 64,
    "painsath": 65,
    "chhiyasath": 66,
    "sarsath": 67,   "sadsath": 67,
    "arsath": 68,    "adsath": 68,
    "unsattar": 69,  "unhattar": 69,
    # ── 71–79 ──────────────────────────────────────────────────────────────
    "ikhattar": 71,
    "bahattar": 72,  "bahttar": 72,
    "tihattar": 73,  "tehattar": 73,
    "chauhattar": 74,
    "pachhattar": 75, "pachattar": 75,
    "chhihattar": 76,
    "sathattar": 77,
    "athattar": 78,  "atthattar": 78,
    "unyaasi": 79,   "unnayasi": 79,
    # ── 81–89 ──────────────────────────────────────────────────────────────
    "ikyaasi": 81,   "ikkyaasi": 81,
    "bayasi": 82,    "byaasi": 82,
    "tirasi": 83,
    "chaurasi": 84,
    "pachasi": 85,   "pachaasi": 85,
    "chhiyasi": 86,  "chhiyaasi": 86,
    "sattasi": 87,   "sataasi": 87,
    "athasi": 88,    "athaasi": 88,
    "unabbe": 89,    "unnabbe": 89,
    # ── 91–99 ──────────────────────────────────────────────────────────────
    "ikyaanave": 91, "ikyaanbbe": 91,
    "baanave": 92,
    "tiranave": 93,  "tiranve": 93,
    "chauraanave": 94, "chauranave": 94,
    "pachaanave": 95,  "pachanave": 95,
    "chhiyanave": 96,  "chhiyanve": 96,
    "sataanave": 97,   "satanave": 97,
    "atthaanave": 98,  "athanave": 98,
    "ninyaanave": 99,  "ninyanave": 99,  "ninyanve": 99,  "nintyanve": 99,
}

_EN_UNIT_WORD = {"hundred": 100, "thousand": 1000, "lakh": 100000}
# Both `hazaar` and `hazar` are accepted transliterations. Crore added for
# high-value transaction inputs ("paanch karod ki property").
_HI_UNIT_WORD = {
    "sau": 100, "hazaar": 1000, "hazar": 1000,
    "lakh": 100000, "karod": 10000000, "crore": 10000000,
}

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

# ---------------------------------------------------------------------------
# Multi-part English compound numbers  (highest priority among word matchers)
# Handles: "one thousand six hundred", "two lakh fifty thousand", etc.
# Pattern: [N lakh] [M thousand] [P hundred]  — at least one part required,
# but captures the entire span so simpler matchers don't double-count.
# ---------------------------------------------------------------------------
_EN_NUM_WORDS = "|".join(sorted(_EN_NUMERAL, key=len, reverse=True))  # longest first
_EN_TENS_WORDS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_EN_ONES_WORDS = "one|two|three|four|five|six|seven|eight|nine"
# A 1-or-2 word numeral (e.g. "six", "twenty six", "nineteen")
# Tens-ones may be separated by whitespace OR a hyphen ("twenty-five", "ninety-nine").
_EN_1_2_NUM = (
    r"(?:(?:" + _EN_TENS_WORDS + r")[\s\-]+(?:" + _EN_ONES_WORDS + r")|" + _EN_NUM_WORDS + r")"
)
_EN_MULTIPART_RE = re.compile(
    r"\b"
    # Optional lakh part
    r"(?:(" + _EN_1_2_NUM + r")\s+lakh\s*)?"
    # Optional thousand part
    r"(?:(" + _EN_1_2_NUM + r")\s+thousand\s*)?"
    # Optional hundred part
    r"(?:(" + _EN_1_2_NUM + r")\s+hundred\s*)?"
    # Optional trailing bare numeral remainder — must NOT be followed by
    # another unit word (else it belongs to a higher tier). Allows an
    # optional "and" connector ("eight hundred and fifty").
    r"(?:(?:and\s+)?(" + _EN_1_2_NUM + r")\b(?!\s*(?:hundred|thousand|lakh)\b))?"
    r"\b",
    re.IGNORECASE,
)

# "Dropped-hundred" pattern: a leading ones/ten digit followed by a tens
# word implies the speaker omitted "hundred". Colloquial Indian/American
# English elision:
#   "one eighty"          = 1*100 + 80         = 180
#   "seven fifty"         = 7*100 + 50         = 750
#   "one ninety nine"     = 1*100 + 90 + 9     = 199
#   "ten fifty"           = 10*100 + 50        = 1050
# Restricted to ones (one..nine) and "ten" as the leading word so we don't
# false-match teens ("eleven eighty" = year 1180, unlikely in expense talk).
_EN_DROPPED_HUNDRED_LEAD = "one|two|three|four|five|six|seven|eight|nine|ten"
_EN_DROPPED_HUNDRED_RE = re.compile(
    r"\b(" + _EN_DROPPED_HUNDRED_LEAD + r")\s+"
    r"(" + _EN_TENS_WORDS + r")"
    r"(?:[\s\-]+(" + _EN_ONES_WORDS + r"))?"
    r"\b(?!\s*(?:hundred|thousand|lakh|sau|hazaar|hazar|crore|karod)\b)",
    re.IGNORECASE,
)

# "Dropped-thousand" pattern: a teens word (11-19) followed by an N-hundred
# phrase implies the speaker omitted "thousand". e.g.
#   "fifteen five hundred"      = 15 * 1000 +  5 * 100         = 15500
#   "twelve five hundred fifty" = 12 * 1000 +  5 * 100 +  50   = 12550
# Safe because teens can't form a tens+ones compound ("fifteen five"
# isn't a number on its own — unlike "twenty five" = 25).
_EN_TEENS_NUMERAL_WORDS = (
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_EN_TEENS_RE_PART = "|".join(_EN_TEENS_NUMERAL_WORDS)
_EN_DROPPED_THOUSAND_RE = re.compile(
    r"\b(" + _EN_TEENS_RE_PART + r")\s+"
    r"(" + _EN_ONES_WORDS + r")\s+hundred"
    r"(?:\s+(?:and\s+)?(" + _EN_1_2_NUM + r")\b(?!\s*(?:hundred|thousand|lakh)\b))?"
    r"\b",
    re.IGNORECASE,
)

_HI_UNIT_RE = re.compile(
    r"\b(" + "|".join(sorted(_HI_NUMERAL, key=len, reverse=True))
    + r")\s+(" + "|".join(sorted(_HI_UNIT_WORD, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Multi-part Hindi compound number matcher
# Handles: "ek hazaar paanch sau" = 1500,
#          "do lakh paanch hazaar" = 205000,
#          "teen sau bees" = 320  (sau + bare numeral remainder), etc.
# Pattern: [N crore/karod] [M lakh] [P hazaar/hazar] [Q sau] [R bare]  —
# at least 2 parts required so single-part phrases fall through to _HI_UNIT_RE.
# ---------------------------------------------------------------------------
_HI_NUM = "|".join(sorted(_HI_NUMERAL, key=len, reverse=True))  # longest first
_HI_UNITS_ORDERED = ["karod", "crore", "lakh", "hazaar", "hazar", "sau"]

# Build a regex that optionally captures each tier in descending order,
# followed by an optional bare numeral remainder (e.g. "bees" in "teen sau bees").
# Groups 1-4: crore, lakh, hazaar, sau (each paired with their unit word).
# Group 5:   trailing bare numeral with NO unit (additive remainder ≤ 999).
def _build_hi_multipart_re() -> re.Pattern:
    num_alt = "|".join(sorted(_HI_NUMERAL, key=len, reverse=True))
    # Tiers in descending order: crore, lakh, thousand, hundred
    tiers = [
        ("karod|crore",     "crore"),
        ("lakh",            "lakh"),
        ("hazaar|hazar",    "thousand"),
        ("sau",             "hundred"),
    ]
    parts = []
    for unit_re, _ in tiers:
        parts.append(
            r"(?:(" + num_alt + r")\s+(?:" + unit_re + r")\s*)?"
        )
    # Optional trailing bare numeral remainder (no unit word following it).
    # Uses a negative lookahead to ensure it is NOT immediately followed by a
    # HI unit word, so "teen sau" is not consumed as "teen" (sau-less remainder).
    # The trailing \b prevents a short numeral (e.g. "ik") from matching INSIDE
    # a longer non-numeral word (e.g. "ikyaanabe"), which would otherwise add
    # spurious value to the multipart total.
    hi_units_alt = "|".join(sorted(_HI_UNIT_WORD, key=len, reverse=True))
    parts.append(
        r"(?:(" + num_alt + r")\b(?!\s*(?:" + hi_units_alt + r")\b))?"
    )
    pattern = r"\b" + "".join(parts)
    return re.compile(pattern, re.IGNORECASE)

_HI_MULTIPART_RE = _build_hi_multipart_re()

# Bare unit word (e.g. "sau rupay ka chai", "hundred bucks", "lakh ki car").
# Lowest-priority matcher — quantified phrases like "ek sau" / "five hundred"
# / "100 sau" claim their span first via higher-priority bands, so the bare
# matcher only fires when no numeral precedes the unit.
_BARE_UNIT_WORDS = {**_HI_UNIT_WORD, **_EN_UNIT_WORD}
_BARE_UNIT_RE = re.compile(
    r"\b(" + "|".join(sorted(_BARE_UNIT_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Bare numeral followed by a currency word, e.g. "fifty rupees",
# "pachas rupay", "twenty bucks". Without the currency suffix a bare
# numeral is too ambiguous to capture (could be a quantity, time, etc.),
# but the currency word makes intent unambiguous.
_CURRENCY_WORDS = (
    "rupees", "rupee", "rupay", "rupaya", "rupaye", "rupye",
    "rs", "inr",
    "bucks", "dollars", "dollar", "usd",
    "paisa", "paise", "paisay",
)
_CURRENCY_WORD_ALT = "|".join(sorted(_CURRENCY_WORDS, key=len, reverse=True))
# Reuse _EN_1_2_NUM (twenty five | fifteen | etc.) and the full Hindi numeral set.
_NUMERAL_CURRENCY_RE = re.compile(
    r"\b("
    + _EN_1_2_NUM + r"|"
    + "|".join(sorted(_HI_NUMERAL, key=len, reverse=True))
    + r")\s+(?:" + _CURRENCY_WORD_ALT + r")\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_REJECT_KEYWORDS = (
    "phone number", "mobile number", "otp", "pin",
    "account number", "account no", "upi id", "order id",
)
# Compiled word-boundary forms so "pin" doesn't match "Shop|ping|" tail. The
# multi-word phrases use \s+ to tolerate any whitespace between the tokens.
_REJECT_KEYWORD_PATTERNS = tuple(
    re.compile(r"\b" + r"\s+".join(re.escape(t) for t in kw.split()) + r"\b",
               re.IGNORECASE)
    for kw in _REJECT_KEYWORDS
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
    snippet = text[lookback_start:start]
    return any(rx.search(snippet) for rx in _REJECT_KEYWORD_PATTERNS)


def _match_plain_digits(
    text: str, blocked: list[tuple[int, int]],
) -> list[AmountCandidate]:
    out = []
    for m in _PLAIN_DIGITS_RE.finditer(text):
        raw = m.group(0)
        # Reject by integer-part length only (so "12345.67" stays OK while
        # "9876543210" / "9876543210.50" are rejected as phone/OTP-like).
        integer_part = raw.split(".")[0].replace(",", "")
        if len(integer_part) > _MAX_PLAIN_DIGIT_LEN:
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


def _parse_en_1_2_num(word: str) -> int:
    """Parse a 1-or-2 word English numeral like 'twenty six', 'twenty-six', or 'nineteen'."""
    parts = word.lower().replace("-", " ").split()
    if len(parts) == 2:
        return _TENS[parts[0]] + _EN_NUMERAL[parts[1]]
    return _EN_NUMERAL[parts[0]]


def _match_number_word_en(text: str) -> list[AmountCandidate]:
    out: list[AmountCandidate] = []

    # Dropped-thousand FIRST: "fifteen five hundred" = 15500.
    # Wider span than the simple/compound matchers below, so its claim
    # suppresses the inner "five hundred" they'd otherwise emit.
    for m in _EN_DROPPED_THOUSAND_RE.finditer(text):
        teens = _EN_NUMERAL[m.group(1).lower()]
        ones  = _EN_NUMERAL[m.group(2).lower()]
        rem   = m.group(3)
        total = float(teens * 1000 + ones * 100)
        if rem:
            total += _parse_en_1_2_num(rem.strip())
        out.append(AmountCandidate(
            value=total,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))

    # Multi-part FIRST: "one thousand six hundred", "two lakh fifty thousand", etc.
    # The regex is optional-group-heavy so we get many zero-width matches;
    # only emit a candidate when at least one group is non-None.
    for m in _EN_MULTIPART_RE.finditer(text):
        lakh_str = m.group(1)
        thou_str = m.group(2)
        hund_str = m.group(3)
        rem_str  = m.group(4)
        if not any((lakh_str, thou_str, hund_str, rem_str)):
            continue
        # Emit when at least one real tier (lakh/thousand/hundred) is set
        # — this lets multi-word numerals like "twenty five thousand"
        # be captured here (with span over the whole phrase) so the
        # narrower simple matcher's "five thousand" is suppressed by
        # the claimed-span check. A remainder-only match (just "fifty")
        # is left for the numeral_currency / simpler matchers.
        tier_filled = sum(1 for x in (lakh_str, thou_str, hund_str) if x)
        if tier_filled == 0:
            continue
        total = 0.0
        if lakh_str:
            total += _parse_en_1_2_num(lakh_str.strip()) * 100_000
        if thou_str:
            total += _parse_en_1_2_num(thou_str.strip()) * 1_000
        if hund_str:
            total += _parse_en_1_2_num(hund_str.strip()) * 100
        if rem_str:
            total += _parse_en_1_2_num(rem_str.strip())
        out.append(AmountCandidate(
            value=total,
            raw=m.group(0).strip(), span=(m.start(), m.start() + len(m.group(0).rstrip())),
            status="active", source="number_word_en",
        ))

    # Compound "twenty five hundred" = (20 + 5) * 100 = 2500
    for m in _EN_COMPOUND_RE.finditer(text):
        tens = _TENS[m.group(1).lower()]
        ones = _EN_NUMERAL[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float((tens + ones) * 100),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))
    # Simple "one thousand", "six hundred"
    for m in _EN_SIMPLE_RE.finditer(text):
        num = _EN_NUMERAL[m.group(1).lower()]
        unit = _EN_UNIT_WORD[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float(num * unit),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))

    # Dropped-hundred colloquial: "one eighty" = 180, "seven fifty" = 750.
    # Runs AFTER simple/compound so phrases like "five hundred" (where
    # "hundred" is explicit) are claimed first; the trailing negative
    # lookahead already prevents this matcher from biting into them.
    for m in _EN_DROPPED_HUNDRED_RE.finditer(text):
        lead = _EN_NUMERAL[m.group(1).lower()]
        tens = _TENS[m.group(2).lower()]
        ones_str = m.group(3)
        ones = _EN_NUMERAL[ones_str.lower()] if ones_str else 0
        out.append(AmountCandidate(
            value=float(lead * 100 + tens + ones),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))
    return out


def _match_number_word_hi(text: str) -> list[AmountCandidate]:
    out: list[AmountCandidate] = []

    # Multi-part FIRST: "ek hazaar paanch sau" = 1500,
    #                   "teen sau bees" = 320 (sau + bare remainder)
    # _HI_MULTIPART_RE is optional-group-heavy so many zero-width matches arise;
    # only emit when >= 2 parts are non-None (tier groups 1-4 + optional bare group 5).
    _TIER_MULTIPLIERS = [10_000_000, 100_000, 1_000, 100]  # crore, lakh, hazaar, sau
    for m in _HI_MULTIPART_RE.finditer(text):
        tier_groups = [m.group(i + 1) for i in range(4)]   # groups 1-4: unit tiers
        bare_group  = m.group(5)                            # group 5: bare remainder
        all_parts   = [g for g in tier_groups if g] + ([bare_group] if bare_group else [])
        if len(all_parts) < 2:
            continue
        total = 0.0
        for i, g in enumerate(tier_groups):
            if g:
                total += _HI_NUMERAL[g.strip().lower()] * _TIER_MULTIPLIERS[i]
        if bare_group:
            total += _HI_NUMERAL[bare_group.strip().lower()]   # additive remainder
        raw = m.group(0).rstrip()
        span = (m.start(), m.start() + len(raw))
        out.append(AmountCandidate(
            value=total,
            raw=raw, span=span,
            status="active", source="number_word_hi",
        ))

    # Simple single-unit: "ek hazaar" = 1000, "paanch sau" = 500
    for m in _HI_UNIT_RE.finditer(text):
        num = _HI_NUMERAL[m.group(1).lower()]
        unit = _HI_UNIT_WORD[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float(num * unit),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_hi",
        ))
    return out


def _match_numeral_currency(text: str) -> list[AmountCandidate]:
    """Match a bare 1-or-2 word numeral followed by a currency word
    (e.g. "fifty rupees", "pachas rupay", "twenty bucks").

    Lower priority than the structured matchers — quantified phrases like
    "fifty thousand rupees" are claimed first by the higher-priority
    bands and this matcher only fires for the standalone case.
    """
    out: list[AmountCandidate] = []
    for m in _NUMERAL_CURRENCY_RE.finditer(text):
        word = m.group(1).lower()
        # Hyphenated compounds like "ninety-five" still split into tens+ones.
        parts = word.replace("-", " ").split()
        if len(parts) == 2:
            value = _TENS[parts[0]] + _EN_NUMERAL[parts[1]]
        else:
            value = _EN_NUMERAL.get(parts[0]) or _HI_NUMERAL.get(parts[0])
            if value is None:
                continue
        tail = m.group(0).lower().rsplit(None, 1)[-1]
        currency: CurrencyHint = "USD" if tail in (
            "bucks", "dollars", "dollar", "usd",
        ) else "INR"
        out.append(AmountCandidate(
            value=float(value),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="numeral_currency",
            currency_hint=currency,
        ))
    return out


def _match_bare_unit(text: str) -> list[AmountCandidate]:
    """Match a bare unit word (sau, hazaar, lakh, karod/crore, hundred,
    thousand) with no numeric quantifier. Value = the unit itself.

    Relies on the `claimed`-span system in parse_amounts: quantified
    phrases captured by higher-priority bands consume their span first,
    so bare-unit matches inside them are rejected before emission.
    """
    out = []
    for m in _BARE_UNIT_RE.finditer(text):
        word = m.group(1).lower()
        out.append(AmountCandidate(
            value=float(_BARE_UNIT_WORDS[word]),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="bare_unit_word",
        ))
    return out


# ---------------------------------------------------------------------------
# Indian-additive merge pass
#
# Same-language compound numbers (e.g. "one lakh fifty thousand",
# "ek hazaar paanch sau") are already captured as a single candidate by
# _EN_MULTIPART_RE / _HI_MULTIPART_RE. But CROSS-language and
# digit/word mixed forms (very common in code-switched Indian speech)
# slip through as multiple per-tier candidates that the validator then
# rejects:
#
#   "1 lakh 50 thousand"        -> digits_lakh(100000) + digits_hazaar(50000)
#   "ek lakh five thousand"     -> number_word_hi(100000) + number_word_en(5000)
#   "do hazaar five hundred"    -> number_word_hi(2000)  + number_word_en(500)
#
# Speakers almost always mean the SUM in these forms. We append a synthetic
# SUM candidate (without removing the pieces) so the validator can accept
# either interpretation.
# ---------------------------------------------------------------------------

_UNIT_WORD_SOURCES = frozenset({
    "decimal_k", "k_suffix",
    "digits_lakh", "digits_hazaar", "digits_sau",
    "number_word_en", "number_word_hi",
})

# Magnitude rank: lakh > hazaar > sau > bare. Used to confirm a descending
# additive pattern (the colloquial Indian-number form).
_TIER_BY_SOURCE = {
    "digits_lakh": 100_000,
    "digits_hazaar": 1_000,
    "digits_sau": 100,
    "k_suffix": 1_000,
    "decimal_k": 1_000,
}


def _candidate_tier(c: "AmountCandidate") -> int:
    """Approximate magnitude tier of a candidate (powers of 10).

    Used as a coarse 'is this number bigger than that one?' check for the
    descending-additive heuristic.
    """
    if c.source in _TIER_BY_SOURCE:
        return _TIER_BY_SOURCE[c.source]
    # For multi-tier number_word matchers, fall back to the value itself.
    if c.value >= 100_000:
        return 100_000
    if c.value >= 1_000:
        return 1_000
    if c.value >= 100:
        return 100
    return 1


_ADDITIVE_GAP_RE = re.compile(r"^\s*(?:and|aur|aour|or)?\s*$", re.IGNORECASE)

# Bare English tens (optionally followed by an ones word) that we scoop onto
# a trailing position of a unit-word group whose head was at the sau/hundred
# tier. Captures the colloquial "char sau fifty" = 450, "ek sau twenty" = 120.
_TRAILING_TENS_RE = re.compile(
    r"^\s+(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[\s\-]+(one|two|three|four|five|six|seven|eight|nine))?"
    r"\b(?!\s*(?:hundred|thousand|lakh|sau|hazaar|hazar|crore|karod)\b)",
    re.IGNORECASE,
)


def _scoop_trailing_tens(text: str, after_pos: int) -> tuple[int, int] | None:
    """If a bare English tens (e.g. 'fifty', 'twenty five') sits immediately
    after `after_pos`, return (value, new_end). Else None.
    """
    m = _TRAILING_TENS_RE.match(text[after_pos:])
    if not m:
        return None
    tens = _TENS[m.group(1).lower()]
    ones = _EN_NUMERAL[m.group(2).lower()] if m.group(2) else 0
    return tens + ones, after_pos + m.end()


def _is_additive_continuation(
    text: str, prev: "AmountCandidate", cur: "AmountCandidate",
) -> bool:
    """True if cur immediately follows prev in an Indian-additive pattern."""
    if prev.source not in _UNIT_WORD_SOURCES or cur.source not in _UNIT_WORD_SOURCES:
        return False
    # Gap between them is at most a connector word.
    gap = text[prev.span[1]:cur.span[0]]
    if not _ADDITIVE_GAP_RE.match(gap):
        return False
    # Descending magnitude: cur's tier must be strictly smaller than prev's.
    if _candidate_tier(cur) >= _candidate_tier(prev):
        return False
    # cur's value must fit inside prev's tier (e.g. thousand-piece < 100,000).
    if cur.value >= _candidate_tier(prev):
        return False
    return True


def _merge_indian_additive(
    text: str, candidates: list[AmountCandidate],
) -> list[AmountCandidate]:
    """Append synthetic SUM candidates for adjacent unit-word amounts that
    form an Indian-additive compound. Originals are kept (still active) so
    the validator accepts either the merged total or the individual parts.

    Also handles the "trailing tens" colloquialism where the final piece is
    a bare English tens word with no unit: 'char sau fifty' = 450.
    """
    if not candidates:
        return candidates
    by_pos = sorted(candidates, key=lambda c: c.span[0])
    extras: list[AmountCandidate] = []

    i = 0
    while i < len(by_pos):
        group = [by_pos[i]]
        j = i + 1
        while j < len(by_pos) and _is_additive_continuation(text, group[-1], by_pos[j]):
            group.append(by_pos[j])
            j += 1
        total = sum(c.value for c in group)
        end_pos = group[-1].span[1]
        # Scoop a trailing bare-tens word only when the last component is at
        # the sau/hundred tier (value < 1000) — that's the colloquial form.
        scoop_happened = False
        if (
            group[-1].source in _UNIT_WORD_SOURCES
            and group[-1].value < 1000
        ):
            scooped = _scoop_trailing_tens(text, end_pos)
            if scooped is not None:
                add_val, end_pos = scooped
                total += add_val
                scoop_happened = True
        if len(group) >= 2 or scoop_happened:
            span = (group[0].span[0], end_pos)
            raw = text[span[0]:span[1]]
            # Reuse the source label of the head; this is a coarse hint and
            # callers should look at the value, not the source, for routing.
            extras.append(AmountCandidate(
                value=float(total), raw=raw, span=span,
                status="active",
                source=group[0].source,
                currency_hint=group[0].currency_hint,
            ))
        i = j if len(group) >= 2 else i + 1

    return list(candidates) + extras


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

    # Bare numeral + currency word ("fifty rupees", "pachas rupay"). Runs
    # after structured matchers so phrases like "fifty thousand rupees"
    # are already claimed by higher-priority bands.
    for cand in _match_numeral_currency(text):
        if _overlaps(cand.span, claimed):
            continue
        out.append(cand)
        claimed.append(cand.span)

    # Bare unit word — runs last so any quantified phrase has already
    # claimed its span (and the bare unit within it).
    for cand in _match_bare_unit(text):
        if _overlaps(cand.span, claimed):
            continue
        out.append(cand)
        claimed.append(cand.span)

    out.sort(key=lambda c: c.span[0])
    out = _merge_indian_additive(text, out)
    out.sort(key=lambda c: c.span[0])
    out = _apply_supersession(text, out)
    return out
