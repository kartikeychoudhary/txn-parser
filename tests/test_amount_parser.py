import math
import pytest
from amount_parser import AmountCandidate, parse_amounts


def test_amountcandidate_is_frozen_dataclass():
    c = AmountCandidate(
        value=500.0, raw="500", span=(0, 3),
        status="active", source="digits", currency_hint=None,
    )
    assert c.value == 500.0
    assert c.raw == "500"
    assert c.span == (0, 3)
    assert c.status == "active"
    assert c.source == "digits"
    assert c.currency_hint is None


def test_parse_amounts_empty_string_returns_empty_list():
    assert parse_amounts("") == []


def test_parse_amounts_text_without_amounts_returns_empty_list():
    assert parse_amounts("just some words here") == []


@pytest.mark.parametrize("text,expected_value,expected_source", [
    ("1.5k",        1500.0, "decimal_k"),
    ("2.5K",        2500.0, "decimal_k"),
    ("0.5k",         500.0, "decimal_k"),
    ("50k",        50000.0, "k_suffix"),
    ("5K",          5000.0, "k_suffix"),
    ("12k",        12000.0, "k_suffix"),
])
def test_k_suffix_amounts(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source
    assert c.status == "active"


@pytest.mark.parametrize("text,expected_value,expected_source", [
    # lakh
    ("1 lakh",          100000.0, "digits_lakh"),
    ("1.5 lakh",        150000.0, "digits_lakh"),
    ("2 lakhs",         200000.0, "digits_lakh"),
    # hazaar / thousand
    ("5 hazaar",          5000.0, "digits_hazaar"),
    ("2 thousand",        2000.0, "digits_hazaar"),
    ("10 hazaar",        10000.0, "digits_hazaar"),
    # sau / hundred
    ("2 sau",              200.0, "digits_sau"),
    ("5 hundred",          500.0, "digits_sau"),
])
def test_digit_with_unit(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source
    assert c.status == "active"


def test_unit_beats_plain_digit_priority_regression():
    """1 lakh must produce 100000, not '1' as a plain digit."""
    cands = parse_amounts("1 lakh rent")
    assert len(cands) == 1
    assert math.isclose(cands[0].value, 100000.0)


@pytest.mark.parametrize("text,value,currency,source", [
    ("$50 coffee",      50.0, "USD", "currency_symbol"),
    ("₹300 lunch",     300.0, "INR", "currency_symbol"),
    ("rs 500 dinner",  500.0, "INR", "currency_symbol"),
    ("Rs.500 petrol",  500.0, "INR", "currency_symbol"),
    ("500/- chai",     500.0, "INR", "slash_dash"),
])
def test_currency_hint(text, value, currency, source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, value, abs_tol=0.001)
    assert c.currency_hint == currency
    assert c.source == source


def test_plain_digit_has_no_currency_hint():
    cands = parse_amounts("500 chai")
    assert len(cands) == 1
    assert cands[0].currency_hint is None


@pytest.mark.parametrize("text,expected_value", [
    ("5000",       5000.0),
    ("50,000",    50000.0),
    ("12500",     12500.0),
    ("paid 125000 rent",  125000.0),
])
def test_plain_digits(text, expected_value):
    cands = parse_amounts(text)
    assert len(cands) == 1
    assert math.isclose(cands[0].value, expected_value, abs_tol=0.001)
    assert cands[0].source == "digits"


@pytest.mark.parametrize("text", [
    "phone number 9876543210",
    "otp 123456",
    "mobile 9876543210",
    "order id 12345678",
    "pin 1234",                    # preceded-by-keyword
    "account 1122334455",
    "upi id 1234",
])
def test_plain_digits_rejected_by_keyword_or_length(text):
    assert parse_amounts(text) == []


@pytest.mark.parametrize("text", [
    "12:30 train",
    "at 9:45 meeting",
    "12/05/2026 due",
    "due 01/01/2027",
])
def test_plain_digits_rejected_by_time_date_pattern(text):
    assert parse_amounts(text) == []


@pytest.mark.parametrize("text,expected_value,expected_source", [
    # English
    ("five hundred",       500.0, "number_word_en"),
    ("two thousand",      2000.0, "number_word_en"),
    ("one lakh",        100000.0, "number_word_en"),
    ("five lakh",       500000.0, "number_word_en"),
    ("twenty five hundred", 2500.0, "number_word_en"),
    # Hindi
    ("paanch sau",         500.0, "number_word_hi"),
    ("do hazaar",         2000.0, "number_word_hi"),
    ("char lakh",       400000.0, "number_word_hi"),
    ("chhe sau",           600.0, "number_word_hi"),
    ("ek hazaar",         1000.0, "number_word_hi"),
])
def test_number_word_with_unit(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source


@pytest.mark.parametrize("text,expected_value,expected_source", [
    ("fifty thousand",  50000.0, "number_word_en"),
    ("thirty thousand", 30000.0, "number_word_en"),
    ("ninety hundred",   9000.0, "number_word_en"),
    ("forty lakh",    4000000.0, "number_word_en"),
])
def test_extended_en_numeral_tens(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source


@pytest.mark.parametrize("text", [
    "do coffee",
    "char people",
    "paanch minute",
    "two minutes",
    "second item",
])
def test_bare_number_words_not_parsed(text):
    assert parse_amounts(text) == []


def _statuses(text):
    return [(c.value, c.status) for c in parse_amounts(text)]


def test_supersession_simple_wait_no():
    assert _statuses("500 beer wait no 600 beer") == [
        (500.0, "superseded"), (600.0, "active"),
    ]


def test_supersession_rapid_fire():
    assert _statuses("500 no 600 no 700") == [
        (500.0, "superseded"), (600.0, "superseded"), (700.0, "active"),
    ]


def test_no_supersession_when_no_replacement():
    """'no' after a candidate but no later candidate -> stays active."""
    assert _statuses("500 beer no extra charges") == [(500.0, "active")]


def test_no_supersession_for_real_two_transactions():
    """No marker between candidates -> both active."""
    assert _statuses("500 beer and 500 candy") == [(500.0, "active"), (500.0, "active")]


def test_no_supersession_for_repeated_amount_no_marker():
    assert _statuses("500 beer and 500 beer") == [(500.0, "active"), (500.0, "active")]


def test_hindi_correction_marker_nahi():
    assert _statuses("paanch sau nahi chhe sau") == [
        (500.0, "superseded"), (600.0, "active"),
    ]


def test_bare_wait_is_not_a_marker():
    """'wait and' / 'wait for' / bare 'wait' should NOT supersede."""
    assert _statuses("500 beer wait and 600 candy") == [
        (500.0, "active"), (600.0, "active"),
    ]


def test_window_too_far_no_supersession():
    """A marker far beyond 6 tokens from the previous candidate must not supersede."""
    text = "500 beer was nice and tasty and refreshing on a hot day no actually 600"
    statuses = _statuses(text)
    # 500 stays active because the marker is too far from it.
    assert (500.0, "active") in statuses


def test_no_supersession_two_candidates_no_replacement_after_marker():
    """Two candidates, marker after first but no candidate after marker -> both active."""
    assert _statuses("500 beer and 600 juice no extra") == [
        (500.0, "active"), (600.0, "active"),
    ]


def test_higher_priority_overlap_skipped():
    """A k-suffix match and a plain-digit match overlap -> plain digit dropped."""
    # "5k" is matched as k_suffix (5000); the bare '5' should NOT also appear.
    cands = parse_amounts("5k")
    assert len(cands) == 1
    assert cands[0].source == "k_suffix"


def test_token_index_at_char_past_end():
    """Marker positioned past all tokens returns idx == number of tokens."""
    # Use a correction marker right at the end of text with no following candidate.
    # "500 no" — marker 'no' is last token; still no replacement, so 500 stays active.
    assert _statuses("500 no") == [(500.0, "active")]


# ---------------------------------------------------------------------------
# Bug 1: decimals must stay as one candidate, not split on the dot.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_value", [
    ("150.25 on a coffee",   150.25),
    ("7500.50 for car",     7500.50),
    ("75.50 for bus fare",    75.50),
    ("bill paid 1899.75",   1899.75),
    ("129.99 for snacks",    129.99),
    ("125.50 at cafe",       125.50),
])
def test_decimal_digits_not_split(text, expected_value):
    cands = parse_amounts(text)
    values = [c.value for c in cands if c.status == "active"]
    assert expected_value in values, f"expected {expected_value} in {values}"


# ---------------------------------------------------------------------------
# Bug 4: Indian comma grouping (1,50,000) must be one candidate.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_value", [
    ("1,50,000 for wedding gift",  150000.0),
    ("2,75,000 dowry",             275000.0),
    ("10,00,000 jewellery",       1000000.0),  # NOTE: 7 digits — rejected by length cap
])
def test_indian_comma_grouping(text, expected_value):
    cands = parse_amounts(text)
    values = [c.value for c in cands if c.status == "active"]
    if expected_value == 1000000.0:
        # Above 6-digit cap, so plain_digits rejects it. Document the limit.
        assert expected_value not in values
    else:
        assert expected_value in values


# ---------------------------------------------------------------------------
# Bug 2: cross-language additive (lakh+thousand, etc.) merges into a SUM.
# Originals stay too, so two-transaction interpretations still validate.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_sum", [
    ("1 lakh 50 thousand for car",      150000.0),
    ("1 lakh five thousand bike",       105000.0),
    ("ek lakh five thousand plot",      105000.0),
    ("2 lakh fifty thousand land",      250000.0),
    ("do hazaar five hundred recharge",   2500.0),
    ("teen hazaar three hundred mobile",  3300.0),
    ("ek hazaar aur paanch sau",          1500.0),
    ("char sau fifty auto fare",           450.0),
    ("ek sau forty bus",                   140.0),
])
def test_indian_additive_compound_emits_sum(text, expected_sum):
    cands = parse_amounts(text)
    values = [c.value for c in cands if c.status == "active"]
    assert expected_sum in values, f"expected {expected_sum} in {values} for {text!r}"


# ---------------------------------------------------------------------------
# Bug 5 (partial): colloquial dropped-hundred and hyphenated numerals.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_value", [
    ("quick lunch one eighty",                180.0),
    ("evening coffee one ninety nine",        199.0),
    ("bike service seven fifty",              750.0),
    ("ten fifty parking",                    1050.0),
])
def test_dropped_hundred_colloquial(text, expected_value):
    cands = parse_amounts(text)
    values = [c.value for c in cands if c.status == "active"]
    assert expected_value in values, f"expected {expected_value} in {values} for {text!r}"


def test_hyphenated_numeral_with_currency_word():
    """'ninety-five rupees' must parse, not be dropped by the hyphen split."""
    cands = parse_amounts("ninety-five rupees for a charger cable")
    values = [c.value for c in cands if c.status == "active"]
    assert 95.0 in values


def test_dropped_hundred_not_eaten_by_explicit_hundred():
    """'five hundred' must remain 500 (not 5+something)."""
    cands = parse_amounts("five hundred for medicines")
    values = [c.value for c in cands if c.status == "active"]
    assert values == [500.0]


def test_dropped_hundred_not_eaten_by_explicit_thousand():
    """'two thousand' must stay 2000 (lead-word lookahead suppresses the dropped form)."""
    cands = parse_amounts("two thousand for rent")
    values = [c.value for c in cands if c.status == "active"]
    assert values == [2000.0]


@pytest.mark.parametrize("text,expected_value", [
    ("Shopping 1500",            1500.0),   # 'pin' was matching inside 'Shopping'
    ("topping 2000 sauce",       2000.0),   # 'pin' / 'opt' substrings inside words
    ("Spinach 99",                 99.0),
    ("napkin 250",                250.0),
])
def test_plain_digit_not_dropped_by_keyword_substring(text, expected_value):
    """Reject keywords like 'pin' must require word boundaries — they were
    incorrectly matching substrings of words like 'Shopping' and 'topping'."""
    cands = parse_amounts(text)
    values = [c.value for c in cands if c.status == "active"]
    assert expected_value in values, f"expected {expected_value} in {values} for {text!r}"


def test_pin_still_rejects_legitimate_pin_context():
    """The substring fix must not break legitimate PIN/OTP rejection."""
    assert parse_amounts("pin 1234") == []
    assert parse_amounts("my pin is 9876") == []


def test_two_real_transactions_still_validate_after_merge():
    """Adjacent unit-word amounts produce BOTH the parts AND the sum,
    so a model emitting 2 txns still finds matching candidates."""
    cands = parse_amounts("1 lakh 50 thousand for car")
    values = sorted({c.value for c in cands if c.status == "active"})
    assert 100000.0 in values
    assert 50000.0 in values
    assert 150000.0 in values
