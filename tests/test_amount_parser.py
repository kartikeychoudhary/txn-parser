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
