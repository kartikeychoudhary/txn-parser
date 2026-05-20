import pytest

from _lib import clean_input_line, normalize_input


@pytest.mark.parametrize("raw,expected", [
    # Numbered list — strip
    ("1. 500 rs on beer", "500 rs on beer"),
    ("2) 500 rs on beer", "500 rs on beer"),
    ("12. paid 200 for chai", "paid 200 for chai"),
    # Bullets — strip
    ("- 500 rs on beer", "500 rs on beer"),
    ("* 500 rs on beer", "500 rs on beer"),
    ("  - 500 rs on beer", "500 rs on beer"),
    # Digit-led NATURAL content — must NOT strip (regression for the old regex bug)
    ("500 beer", "500 beer"),
    ("200 chai", "200 chai"),
    ("1 lakh rent", "1 lakh rent"),
    ("500 rs on beer 50 rs on candy", "500 rs on beer 50 rs on candy"),
    ("12500 emi for laptop", "12500 emi for laptop"),
    # Quotes — strip
    ('"500 beer"', "500 beer"),
    ("'500 beer'", "500 beer"),
    # Junk — None
    ("", None),
    ("```", None),
    ("output: 500 beer", None),
    ("example: something", None),
    ("ab", None),                  # too short
    ("x" * 301, None),             # too long
])
def test_clean_input_line(raw, expected):
    assert clean_input_line(raw) == expected


def test_normalize_input_collapses_whitespace_and_lowercases():
    assert normalize_input("  500   Rs On BEER  ") == "500 rs on beer"


def test_normalize_input_idempotent():
    once = normalize_input("500 beer ")
    assert normalize_input(once) == once


def test_normalize_input_distinguishes_real_differences():
    assert normalize_input("500 beer") != normalize_input("500 chai")
