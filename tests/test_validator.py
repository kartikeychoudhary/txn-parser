import json

from validator import (
    ValidationError, ValidationResult,
    validate_example, serialize_validation_result,
)


def _good_output(amount=500, currency="INR", item="beer", category="Drinks", type_="expense"):
    return {"transactions": [{
        "amount": amount, "currency": currency, "item": item,
        "category": category, "type": type_,
    }]}


def test_validationerror_shape():
    e = ValidationError(code="SCHEMA_INVALID", path="/transactions",
                        message="x", severity="error")
    assert e.code == "SCHEMA_INVALID"
    assert e.severity == "error"


def test_schema_invalid_missing_field_returns_early():
    bad = {"transactions": [{"amount": 500}]}  # missing currency/item/etc
    result = validate_example("500 beer", bad)
    assert not result.ok
    codes = [e.code for e in result.errors]
    # SCHEMA_INVALID present, no downstream codes since validator returns early
    assert "SCHEMA_INVALID" in codes
    assert "AMOUNT_NOT_IN_INPUT" not in codes


def test_schema_invalid_wrong_type():
    bad = {"transactions": "not an array"}
    result = validate_example("500 beer", bad)
    assert not result.ok
    assert any(e.code == "SCHEMA_INVALID" for e in result.errors)


def test_schema_invalid_bad_enum():
    bad = _good_output()
    bad["transactions"][0]["category"] = "NotACategory"
    result = validate_example("500 beer", bad)
    assert not result.ok
    assert any(e.code == "SCHEMA_INVALID" for e in result.errors)


def test_serialize_validation_result_excludes_candidates_and_is_json_safe():
    result = validate_example("500 beer", _good_output())
    payload = serialize_validation_result(result)
    assert "candidates" not in payload
    assert "errors" in payload
    # Round-trip through JSON cleanly.
    json.loads(json.dumps(payload))


def test_amount_not_in_input_when_model_invents_amount():
    out = _good_output(amount=500)
    result = validate_example("600 beer", out)
    assert not result.ok
    assert any(e.code == "AMOUNT_NOT_IN_INPUT" for e in result.errors)


def test_amount_in_input_passes():
    out = _good_output(amount=500)
    result = validate_example("500 beer", out)
    assert result.ok
    assert result.amount_values_match_active_candidates is True


def test_superseded_amount_used_flagged():
    out = _good_output(amount=500)
    result = validate_example("500 beer wait no 600 beer", out)
    assert not result.ok
    assert any(e.code == "SUPERSEDED_AMOUNT_USED" for e in result.errors)
    assert result.superseded_amount_used is True


def test_currency_mismatch_uses_per_candidate_hint():
    out = _good_output(amount=50, currency="INR")
    result = validate_example("$50 coffee", out)
    assert not result.ok
    assert any(e.code == "CURRENCY_MISMATCH" for e in result.errors)


def test_mixed_currency_input_happy_path():
    out = {"transactions": [
        {"amount": 50,  "currency": "USD", "item": "coffee", "category": "Drinks", "type": "expense"},
        {"amount": 300, "currency": "INR", "item": "lunch",  "category": "Food",   "type": "expense"},
    ]}
    result = validate_example("$50 coffee and ₹300 lunch", out)
    assert result.ok, result.errors


def test_txn_count_exceeds_candidates():
    out = {"transactions": [
        _good_output()["transactions"][0],
        _good_output(item="chai", category="Drinks")["transactions"][0],
    ]}
    result = validate_example("500 beer", out)
    assert not result.ok
    assert any(e.code == "TXN_COUNT_EXCEEDS_CANDIDATES" for e in result.errors)


def test_txn_count_below_candidates_is_warning_only():
    out = _good_output(amount=500)
    result = validate_example("500 beer and 50 candy", out)
    codes = [e.code for e in result.errors]
    assert "TXN_COUNT_BELOW_CANDIDATES" in codes
    severities = {e.code: e.severity for e in result.errors}
    assert severities["TXN_COUNT_BELOW_CANDIDATES"] == "warning"
    assert result.ok is True


def test_suspicious_duplicate_unjustified():
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer", out)
    assert not result.ok
    assert any(e.code == "SUSPICIOUS_DUPLICATE" for e in result.errors)
    assert result.duplicate_transactions_found is True


def test_duplicate_justified_by_repeated_amount_and_item():
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer and 500 beer", out)
    assert result.ok, result.errors


def test_duplicate_justified_by_repetition_cue():
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer twice", out)
    # Repetition cue justifies the duplicate, so no SUSPICIOUS_DUPLICATE error
    assert not any(e.code == "SUSPICIOUS_DUPLICATE" for e in result.errors)


def test_duplicate_not_justified_by_different_item_with_same_amount():
    """'500 beer and 500 candy' -> two beer txns is still suspicious."""
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer and 500 candy", out)
    assert not result.ok
    assert any(e.code == "SUSPICIOUS_DUPLICATE" for e in result.errors)


def test_no_amount_in_input_suppresses_per_txn_errors():
    out = _good_output(amount=500)
    result = validate_example("just some words here", out)
    codes = [e.code for e in result.errors]
    assert "NO_AMOUNT_IN_INPUT" in codes
    assert "AMOUNT_NOT_IN_INPUT" not in codes


def test_warn_mode_returns_ok_true_with_errors():
    out = _good_output(amount=500)
    result = validate_example("600 beer", out, mode="warn")
    assert result.ok is True
    assert any(e.code == "AMOUNT_NOT_IN_INPUT" for e in result.errors)
