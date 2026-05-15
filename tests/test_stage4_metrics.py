import importlib


eval_mod = importlib.import_module("04_eval")
score_validation_fields = eval_mod.score_validation_fields


def _expected():
    return {"transactions": [{
        "amount": 500, "currency": "INR", "item": "beer",
        "category": "Drinks", "type": "expense",
    }]}


def test_invalid_json_produces_json_parse_failed():
    fields = score_validation_fields("500 beer", _expected(), "not json")
    codes = [e["code"] for e in fields["validation_errors"]]
    assert codes == ["JSON_PARSE_FAILED"]
    assert fields["amount_exact"] is False
    assert fields["txn_count_exact"] is False
    assert fields["amount_values_match_active_candidates"] is False
    assert fields["txn_count_matches_active_candidates"] is False
    assert fields["duplicate_transactions_found"] is False
    assert fields["superseded_amount_used"] is False


def test_helper_accepts_pre_parsed_predicted_to_avoid_double_parse():
    predicted = {"transactions": [{
        "amount": 500, "currency": "INR", "item": "beer",
        "category": "Drinks", "type": "expense",
    }]}
    fields = score_validation_fields("500 beer", _expected(),
                                     predicted_raw="<unused>",
                                     predicted=predicted)
    assert fields["amount_exact"] is True
    assert fields["txn_count_exact"] is True
    assert fields["amount_values_match_active_candidates"] is True


def test_amount_exact_uses_isclose_so_int_vs_float_equal():
    predicted = {"transactions": [{
        "amount": 500.0, "currency": "INR", "item": "beer",
        "category": "Drinks", "type": "expense",
    }]}
    fields = score_validation_fields("500 beer", _expected(),
                                     predicted_raw="<unused>",
                                     predicted=predicted)
    assert fields["amount_exact"] is True
