import pytest
from _retry import retry_with_backoff


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def test_succeeds_on_first_try():
    calls = []
    def ok():
        calls.append(1)
        return "ok"
    result = retry_with_backoff(
        ok,
        retryable=(TransientError,),
        sleep=lambda _: None,
    )
    assert result == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    state = {"attempts": 0}
    def flaky():
        state["attempts"] += 1
        if state["attempts"] < 3:
            raise TransientError("nope")
        return "ok"
    result = retry_with_backoff(
        flaky,
        retryable=(TransientError,),
        max_retries=3,
        sleep=lambda _: None,
    )
    assert result == "ok"
    assert state["attempts"] == 3


def test_exhausts_retries_and_reraises():
    def always_fails():
        raise TransientError("persistent")
    with pytest.raises(TransientError):
        retry_with_backoff(
            always_fails,
            retryable=(TransientError,),
            max_retries=2,
            sleep=lambda _: None,
        )


def test_non_retryable_exception_propagates_immediately():
    state = {"calls": 0}
    def fails():
        state["calls"] += 1
        raise PermanentError("auth")
    with pytest.raises(PermanentError):
        retry_with_backoff(
            fails,
            retryable=(TransientError,),
            max_retries=5,
            sleep=lambda _: None,
        )
    # Permanent exception is not in `retryable` tuple, so no retries happened.
    assert state["calls"] == 1


def test_is_retryable_predicate_filters():
    """Used by Gemini: ClientError is retryable iff status in {429,5xx}."""
    class StatusError(Exception):
        def __init__(self, status):
            super().__init__(f"status={status}")
            self.status_code = status

    state = {"calls": 0}
    def call_with_400():
        state["calls"] += 1
        raise StatusError(400)

    with pytest.raises(StatusError):
        retry_with_backoff(
            call_with_400,
            retryable=(StatusError,),
            is_retryable=lambda e: e.status_code in {429, 500, 502, 503, 504},
            max_retries=5,
            sleep=lambda _: None,
        )
    # 400 fails the predicate, so we bail after the first call.
    assert state["calls"] == 1


def test_sleep_durations_grow_exponentially():
    sleeps = []
    def fail():
        raise TransientError("x")
    with pytest.raises(TransientError):
        retry_with_backoff(
            fail,
            retryable=(TransientError,),
            max_retries=3,
            base_delay=1.0,
            jitter=0.0,
            sleep=sleeps.append,
        )
    # 3 retries → 3 sleeps. Each at least 1, 2, 4.
    assert len(sleeps) == 3
    assert sleeps[0] >= 1.0
    assert sleeps[1] >= 2.0
    assert sleeps[2] >= 4.0
