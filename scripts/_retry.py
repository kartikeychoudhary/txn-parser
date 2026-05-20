"""Exponential-backoff retry with jitter. Pure stdlib.

Used by DeepSeekProvider, GeminiProvider, and any future provider that
talks to an external API. Slice 4 will likely add observability hooks
here (per-attempt latency, attempt counter) — keep the surface small now.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, Type, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retryable: tuple[Type[BaseException], ...],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
    is_retryable: Callable[[BaseException], bool] | None = None,
    logger_name: str = "retry",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`; on a retryable exception, sleep and retry up to max_retries.

    Retry predicate:
      - Exception type must be in `retryable` tuple, AND
      - If `is_retryable` is provided, it must return True for the exception.

    Sleep duration: min(max_delay, base_delay * 2**attempt) + uniform(0, jitter).
    On final failure: re-raises the original exception unchanged.
    """
    logger = logging.getLogger(logger_name)
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable as e:
            if is_retryable is not None and not is_retryable(e):
                raise   # non-retryable subclass; bail immediately
            last_exc = e
            if attempt >= max_retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, jitter)
            logger.warning(
                "Attempt %d/%d failed: %s. Sleeping %.1fs before retry.",
                attempt + 1, max_retries + 1, e, delay,
            )
            sleep(delay)
    assert last_exc is not None
    raise last_exc
