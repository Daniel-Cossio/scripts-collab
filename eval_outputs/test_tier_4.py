"""Asynchronous Token Bucket rate limiter implementation and tests."""

from __future__ import annotations

import asyncio
import time
from typing import Optional


class AsyncTokenBucket:
    """Token bucket rate limiter that works with asyncio coroutines.

    Args:
        capacity: Maximum number of tokens the bucket can hold.
        refill_rate: Number of tokens added per second.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """Consume tokens immediately if available.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False otherwise.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_tokens(
        self, tokens: float = 1.0, timeout: Optional[float] = None
    ) -> bool:
        """Wait asynchronously until enough tokens are available.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            True if tokens were acquired, False if timeout expired.
        """
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait_time = min(remaining, 0.01)
            else:
                wait_time = 0.01

            await asyncio.sleep(wait_time)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
def bucket() -> AsyncTokenBucket:
    return AsyncTokenBucket(capacity=10.0, refill_rate=10.0)


@pytest.mark.asyncio
async def test_immediate_acquire_succeeds(bucket: AsyncTokenBucket) -> None:
    result = await bucket.acquire(tokens=1.0)
    assert result is True


@pytest.mark.asyncio
async def test_immediate_acquire_fails_when_exhausted(bucket: AsyncTokenBucket) -> None:
    # Drain all tokens
    for _ in range(10):
        await bucket.acquire(tokens=1.0)
    result = await bucket.acquire(tokens=1.0)
    assert result is False


@pytest.mark.asyncio
async def test_tokens_refill_correctly(bucket: AsyncTokenBucket) -> None:
    # Drain all tokens
    for _ in range(10):
        await bucket.acquire(tokens=1.0)
    # Wait for partial refill
    await asyncio.sleep(0.5)
    result = await bucket.acquire(tokens=1.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_succeeds(bucket: AsyncTokenBucket) -> None:
    # Drain all tokens
    for _ in range(10):
        await bucket.acquire(tokens=1.0)
    result = await bucket.wait_for_tokens(tokens=1.0, timeout=2.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_returns_false_on_timeout(bucket: AsyncTokenBucket) -> None:
    # Create a bucket with very low refill rate
    slow_bucket = AsyncTokenBucket(capacity=1.0, refill_rate=0.1)
    await slow_bucket.acquire(tokens=1.0)
    result = await slow_bucket.wait_for_tokens(tokens=1.0, timeout=0.1)
    assert result is False


@pytest.mark.asyncio
async def test_concurrent_tasks_acquire_safely() -> None:
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=100.0)
    results = await asyncio.gather(
        *(bucket.acquire(tokens=1.0) for _ in range(15))
    )
    # Exactly 10 should succeed (capacity=10)
    assert sum(results) == 10


@pytest.mark.asyncio
async def test_wait_for_tokens_concurrent() -> None:
    bucket = AsyncTokenBucket(capacity=5.0, refill_rate=10.0)
    # Drain all tokens
    for _ in range(5):
        await bucket.acquire(tokens=1.0)

    # Multiple tasks waiting for tokens
    results = await asyncio.gather(
        *(bucket.wait_for_tokens(tokens=1.0, timeout=2.0) for _ in range(10))
    )
    # At least 5 should succeed (capacity=5, plus refills during wait)
    assert sum(results) >= 5


@pytest.mark.asyncio
async def test_acquire_partial_tokens(bucket: AsyncTokenBucket) -> None:
    result = await bucket.acquire(tokens=3.5)
    assert result is True
    assert bucket._tokens == pytest.approx(6.5, abs=0.1)


@pytest.mark.asyncio
async def test_wait_for_tokens_timeout_zero() -> None:
    bucket = AsyncTokenBucket(capacity=1.0, refill_rate=0.01)
    await bucket.acquire(tokens=1.0)
    result = await bucket.wait_for_tokens(tokens=1.0, timeout=0.0)
    assert result is False


@pytest.mark.asyncio
async def test_wait_for_tokens_no_timeout() -> None:
    bucket = AsyncTokenBucket(capacity=1.0, refill_rate=100.0)
    await bucket.acquire(tokens=1.0)
    result = await bucket.wait_for_tokens(tokens=1.0, timeout=None)
    assert result is True