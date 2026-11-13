import asyncio
import time
from typing import Optional


class AsyncTokenBucket:
    """An asynchronous token bucket rate limiter."""

    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize the token bucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold.
            refill_rate: Number of tokens added per second.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if refill_rate < 0:
            raise ValueError("Refill rate must be non-negative")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called while holding the lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            tokens_to_add = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens immediately.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False otherwise.
        """
        if tokens <= 0:
            return True
        if tokens > self.capacity:
            return False

        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Wait asynchronously until enough tokens are available, then consume them.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            True if tokens were consumed, False if timeout expired.
        """
        if tokens <= 0:
            return True
        if tokens > self.capacity:
            return False

        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # Calculate how long to wait for enough tokens
                needed = tokens - self._tokens
                if self.refill_rate <= 0:
                    # No refill, so we can never get enough tokens
                    if deadline is not None:
                        # Wait until timeout
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                        return False
                    else:
                        # Wait forever (but this will hang)
                        await asyncio.sleep(0.1)
                        continue

                wait_time = needed / self.refill_rate

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

            # Sleep outside the lock to allow other tasks to proceed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.001)


import pytest


@pytest.mark.asyncio
async def test_immediate_acquire_succeeds_when_tokens_available():
    """Test that acquire succeeds when tokens are available."""
    bucket = AsyncTokenBucket(capacity=10, refill_rate=1.0)
    result = await bucket.acquire(5)
    assert result is True
    # Should have 5 tokens left
    result = await bucket.acquire(5)
    assert result is True
    # Should have 0 tokens left
    result = await bucket.acquire(1)
    assert result is False


@pytest.mark.asyncio
async def test_immediate_acquire_fails_when_tokens_exhausted():
    """Test that acquire fails when tokens are exhausted."""
    bucket = AsyncTokenBucket(capacity=5, refill_rate=0.0)
    # Consume all tokens
    for _ in range(5):
        result = await bucket.acquire(1)
        assert result is True
    # Should fail now
    result = await bucket.acquire(1)
    assert result is False


@pytest.mark.asyncio
async def test_tokens_refill_correctly_over_time():
    """Test that tokens refill correctly over time."""
    bucket = AsyncTokenBucket(capacity=10, refill_rate=10.0)
    # Consume all tokens
    for _ in range(10):
        await bucket.acquire(1)
    # Wait for refill (1 second should add 10 tokens)
    await asyncio.sleep(1.1)
    # Should be able to acquire tokens again
    result = await bucket.acquire(5)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_successfully_waits_and_acquires():
    """Test that wait_for_tokens waits and acquires when refilled."""
    bucket = AsyncTokenBucket(capacity=5, refill_rate=10.0)
    # Consume all tokens
    for _ in range(5):
        await bucket.acquire(1)
    # Wait for tokens with a timeout
    result = await bucket.wait_for_tokens(3, timeout=2.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_returns_false_on_timeout():
    """Test that wait_for_tokens returns False on timeout."""
    bucket = AsyncTokenBucket(capacity=5, refill_rate=0.1)
    # Consume all tokens
    for _ in range(5):
        await bucket.acquire(1)
    # Try to wait for more tokens than can be refilled in the timeout
    result = await bucket.wait_for_tokens(10, timeout=0.5)
    assert result is False


@pytest.mark.asyncio
async def test_concurrent_tasks_acquiring_tokens_safely():
    """Test that concurrent tasks can safely acquire tokens."""
    bucket = AsyncTokenBucket(capacity=10, refill_rate=100.0)

    async def worker(worker_id: int) -> bool:
        return await bucket.acquire(1)

    # Launch multiple concurrent tasks
    tasks = [worker(i) for i in range(20)]
    results = await asyncio.gather(*tasks)

    # With capacity 10 and high refill rate, some should succeed
    # At least the first 10 should succeed immediately
    successful = sum(1 for r in results if r)
    assert successful >= 10
    assert successful <= 20