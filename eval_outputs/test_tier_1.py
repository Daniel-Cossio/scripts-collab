import asyncio
import time
from typing import Optional


class AsyncTokenBucket:
    """An asynchronous token bucket rate limiter."""

    def __init__(self, capacity: float, refill_rate: float):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called while holding the lock."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Consume tokens immediately if available.
        Returns True if tokens were consumed, False otherwise.
        """
        if tokens <= 0:
            return True
        if tokens > self.capacity:
            return False

        async with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float = 1.0, timeout: float = None) -> bool:
        """
        Wait asynchronously until enough tokens are available, then consume them.
        Returns True if tokens were consumed, False if timeout expired.
        """
        if tokens <= 0:
            return True
        if tokens > self.capacity:
            return False

        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        async with self.lock:
            self._refill()
            while True:
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
