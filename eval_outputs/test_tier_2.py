import asyncio
import time
from typing import Optional

class AsyncTokenBucket:
    """
    An asynchronous implementation of the Token Bucket algorithm for rate limiting.
    """
    def __init__(self, capacity: float, refill_rate: float):
        """
        :param capacity: Maximum number of tokens the bucket can hold.
        :param refill_rate: Number of tokens added to the bucket per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def _refill(self) -> None:
        """Internal method to update token count based on elapsed time."""
        now = time.monotonic()
        delta = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        self.last_update = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Attempts to consume tokens immediately.
        Returns True if successful, False otherwise.
        """
        async with self._lock:
            await self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Waits asynchronously until enough tokens are available.
        Returns True if tokens were acquired, False if timeout occurred.
        """
        start_time = time.monotonic()
        
        while True:
            async with self._lock:
                await self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate time needed until enough tokens are available
                needed = tokens - self.tokens
                wait_time = needed / self.refill_rate
            
            # Check for timeout
            elapsed = time.monotonic() - start_time
            if timeout is not None and elapsed + wait_time > timeout:
                # We can still wait for the remaining timeout period
                remaining = timeout - elapsed
                if remaining <= 0:
                    return False
                await asyncio.sleep(remaining)
                # Final check after timeout sleep
                async with self._lock:
                    await self._refill()
                    if self.tokens >= tokens:
                        self.tokens -= tokens
                        return True
                    return False
            
            await asyncio.sleep(wait_time)

# --- Test Suite ---
import pytest
import pytest_asyncio

@pytest.mark.asyncio
async def test_immediate_acquire_success():
    # Capacity 5, refill 1/s. Should acquire 5 immediately.
    bucket = AsyncTokenBucket(5.0, 1.0)
    assert await bucket.acquire(5.0) is True
    assert await bucket.acquire(1.0) is False

@pytest.mark.asyncio
async def test_immediate_acquire_fail():
    bucket = AsyncTokenBucket(1.0, 1.0)
    assert await bucket.acquire(1.0) is True
    assert await bucket.acquire(1.0) is False

@pytest.mark.asyncio
async def test_token_refill():
    # Capacity 2, refill 10 per second.
    bucket = AsyncTokenBucket(2.0, 10.0)
    assert await bucket.acquire(2.0) is True
    assert await bucket.acquire(1.0) is False
    
    # Wait 0.2s -> should refill 2 tokens
    await asyncio.sleep(0.25) 
    assert await bucket.acquire(2.0) is True

@pytest.mark.asyncio
async def test_wait_for_tokens_success():
    # Capacity 1, refill 2 per second (1 token every 0.5s)
    bucket = AsyncTokenBucket(1.0, 2.0)
    assert await bucket.acquire(1.0) is True
    
    # This should wait approx 0.5s and then succeed
    start = time.monotonic()
    result = await bucket.wait_for_tokens(1.0, timeout=1.0)
    end = time.monotonic()
    
    assert result is True
    assert 0.4 <= (end - start) <= 0.7

@pytest.mark.asyncio
async def test_wait_for_tokens_timeout():
    # Capacity 1, refill 1 per second
    bucket = AsyncTokenBucket(1.0, 1.0)
    assert await bucket.acquire(1.0) is True
    
    # Try to acquire 2 tokens (impossible since capacity is 1) with timeout
    result = await bucket.wait_for_tokens(2.0, timeout=0.2)
    assert result is False

@pytest.mark.asyncio
async def test_concurrent_acquisition():
    # Capacity 10, refill 100/s. 
    # 10 tasks trying to acquire 1 token each.
    bucket = AsyncTokenBucket(10.0, 100.0)
    
    async def worker():
        return await bucket.acquire(1.0)

    results = await asyncio.gather(*(worker() for _ in range(10)))
    assert all(results) is True
    
    # 11th should fail
    assert await bucket.acquire(1.0) is False

@pytest.mark.asyncio
async def test_high_concurrency_wait():
    # Small capacity, slow refill, many waiters
    bucket = AsyncTokenBucket(1.0, 5.0) # 1 token every 0.2s
    assert await bucket.acquire(1.0) is True
    
    async def worker():
        return await bucket.wait_for_tokens(1.0, timeout=1.0)

    # Start 3 workers. Total time should be ~0.6s
    start = time.monotonic()
    results = await asyncio.gather(*(worker() for _ in range(3)))
    end = time.monotonic()
    
    assert all(results) is True
    assert 0.5 <= (end - start) <= 0.8