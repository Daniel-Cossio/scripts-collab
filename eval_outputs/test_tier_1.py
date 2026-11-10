import asyncio
import time
from typing import Optional

class AsyncTokenBucket:
    """
    An asynchronous Token Bucket rate limiter.
    
    :param capacity: The maximum number of tokens the bucket can hold.
    :param refill_rate: The number of tokens added to the bucket per second.
    """
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def _refill(self):
        """Internal method to update the token count based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Consumes tokens immediately if available.
        Returns True if successful, False otherwise.
        """
        async with self.lock:
            await self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Waits asynchronously until enough tokens are available.
        Returns True if tokens are acquired, False if timeout occurs.
        """
        async with self.lock:
            await self._refill()
            
            # If we already have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            # Calculate how much time is needed to get the remaining tokens
            needed = tokens - self.tokens
            time_to_wait = needed / self.refill_rate

            if timeout is not None and time_to_wait > timeout:
                return False

            # Wait for the required time
            # Note: We use asyncio.sleep, but since we are inside a lock, 
            # we must be careful. For a more complex bucket, we might release 
            # the lock during sleep, but for a simple bucket, this works well.
            # To allow concurrent wait_for_tokens, we can release the lock 
            # during the sleep or use a more granular approach.
            
            # Optimization: Release lock during sleep to allow other tasks 
            # to perform 'acquire' or 'wait_for_tokens' calculations.
            # However, to keep it simple and consistent with the refill logic:
            
            # We use a small trick: calculate time, sleep, then re-acquire lock
            # to finalize the consumption.
            # But for a standard implementation, we can just sleep:
            await asyncio.sleep(time_to_wait)
            
            # Re-refill after sleep to ensure precision
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

# --- Pytest Suite ---

import pytest

@pytest.mark.asyncio
async def test_immediate_acquire_success():
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    # Should succeed immediately as capacity is 10
    assert await bucket.acquire(5.0) is True
    # Should succeed again as 5 remain
    assert await bucket.acquire(5.0) is True
    # Should fail as 0 remain
    assert await bucket.acquire(1.0) is False

@pytest.mark.asyncio
async def test_refill_logic():
    # Capacity 10, Refill 10 per second
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    await bucket.acquire(10.0) # Empty the bucket
    
    # Wait 0.5 seconds, should have 5 tokens
    await asyncio.sleep(0.5)
    assert await bucket.acquire(5.0) is True
    assert await bucket.acquire(1.0) is False

@pytest.mark.asyncio
async def test_wait_for_tokens_success():
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    await bucket.acquire(10.0) # Empty it
    
    # Wait for 5 tokens (should take 0.5 seconds)
    start_time = time.monotonic()
    success = await bucket.wait_for_tokens(5.0, timeout=1.0)
    end_time = time.monotonic()
    
    assert success is True
    assert (end_time - start_time) >= 0.4  # Allow slight variance
    assert (end_time - start_time) <= 0.7

@pytest.mark.asyncio
async def test_wait_for_tokens_timeout():
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    await bucket.acquire(10.0) # Empty it
    
    # Need 5 tokens at 1 token/sec = 5 seconds. Timeout is 1 second.
    success = await bucket.wait_for_tokens(5.0, timeout=1.0)
    assert success is False

@pytest.mark.asyncio
async def test_concurrent_acquires():
    # 10 tasks trying to acquire 1 token each simultaneously
    # Capacity 10, Refill 10/sec
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    
    # Run 10 tasks concurrently
    tasks = [bucket.acquire(1.0) for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed because capacity is 10
    assert all(results)
    
    # Next one should fail
    assert await bucket.acquire(1.0) is False

@pytest.mark.asyncio
async def test_concurrent_wait_for_tokens():
    # 5 tasks waiting for 2 tokens each
    # Capacity 10, Refill 10/sec
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    await bucket.acquire(10.0) # Empty it
    
    tasks = [bucket.wait_for_tokens(2.0, timeout=2.0) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    assert all(results)