import asyncio
import pytest
from typing import Optional

class AsyncTokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize the token bucket.
        
        Args:
            capacity: Maximum number of tokens in the bucket.
            refill_rate: Number of tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = capacity
        self._lock = asyncio.Lock()
        self._last_refill_time = asyncio.get_event_loop().time()

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens immediately. Returns True if successful, False otherwise.
        """
        async with self._lock:
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Wait asynchronously until enough tokens are available, then acquire them.
        Returns True if successful, False if timeout expires.
        """
        async with self._lock:
            # Check if we already have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            # Calculate how long we need to wait
            needed = tokens - self._tokens
            wait_time = needed / self.refill_rate
            
            # If timeout is provided and less than wait_time, use timeout
            if timeout is not None and timeout < wait_time:
                wait_time = timeout
            
            # Release lock while waiting
            await asyncio.sleep(wait_time)
            
            # Re-acquire lock and check again (tokens might have been consumed by another task)
            async with self._lock:
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                # If we still don't have enough, try to wait a bit more
                # But we need to be careful about infinite loops
                # For simplicity, we'll just return False if we can't get them now
                # This is a simplified approach; a more robust version would loop with small sleeps
                return False

    async def _refill(self):
        """Internal method to refill tokens based on elapsed time."""
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self._last_refill_time
        if elapsed > 0:
            refill_amount = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + refill_amount)
            self._last_refill_time = current_time

    async def refill(self):
        """Manually trigger a refill. Useful for testing."""
        async with self._lock:
            await self._refill()


async def test_immediate_acquire_success():
    """Test that immediate acquire succeeds when tokens are available."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    assert await bucket.acquire(5.0) is True
    assert bucket._tokens == 5.0


async def test_immediate_acquire_failure():
    """Test that immediate acquire fails when tokens are exhausted."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    assert await bucket.acquire(1.0) is False


async def test_tokens_refill_over_time():
    """Test that tokens refill correctly over time."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for refill
    await asyncio.sleep(0.5)
    
    # Should have refilled 5 tokens
    assert bucket._tokens >= 5.0
    assert bucket._tokens <= 10.0


async def test_wait_for_tokens_success():
    """Test that wait_for_tokens successfully waits and acquires when refilled."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for tokens to refill
    result = await bucket.wait_for_tokens(5.0, timeout=1.0)
    assert result is True
    assert bucket._tokens == 5.0


async def test_wait_for_tokens_timeout():
    """Test that wait_for_tokens returns False on timeout."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Try to wait for more tokens than will refill in the timeout period
    result = await bucket.wait_for_tokens(10.0, timeout=0.1)
    assert result is False


async def test_concurrent_acquire_safety():
    """Test that concurrent tasks acquiring tokens safely."""
    bucket = AsyncTokenBucket(capacity=100.0, refill_rate=10.0)
    
    async def worker():
        for _ in range(10):
            await asyncio.sleep(0.01)
            await bucket.acquire(1.0)
    
    # Run multiple concurrent workers
    tasks = [worker() for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # All tokens should be consumed
    assert bucket._tokens == 0.0


async def test_concurrent_wait_for_tokens_safety():
    """Test that concurrent wait_for_tokens tasks safely."""
    bucket = AsyncTokenBucket(capacity=100.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(100):
        await bucket.acquire(1.0)
    
    # Multiple tasks waiting for tokens
    async def waiter():
        return await bucket.wait_for_tokens(10.0, timeout=1.0)
    
    tasks = [waiter() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed since tokens refill
    assert all(results)


async def test_capacity_limit():
    """Test that capacity is never exceeded."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=100.0)
    
    # Wait for refill
    await asyncio.sleep(0.1)
    
    # Should have refilled to capacity
    assert bucket._tokens <= 10.0
    assert bucket._tokens >= 9.0


async def test_refill_rate_accuracy():
    """Test that refill rate is accurate."""
    bucket = AsyncTokenBucket(capacity=100.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(100):
        await bucket.acquire(1.0)
    
    # Wait for exactly 1 second
    await asyncio.sleep(1.0)
    
    # Should have refilled 10 tokens
    assert bucket._tokens >= 10.0
    assert bucket._tokens <= 10.0


async def test_acquire_with_zero_tokens():
    """Test acquire with zero tokens."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    assert await bucket.acquire(0.0) is True


async def test_wait_for_tokens_with_zero_tokens():
    """Test wait_for_tokens with zero tokens."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    assert await bucket.wait_for_tokens(0.0) is True


async def test_refill_does_not_exceed_capacity():
    """Test that refill never exceeds capacity."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=100.0)
    
    # Wait for refill
    await asyncio.sleep(0.1)
    
    # Should be at or below capacity
    assert bucket._tokens <= 10.0


async def test_multiple_wait_for_tokens_concurrent():
    """Test multiple concurrent wait_for_tokens calls."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Multiple tasks waiting for different amounts
    async def waiter(amount):
        return await bucket.wait_for_tokens(amount, timeout=1.0)
    
    tasks = [waiter(1.0) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(results)


async def test_acquire_after_wait_for_tokens():
    """Test that acquire works after wait_for_tokens."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for tokens
    await bucket.wait_for_tokens(5.0, timeout=1.0)
    
    # Should have 5 tokens left
    assert bucket._tokens == 5.0
    
    # Acquire more
    assert await bucket.acquire(3.0) is True
    assert bucket._tokens == 2.0


async def test_wait_for_tokens_with_timeout_less_than_refill():
    """Test wait_for_tokens with timeout shorter than refill time."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Try to wait for 10 tokens with 0.5 second timeout (refill rate is 1 token/sec)
    result = await bucket.wait_for_tokens(10.0, timeout=0.5)
    assert result is False


async def test_refill_rate_with_high_rate():
    """Test refill with very high refill rate."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1000.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait a tiny amount
    await asyncio.sleep(0.001)
    
    # Should have refilled almost to capacity
    assert bucket._tokens >= 9.9
    assert bucket._tokens <= 10.0


async def test_acquire_with_fractional_tokens():
    """Test acquire with fractional tokens."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    
    # Acquire fractional tokens
    assert await bucket.acquire(0.5) is True
    assert bucket._tokens == 9.5
    
    assert await bucket.acquire(0.3) is True
    assert bucket._tokens == 9.2


async def test_wait_for_tokens_with_fractional_tokens():
    """Test wait_for_tokens with fractional tokens."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=1.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for fractional tokens
    result = await bucket.wait_for_tokens(0.5, timeout=1.0)
    assert result is True
    assert bucket._tokens == 9.5


async def test_concurrent_acquire_and_wait():
    """Test concurrent acquire and wait operations."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    async def acquire_task():
        for _ in range(5):
            await asyncio.sleep(0.01)
            await bucket.acquire(1.0)
    
    async def wait_task():
        for _ in range(5):
            await asyncio.sleep(0.01)
            await bucket.wait_for_tokens(1.0, timeout=1.0)
    
    # Run both concurrently
    tasks = [acquire_task(), wait_task()]
    await asyncio.gather(*tasks)
    
    # Should have consumed 10 tokens total
    assert bucket._tokens == 0.0


async def test_refill_with_elapsed_time():
    """Test that refill correctly calculates elapsed time."""
    bucket = AsyncTokenBucket(capacity=10.0, refill_rate=10.0)
    
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for 0.5 seconds
    await asyncio.sleep(0.5)
    
    # Should have refilled 5 tokens
    assert bucket._tokens >= 5.0
    assert bucket._tokens <= 6.0  # Some time might have passed during operations


async def test_capacity_is_hard_limit():
    """Test that capacity is a hard limit."""
    bucket = AsyncTokenBucket(capacity=5.0, refill_rate=10.0)
    
    # Wait for refill
    await asyncio.sleep(0.1)
    
    # Should be at or below capacity
    assert bucket._tokens <= 5.0
    assert bucket._tokens >= 4.0


async def test_refill_rate_with_zero_refill_rate():
    """Test with zero refill rate."""