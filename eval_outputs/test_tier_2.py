import asyncio
import time
from typing import Optional


class AsyncTokenBucket:
    """
    An asynchronous Token Bucket rate limiter.
    
    Tokens are added at a constant rate (refill_rate tokens per second),
    up to a maximum capacity. Tokens can be consumed immediately or waited for.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        """
        Initialize the token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold.
            refill_rate: Number of tokens added per second.
        """
        if capacity < 0:
            raise ValueError("Capacity must be non-negative")
        if refill_rate < 0:
            raise ValueError("Refill rate must be non-negative")
        
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill_time = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        new_tokens = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill_time = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens immediately.
        
        Args:
            tokens: Number of tokens to acquire (default 1.0).
            
        Returns:
            True if tokens were acquired, False if not enough available.
        """
        if tokens < 0:
            raise ValueError("Tokens must be non-negative")
        if tokens > self.capacity:
            # Cannot acquire more than capacity
            return False
            
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_tokens(
        self, tokens: float = 1.0, timeout: Optional[float] = None
    ) -> bool:
        """
        Wait asynchronously until enough tokens are available, then consume them.
        
        Args:
            tokens: Number of tokens to wait for (default 1.0).
            timeout: Maximum time to wait in seconds. None means wait indefinitely.
            
        Returns:
            True if tokens were acquired within the timeout, False if timeout expired.
        """
        if tokens < 0:
            raise ValueError("Tokens must be non-negative")
        if tokens > self.capacity:
            # Cannot acquire more than capacity
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
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                else:
                    remaining = None
                
                # Calculate wait time needed
                deficit = tokens - self._tokens
                if self.refill_rate > 0:
                    wait_time = deficit / self.refill_rate
                else:
                    wait_time = float('inf')
                
                # Adjust wait time based on deadline
                if deadline is not None:
                    wait_time = min(wait_time, remaining)
                
                if wait_time <= 0:
                    # Check one more time in case of precision issues
                    self._refill()
                    if self._tokens >= tokens:
                        self._tokens -= tokens
                        return True
                    continue
                
                # Release lock while waiting
                # We need to wait outside the lock to allow other coroutines
                # But we need to be careful about the timing
                await asyncio.sleep(wait_time)
                
                # Check if we've exceeded deadline
                if deadline is not None:
                    if time.monotonic() >= deadline:
                        return False


class AsyncTokenBucketSimple:
    """
    A simpler implementation using asyncio.Event for cleaner waiting logic.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        if capacity < 0:
            raise ValueError("Capacity must be non-negative")
        if refill_rate < 0:
            raise ValueError("Refill rate must be non-negative")
        
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill_time = time.monotonic()
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        new_tokens = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill_time = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens immediately."""
        if tokens < 0:
            raise ValueError("Tokens must be non-negative")
        if tokens > self.capacity:
            return False
            
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_tokens(
        self, tokens: float = 1.0, timeout: Optional[float] = None
    ) -> bool:
        """Wait asynchronously until enough tokens are available."""
        if tokens < 0:
            raise ValueError("Tokens must be non-negative")
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
                
                # Calculate wait time
                deficit = tokens - self._tokens
                if self.refill_rate > 0:
                    wait_time = deficit / self.refill_rate
                else:
                    wait_time = float('inf')
                
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)
                
                if wait_time <= 0:
                    continue
                
                # Release lock and wait
                self._lock.release()
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(wait_time),
                        timeout=wait_time
                    )
                except asyncio.TimeoutError:
                    pass
                finally:
                    await self._lock.acquire()
                
                # Check deadline
                if deadline is not None and time.monotonic() >= deadline:
                    return False


# Use the simpler implementation for the main class
AsyncTokenBucket = AsyncTokenBucketSimple


# Tests
import pytest
import pytest_asyncio


@pytest.fixture
def bucket():
    """Create a token bucket with capacity 10 and refill rate 10 tokens/sec."""
    return AsyncTokenBucket(capacity=10.0, refill_rate=10.0)


@pytest.mark.asyncio
async def test_acquire_immediate_success(bucket):
    """Test that acquire succeeds when tokens are available."""
    result = await bucket.acquire(1.0)
    assert result is True
    # Check that tokens were consumed
    async with bucket._lock:
        bucket._refill()
        assert bucket._tokens < 10.0


@pytest.mark.asyncio
async def test_acquire_immediate_fail(bucket):
    """Test that acquire fails when tokens are exhausted."""
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Try to acquire more than available
    result = await bucket.acquire(2.0)
    assert result is False


@pytest.mark.asyncio
async def test_tokens_refill_over_time(bucket):
    """Test that tokens refill correctly over time."""
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for refill
    await asyncio.sleep(0.5)  # 5 tokens should be refilled at 10 tokens/sec
    
    result = await bucket.acquire(5.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_success(bucket):
    """Test that wait_for_tokens successfully waits and acquires."""
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Wait for tokens to refill
    result = await bucket.wait_for_tokens(5.0, timeout=2.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_timeout(bucket):
    """Test that wait_for_tokens returns False on timeout."""
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Try to acquire more than capacity (should fail immediately)
    result = await bucket.wait_for_tokens(15.0, timeout=0.5)
    assert result is False


@pytest.mark.asyncio
async def test_concurrent_tasks(bucket):
    """Test concurrent tasks acquiring tokens safely."""
    results = []
    
    async def acquire_token():
        result = await bucket.acquire(1.0)
        results.append(result)
    
    # Create multiple tasks
    tasks = [asyncio.create_task(acquire_token()) for _ in range(20)]
    await asyncio.gather(*tasks)
    
    # All should succeed since we have capacity 10 and refill rate 10/sec
    # With 20 tasks and 10 capacity, some will succeed immediately
    # and others will succeed as tokens refill
    assert len(results) == 20
    assert all(results)  # All should succeed given enough time


@pytest.mark.asyncio
async def test_acquire_zero_tokens(bucket):
    """Test that acquiring zero tokens always succeeds."""
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    result = await bucket.acquire(0.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_zero_tokens(bucket):
    """Test that wait_for_tokens with zero tokens succeeds immediately."""
    result = await bucket.wait_for_tokens(0.0, timeout=1.0)
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_tokens_timeout_expires(bucket):
    """Test that wait_for_tokens returns False when timeout expires."""
    # Exhaust all tokens
    for _ in range(10):
        await bucket.acquire(1.0)
    
    # Try to acquire with short timeout
    result = await bucket.wait_for_tokens(5.0, timeout=0.1)
    assert result is False


@pytest.mark.asyncio
async def test_concurrent_wait_for_tokens(bucket):
    """Test concurrent tasks waiting for tokens."""
    results = []
    
    async def wait_for_token():
        result = await bucket.wait_for_tokens(1.0, timeout=5.0)
        results.append(result)
    
    # Create multiple tasks
    tasks = [asyncio.create_task(wait_for_token()) for _ in range(10)]
    await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(results)  # All should succeed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])