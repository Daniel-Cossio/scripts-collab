import asyncio
from typing import Optional
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture


class AsyncTokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("Refill rate must be positive")
        
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity
        self._last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> bool:
        if tokens <= 0:
            return True
        
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_update
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        if tokens <= 0:
            return True
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                elapsed = now - self._last_update
                self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
                self._last_update = now
                
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return False
            
            await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting


@pytest.mark.asyncio
class TestAsyncTokenBucket:
    async def test_immediate_acquire_success(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=1)
        assert await bucket.acquire(5) is True
        assert bucket._tokens == 5

    async def test_immediate_acquire_failure(self):
        bucket = AsyncTokenBucket(capacity=5, refill_rate=1)
        assert await bucket.acquire(10) is False
        assert bucket._tokens == 5

    async def test_tokens_refill_over_time(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=10)
        assert await bucket.acquire(10) is True
        assert bucket._tokens == 0
        
        await asyncio.sleep(0.5)
        await bucket.acquire(1)  # Should succeed after refill
        assert bucket._tokens == 4  # 10 - 1 + 10 * 0.5

    async def test_wait_for_tokens_success(self):
        bucket = AsyncTokenBucket(capacity=5, refill_rate=5)
        assert await bucket.acquire(5) is True
        assert bucket._tokens == 0
        
        # Wait for tokens to refill
        result = await bucket.wait_for_tokens(2, timeout=1.0)
        assert result is True
        assert bucket._tokens == 3

    async def test_wait_for_tokens_timeout(self):
        bucket = AsyncTokenBucket(capacity=5, refill_rate=1)
        assert await bucket.acquire(5) is True
        assert bucket._tokens == 0
        
        # Should timeout before tokens refill
        result = await bucket.wait_for_tokens(2, timeout=0.1)
        assert result is False
        assert bucket._tokens == 0

    async def test_concurrent_acquire(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=100)
        
        # First acquire should succeed
        assert await bucket.acquire(5) is True
        assert bucket._tokens == 5
        
        # Concurrent acquires should handle race conditions
        tasks = [bucket.acquire(3) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # One should succeed, others should fail
        assert sum(results) == 1
        assert bucket._tokens == 2

    async def test_zero_tokens_acquire(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=1)
        assert await bucket.acquire(0) is True
        assert bucket._tokens == 10

    async def test_zero_tokens_wait(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=1)
        assert await bucket.wait_for_tokens(0) is True
        assert bucket._tokens == 10

    async def test_negative_tokens_acquire(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=1)
        assert await bucket.acquire(-1) is True
        assert bucket._tokens == 11

    async def test_negative_tokens_wait(self):
        bucket = AsyncTokenBucket(capacity=10, refill_rate=1)
        assert await bucket.wait_for_tokens(-1) is True
        assert bucket._tokens == 11