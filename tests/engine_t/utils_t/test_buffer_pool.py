"""Unit tests for buffer pool system."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.engine.utils.buffer_pool import (
    BufferPool,
    MultiSizeBufferPool,
    acquire_buffer,
    acquire_buffer_context,
    get_global_pool,
    release_buffer,
)


class TestBufferPool:
    """Test single-size buffer pool."""

    def test_init(self):
        """Test buffer pool initialization."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        assert pool.buffer_size == 512
        assert pool.pool_size == 4
        assert pool.available_count == 4

    def test_init_invalid_size(self):
        """Test initialization with invalid sizes."""
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            BufferPool(buffer_size=0)

        with pytest.raises(ValueError, match="pool_size must be positive"):
            BufferPool(buffer_size=512, pool_size=0)

    def test_acquire_release(self):
        """Test basic acquire and release."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        # Acquire buffer
        buffer = pool.acquire()
        assert buffer.shape == (512,)
        assert buffer.dtype == np.float32
        assert pool.available_count == 3

        # Release buffer
        result = pool.release(buffer)
        assert result is True
        assert pool.available_count == 4

    def test_buffer_reuse(self):
        """Test that released buffers are reused."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        # Acquire all buffers
        buffers = [pool.acquire() for _ in range(4)]
        assert pool.available_count == 0

        # Release all buffers
        for buf in buffers:
            pool.release(buf)
        assert pool.available_count == 4

        # Acquire again - should get same buffers back
        new_buffers = [pool.acquire() for _ in range(4)]
        assert set(id(b) for b in new_buffers) == set(id(b) for b in buffers)

    def test_buffer_cleared_on_acquire(self):
        """Test that buffers are cleared when acquired."""
        pool = BufferPool(buffer_size=512, pool_size=2)

        # Acquire, modify, release
        buffer = pool.acquire()
        buffer[:] = 1.0
        pool.release(buffer)

        # Re-acquire - should be cleared
        new_buffer = pool.acquire()
        assert np.all(new_buffer == 0.0)

    def test_pool_exhaustion(self):
        """Test fallback allocation when pool is exhausted."""
        pool = BufferPool(buffer_size=512, pool_size=2)

        # Acquire all pooled buffers
        buffer1 = pool.acquire()
        buffer2 = pool.acquire()
        assert pool.available_count == 0

        # Acquire beyond pool size - should allocate new
        buffer3 = pool.acquire()
        assert buffer3.shape == (512,)
        assert pool.available_count == 0

        stats = pool.get_stats()
        assert stats["fallback_allocations"] == 1

        # Release fallback buffer - won't return to pool
        result = pool.release(buffer3)
        assert result is False
        assert pool.available_count == 0

        # Release pooled buffers
        pool.release(buffer1)
        pool.release(buffer2)
        assert pool.available_count == 2

    def test_context_manager(self):
        """Test context manager for automatic release."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        with pool.acquire_context() as buffer:
            assert buffer.shape == (512,)
            assert pool.available_count == 3

        # Buffer should be released
        assert pool.available_count == 4

    def test_context_manager_exception(self):
        """Test context manager releases buffer even on exception."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        try:
            with pool.acquire_context():
                assert pool.available_count == 3
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Buffer should still be released
        assert pool.available_count == 4

    def test_statistics(self):
        """Test statistics tracking."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        # Acquire and release
        buffer1 = pool.acquire()
        _ = pool.acquire()
        pool.release(buffer1)

        stats = pool.get_stats()
        assert stats["total_acquires"] == 2
        assert stats["total_releases"] == 1
        assert stats["peak_usage"] == 2
        assert stats["fallback_allocations"] == 0
        assert stats["available"] == 3

        # Reset stats
        pool.reset_stats()
        stats = pool.get_stats()
        assert stats["total_acquires"] == 0
        assert stats["total_releases"] == 0

    def test_clear(self):
        """Test clearing the pool."""
        pool = BufferPool(buffer_size=512, pool_size=4)

        # Acquire some buffers
        buffer = pool.acquire()
        buffer[:] = 1.0

        # Clear pool
        pool.clear()

        assert pool.available_count == 4
        stats = pool.get_stats()
        assert stats["total_acquires"] == 0

    def test_thread_safety(self):
        """Test thread-safe operations."""
        pool = BufferPool(buffer_size=512, pool_size=16)

        acquired_buffers = []
        lock = threading.Lock()

        def worker():
            for _ in range(10):
                buffer = pool.acquire()
                time.sleep(0.001)  # Simulate work
                with lock:
                    acquired_buffers.append(buffer)
                pool.release(buffer)

        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check all operations completed
        stats = pool.get_stats()
        assert stats["total_acquires"] == 40
        assert stats["total_releases"] == 40
        assert len(acquired_buffers) == 40


class TestMultiSizeBufferPool:
    """Test multi-size buffer pool."""

    def test_init_default(self):
        """Test default initialization."""
        pool = MultiSizeBufferPool()

        # Should have default sizes
        stats = pool.get_stats()
        assert 512 in stats
        assert 1024 in stats

    def test_init_custom_sizes(self):
        """Test initialization with custom sizes."""
        pool = MultiSizeBufferPool(
            buffer_sizes=[128, 256, 512],
            pool_size_per_size=8,
        )

        stats = pool.get_stats()
        assert len(stats) == 3
        assert all(s["pool_size"] == 8 for s in stats.values())

    def test_acquire_release_different_sizes(self):
        """Test acquiring and releasing different sizes."""
        pool = MultiSizeBufferPool()

        # Acquire different sizes
        buffer_512 = pool.acquire(512)
        buffer_1024 = pool.acquire(1024)

        assert buffer_512.shape == (512,)
        assert buffer_1024.shape == (1024,)

        # Release
        assert pool.release(buffer_512, 512) is True
        assert pool.release(buffer_1024, 1024) is True

    def test_acquire_unsupported_size(self):
        """Test acquiring unsupported size."""
        pool = MultiSizeBufferPool(buffer_sizes=[512, 1024])

        # Acquire unsupported size - should allocate directly
        buffer = pool.acquire(256)
        assert buffer.shape == (256,)

        # Release won't return to pool
        assert pool.release(buffer, 256) is False

    def test_context_manager(self):
        """Test context manager for multi-size pool."""
        pool = MultiSizeBufferPool()

        with pool.acquire_context(512) as buffer:
            assert buffer.shape == (512,)

    def test_reset_stats(self):
        """Test resetting statistics."""
        pool = MultiSizeBufferPool()

        # Use some buffers
        pool.acquire(512)
        pool.acquire(1024)

        # Reset
        pool.reset_stats()

        stats = pool.get_stats()
        for size_stats in stats.values():
            assert size_stats["total_acquires"] == 0


class TestGlobalPool:
    """Test global buffer pool functions."""

    def test_get_global_pool(self):
        """Test getting global pool instance."""
        pool1 = get_global_pool()
        pool2 = get_global_pool()

        # Should be same instance
        assert pool1 is pool2

    def test_acquire_release_buffer(self):
        """Test global acquire/release functions."""
        buffer = acquire_buffer(512)
        assert buffer.shape == (512,)

        result = release_buffer(buffer, 512)
        assert result is True

    def test_acquire_buffer_context(self):
        """Test global context manager."""
        with acquire_buffer_context(512) as buffer:
            assert buffer.shape == (512,)
            buffer[:] = 1.0

        # Buffer should be released
        pool = get_global_pool()
        stats = pool.get_stats()
        assert stats[512]["total_releases"] > 0


class TestBufferPoolIntegration:
    """Integration tests for buffer pool."""

    def test_typical_audio_workflow(self):
        """Test typical audio processing workflow."""
        pool = BufferPool(buffer_size=512, pool_size=8)

        # Simulate multiple render cycles
        for _ in range(100):
            with pool.acquire_context() as buffer:
                # Simulate audio processing
                buffer[:] = np.random.randn(512).astype(np.float32)
                _ = np.sum(buffer)

        # Check statistics
        stats = pool.get_stats()
        assert stats["total_acquires"] == 100
        assert stats["total_releases"] == 100
        assert stats["fallback_allocations"] == 0
        assert stats["available"] == 8

    def test_parallel_processing(self):
        """Test parallel audio processing."""
        pool = BufferPool(buffer_size=512, pool_size=32)

        def process_audio():
            for _ in range(50):
                with pool.acquire_context() as buffer:
                    buffer[:] = np.sin(np.linspace(0, np.pi, 512)).astype(np.float32)
                    _ = np.mean(buffer)

        # Run multiple threads
        threads = [threading.Thread(target=process_audio) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check all operations completed correctly
        stats = pool.get_stats()
        assert stats["total_acquires"] == 400  # 8 threads * 50 iterations
        assert stats["total_releases"] == 400
        assert stats["available"] == 32

    def test_memory_efficiency(self):
        """Test that pool reduces allocations."""
        pool = BufferPool(buffer_size=512, pool_size=16)

        # Many acquire/release cycles
        for _ in range(1000):
            buffer = pool.acquire()
            pool.release(buffer)

        stats = pool.get_stats()
        # Should have zero fallback allocations
        assert stats["fallback_allocations"] == 0
        # Peak usage should be 1 (only one buffer at a time)
        assert stats["peak_usage"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
