"""Audio buffer pool for reducing garbage collection pressure.

This module provides a thread-safe buffer pool that reuses pre-allocated
numpy arrays for audio processing, significantly reducing memory allocations
and GC pauses during real-time audio rendering.

Usage:
    pool = BufferPool(buffer_size=512, pool_size=32)

    # Acquire a buffer
    buffer = pool.acquire()

    # Use the buffer
    buffer[:] = process_audio(buffer)

    # Release back to pool
    pool.release(buffer)

    # Or use context manager (recommended)
    with pool.acquire_context() as buffer:
        buffer[:] = process_audio(buffer)
        # Auto-released on exit
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np


class BufferPool:
    """Thread-safe pool of reusable audio buffers.

    Reduces garbage collection pressure by maintaining a pool of pre-allocated
    numpy arrays that can be acquired, used, and returned for reuse.

    Attributes:
        buffer_size: Size of each buffer in samples
        pool_size: Maximum number of buffers to pool
    """

    def __init__(
        self,
        buffer_size: int,
        pool_size: int = 32,
        dtype: np.dtype = np.float32,
    ):
        """Initialize buffer pool.

        Args:
            buffer_size: Number of samples per buffer
            pool_size: Maximum number of pooled buffers
            dtype: Data type for buffers (default: float32)
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")

        self._buffer_size = buffer_size
        self._pool_size = pool_size
        self._dtype = dtype

        # Pre-allocate all buffers
        self._pool: list[np.ndarray] = []
        self._available: set[int] = set()
        self._lock = threading.Lock()

        # Create initial pool
        for i in range(pool_size):
            buffer = np.zeros(buffer_size, dtype=dtype)
            self._pool.append(buffer)
            self._available.add(i)

        # Statistics
        self._total_acquires = 0
        self._total_releases = 0
        self._peak_usage = 0
        self._fallback_allocations = 0

    def acquire(self) -> np.ndarray:
        """Acquire a buffer from the pool.

        Returns a zero-filled buffer ready for use. If the pool is empty,
        allocates a new buffer (tracked as fallback allocation).

        Returns:
            numpy array of shape (buffer_size,) and dtype
        """
        with self._lock:
            self._total_acquires += 1

            if self._available:
                # Get buffer from pool
                idx = self._available.pop()
                buffer = self._pool[idx]
                buffer.fill(0)  # Clear for safety

                # Track usage
                current_usage = self._pool_size - len(self._available)
                self._peak_usage = max(self._peak_usage, current_usage)

                return buffer

            # Pool exhausted - allocate new buffer
            self._fallback_allocations += 1
            return np.zeros(self._buffer_size, dtype=self._dtype)

    def release(self, buffer: np.ndarray) -> bool:
        """Return a buffer to the pool.

        Args:
            buffer: Buffer to return (must be from this pool)

        Returns:
            True if buffer was returned to pool, False if not recognized
        """
        with self._lock:
            self._total_releases += 1

            # Find buffer in pool
            for i, pooled in enumerate(self._pool):
                if pooled is buffer:
                    self._available.add(i)
                    return True

            # Buffer not from this pool (fallback allocation)
            return False

    @contextmanager
    def acquire_context(self) -> Generator[np.ndarray, None, None]:
        """Context manager for automatic buffer acquire/release.

        Usage:
            with pool.acquire_context() as buffer:
                buffer[:] = process_audio(buffer)
                # Automatically released on exit

        Yields:
            numpy array buffer
        """
        buffer = self.acquire()
        try:
            yield buffer
        finally:
            self.release(buffer)

    def get_stats(self) -> dict[str, int | float]:
        """Get pool statistics.

        Returns:
            Dictionary containing:
                - total_acquires: Total acquire() calls
                - total_releases: Total release() calls
                - peak_usage: Maximum simultaneous buffers in use
                - fallback_allocations: Times pool was exhausted
                - available: Currently available buffers
                - utilization: Peak usage as percentage of pool size
        """
        with self._lock:
            available = len(self._available)
            utilization = (self._peak_usage / self._pool_size) * 100

            return {
                "total_acquires": self._total_acquires,
                "total_releases": self._total_releases,
                "peak_usage": self._peak_usage,
                "fallback_allocations": self._fallback_allocations,
                "available": available,
                "utilization": utilization,
                "pool_size": self._pool_size,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._reset()

    def clear(self) -> None:
        """Clear the pool and release all buffers.

        Warning: This invalidates any buffers currently acquired from the pool.
        Only call when no buffers are in use.
        """
        with self._lock:
            self._pool.clear()
            self._available.clear()

            # Recreate pool
            for i in range(self._pool_size):
                buffer = np.zeros(self._buffer_size, dtype=self._dtype)
                self._pool.append(buffer)
                self._available.add(i)

            # Reset stats (inline to avoid deadlock with reset_stats())
            self._reset()

    def _reset(self):
        self._total_acquires = 0
        self._total_releases = 0
        self._peak_usage = 0
        self._fallback_allocations = 0

    @property
    def buffer_size(self) -> int:
        """Get buffer size in samples."""
        return self._buffer_size

    @property
    def pool_size(self) -> int:
        """Get maximum pool size."""
        return self._pool_size

    @property
    def available_count(self) -> int:
        """Get number of currently available buffers."""
        with self._lock:
            return len(self._available)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"BufferPool(size={self._buffer_size}, "
            f"pool={self._pool_size}, "
            f"available={stats['available']}, "
            f"peak_usage={stats['peak_usage']})"
        )


class MultiSizeBufferPool:
    """Buffer pool supporting multiple buffer sizes.

    Maintains separate pools for different buffer sizes commonly used
    in audio processing (e.g., 128, 256, 512, 1024 samples).

    Usage:
        pool = MultiSizeBufferPool(pool_size_per_size=16)
        buffer = pool.acquire(512)  # Get 512-sample buffer
        pool.release(buffer, 512)
    """

    def __init__(
        self,
        buffer_sizes: list[int] | None = None,
        pool_size_per_size: int = 16,
        dtype: np.dtype = np.float32,
    ):
        """Initialize multi-size buffer pool.

        Args:
            buffer_sizes: List of buffer sizes to support
                         (default: [64, 128, 256, 512, 1024, 2048])
            pool_size_per_size: Pool size for each buffer size
            dtype: Data type for buffers
        """
        if buffer_sizes is None:
            buffer_sizes = [64, 128, 256, 512, 1024, 2048]

        self._pools: dict[int, BufferPool] = {}
        for size in buffer_sizes:
            self._pools[size] = BufferPool(size, pool_size_per_size, dtype)

        self._lock = threading.Lock()

    def acquire(self, size: int) -> np.ndarray:
        """Acquire buffer of specified size.

        Args:
            size: Buffer size in samples

        Returns:
            numpy array of requested size
        """
        if size in self._pools:
            return self._pools[size].acquire()

        # Size not in pool - allocate directly
        return np.zeros(size, dtype=self._pools[list(self._pools.keys())[0]]._dtype)

    def release(self, buffer: np.ndarray, size: int) -> bool:
        """Release buffer back to pool.

        Args:
            buffer: Buffer to release
            size: Size of buffer (must match original acquire size)

        Returns:
            True if returned to pool, False otherwise
        """
        if size in self._pools:
            return self._pools[size].release(buffer)
        return False

    @contextmanager
    def acquire_context(self, size: int) -> Generator[np.ndarray, None, None]:
        """Context manager for automatic buffer acquire/release.

        Args:
            size: Buffer size in samples

        Yields:
            numpy array buffer
        """
        buffer = self.acquire(size)
        try:
            yield buffer
        finally:
            self.release(buffer, size)

    def get_stats(self) -> dict[int, dict[str, int | float]]:
        """Get statistics for all pools.

        Returns:
            Dictionary mapping buffer size to pool statistics
        """
        with self._lock:
            return {size: pool.get_stats() for size, pool in self._pools.items()}

    def reset_stats(self) -> None:
        """Reset statistics for all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.reset_stats()

    def clear(self) -> None:
        """Clear all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()


# Global buffer pool instance (can be used across modules)
_global_pool: MultiSizeBufferPool | None = None


def get_global_pool() -> MultiSizeBufferPool:
    """Get or create the global buffer pool instance.

    Returns:
        Global MultiSizeBufferPool instance
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = MultiSizeBufferPool()
    return _global_pool


def acquire_buffer(size: int) -> np.ndarray:
    """Acquire buffer from global pool.

    Args:
        size: Buffer size in samples

    Returns:
        numpy array buffer
    """
    return get_global_pool().acquire(size)


def release_buffer(buffer: np.ndarray, size: int) -> bool:
    """Release buffer to global pool.

    Args:
        buffer: Buffer to release
        size: Size of buffer

    Returns:
        True if returned to pool
    """
    return get_global_pool().release(buffer, size)


@contextmanager
def acquire_buffer_context(size: int) -> Generator[np.ndarray, None, None]:
    """Context manager for global pool buffer.

    Args:
        size: Buffer size in samples

    Yields:
        numpy array buffer
    """
    buffer = acquire_buffer(size)
    try:
        yield buffer
    finally:
        release_buffer(buffer, size)
