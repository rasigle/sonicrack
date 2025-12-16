"""Audio engine for real-time synthesis and playback."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from PyQt6 import QtCore

from src.constants import DEFAULT_SAMPLE_RATE

if TYPE_CHECKING:
    from src.gui.core.module import AudioModule

logger = logging.getLogger(__name__)


class AudioEngine(QtCore.QObject):
    """Audio engine coordinates module processing.

    This is ONLY a coordinator - it processes modules in topological order.
    The Output module handles all audio I/O (sounddevice streams, buffers, etc).
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, buffer_size: int = 2048):
        """Initialize the audio engine."""
        super().__init__()

        self.sample_rate: float = sample_rate
        self.buffer_size: float = buffer_size

        self.modules = []
        self.connections = []

    def add_module(self, mod):
        """Add a module to the engine.

        Args:
            mod: Module to add
        """
        self.modules.append(mod)

    def process(self):
        """Process all modules in topological order.

        This is called by the Output module's audio callback.
        """
        ordered = self._build_graph()
        for module in ordered:
            module.process()

    def _build_graph(self) -> list[AudioModule]:
        """Build a topologically ordered list of audio components."""
        graph = defaultdict(set)
        indegree = defaultdict(int)

        for m in self.modules:
            indegree[m] = 0

        for src, outp, dst, inp in self.connections:
            graph[src].add(dst)
            indegree[dst] += 1

        # topological sort
        queue = deque([m for m in self.modules if indegree[m] == 0])
        order = []
        while queue:
            m = queue.popleft()
            order.append(m)
            for nbr in graph[m]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    queue.append(nbr)

        return order
