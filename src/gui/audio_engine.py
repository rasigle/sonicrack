"""Audio engine for real-time synthesis and playback."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from PyQt6 import QtCore

from src.constants import DEFAULT_SAMPLE_RATE
from src.gui.audio_config import audio_config

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

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._process_cycle)
        self._timer.setInterval(max(1, int(self.buffer_size / self.sample_rate * 1000)))

        audio_config.add_sample_rate_listener(self._on_config_changed)
        audio_config.add_buffer_size_listener(self._on_config_changed)

    def add_module(self, mod):
        """Add a module to the engine.

        Args:
            mod: Module to add
        """
        self.modules.append(mod)

    def process(self, num_samples: int | None = None):
        """Process all modules in topological order."""
        if num_samples is None:
            num_samples = audio_config.buffer_size

        ordered = self._build_graph()
        for module in ordered:
            module.process(num_samples)

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

    def _on_config_changed(self, _):
        self._update_timer_interval()

    def _update_timer_interval(self):
        buffer = audio_config.buffer_size
        rate = audio_config.sample_rate
        interval_ms = max(1, int(buffer / rate * 1000))
        self._timer.setInterval(interval_ms)

    def _process_cycle(self):
        self.process(audio_config.buffer_size)

    def start(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()

    def set_sample_rate(self, value: int):
        self.sample_rate = value
        self._timer.setInterval(max(1, int(audio_config.buffer_size / value * 1000)))

    def set_buffer_size(self, value: int):
        self.buffer_size = value
        self._timer.setInterval(max(1, int(value / audio_config.sample_rate * 1000)))
