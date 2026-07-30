"""Shared MIDI note-on trigger pulse helper.

Implementation lives in ``soniclab.midi_io``; this module re-exports the public
API for existing SonicRack import paths.
"""

from __future__ import annotations

from soniclab.midi_io import MIDITriggerOutput

__all__ = ["MIDITriggerOutput"]
