"""MIDI and related input modules for the modular synth GUI."""

from sonicrack.gui.modules.input.midi_input import MIDIInputModule
from sonicrack.gui.modules.input.midi_keyboard import MIDIKeyboardModule
from sonicrack.gui.modules.input.midi_poly_cv import MIDIPolyCVModule
from sonicrack.gui.modules.input.midi_trigger import MIDITriggerOutput
from sonicrack.gui.modules.input.midi_worker_thread import MIDIWorkerThread

__all__ = [
    "MIDIInputModule",
    "MIDIKeyboardModule",
    "MIDIPolyCVModule",
    "MIDITriggerOutput",
    "MIDIWorkerThread",
]
