"""Library of module types for the modular synth.

This file contains concrete implementations of various audio modules
that can be placed on the patch canvas.
"""

from gui.modules.clipper_simple import ClipperModule
from gui.modules.envelope_adsr import ADSRModule
from gui.modules.lfo import LFOModule
from gui.modules.mixer import MixerModule
from gui.modules.oscillator import OscillatorModule
from gui.modules.output import OutputModule
from gui.modules.pan_mod import PannerModule
from gui.modules.pan_simple import SimplePannerModule
from gui.modules.volume_mod import VolumeModule
from gui.modules.volume_simple import SimpleVolumeModule

# Module registry for easy instantiation
MODULE_REGISTRY = {
    "Oscillator": OscillatorModule,
    "LFO": LFOModule,
    "ADSR Envelope": ADSRModule,
    "Mixer": MixerModule,
    "Gain": SimpleVolumeModule,
    "Clipper": ClipperModule,
    "Volume (Mod)": VolumeModule,
    "Panner": SimplePannerModule,
    "Panner (Mod)": PannerModule,
    "Output": OutputModule,
}
