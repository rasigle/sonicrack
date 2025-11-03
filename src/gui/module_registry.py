"""Library of module types for the modular synth.

This file contains concrete implementations of various audio modules
that can be placed on the patch canvas.
"""

from src.gui.modules.clipper_simple import ClipperModule
from src.gui.modules.envelope_adsr import ADSRModule
from src.gui.modules.lfo import LFOModule
from src.gui.modules.mixer import MixerModule
from src.gui.modules.oscillator import OscillatorModule
from src.gui.modules.output import OutputModule
from src.gui.modules.pan_mod import PannerModule
from src.gui.modules.pan_simple import SimplePannerModule
from src.gui.modules.volume_mod import VolumeModule
from src.gui.modules.volume_simple import SimpleVolumeModule

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
