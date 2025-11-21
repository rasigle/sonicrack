"""Test MIDI-triggered synthesis workflow with new modules.

This test verifies that the MIDI Input → Oscillator (Freq Mod) → ADSR workflow
works correctly.
"""

import numpy as np

from engine.io.midi import (
    MIDIToCV,
    NoteOnMessage,
    NoteOffMessage,
    CVFrequencyOutput,
    CVGateOutput,
)
from engine import SineOscillator, ADSREnvelope, ModulatedOscillator
from engine.modulator import GateTriggeredADSR


class TestMIDIWorkflow:
    """Test complete MIDI-to-synthesis workflows."""

    def test_midi_to_modulated_oscillator(self):
        """Test MIDI Input Freq → Oscillator (Freq Mod)."""
        # Create MIDI to CV converter
        cv = MIDIToCV()
        freq_output = CVFrequencyOutput(cv)

        # Create oscillator with frequency modulation
        base_osc = SineOscillator(440)  # Base frequency (ignored when modulated)

        # Frequency modulation function: use CV frequency directly
        def freq_mod_func(base_freq, cv_freq):
            return cv_freq

        mod_osc = ModulatedOscillator(base_osc, freq_output, freq_mod=freq_mod_func)

        # Trigger MIDI note C4 (261.63 Hz)
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))

        # Generate samples
        samples = mod_osc.get_samples(100)

        assert len(samples) == 100
        # Oscillator should be playing at C4 frequency, not base 440 Hz
        # (We'd need to check phase progression to verify exact frequency)

    def test_midi_gate_to_adsr(self):
        """Test MIDI Input Gate → ADSR Envelope."""
        # Create MIDI to CV converter
        cv = MIDIToCV()
        gate_output = CVGateOutput(cv)

        # Create ADSR envelope
        adsr = ADSREnvelope(
            attack_duration=0.1,
            decay_duration=0.1,
            sustain_level=0.7,
            release_duration=0.2,
        )

        # Wrap with gate trigger
        gate_adsr = GateTriggeredADSR(adsr, gate_output)

        # Initially gate is 0
        assert cv.gate == 0.0
        gate_adsr.get_samples(10)
        # Should be 0 or very small (no note triggered)

        # Trigger note on
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        assert cv.gate == 1.0

        # Generate samples - should be in attack phase
        samples2 = gate_adsr.get_samples(100)
        assert len(samples2) == 100
        # Envelope should be rising (attack phase)
        assert samples2[0] < samples2[-1]

        # Trigger note off
        cv.process_message(NoteOffMessage(0.1, 0, 60))
        assert cv.gate == 0.0

        # Generate samples - should be in release phase
        samples3 = gate_adsr.get_samples(100)
        # Envelope should be falling (release phase)
        assert samples3[0] > samples3[-1]

    def test_complete_midi_synth_workflow(self):
        """Test complete workflow: MIDI → Freq Mod Osc → Gate ADSR → Volume."""
        # Create MIDI to CV converter
        cv = MIDIToCV()
        freq_output = CVFrequencyOutput(cv)
        gate_output = CVGateOutput(cv)

        # Create oscillator with frequency modulation
        base_osc = SineOscillator(440)

        def freq_mod_func(base, cv_freq):
            return cv_freq

        mod_osc = ModulatedOscillator(base_osc, freq_output, freq_mod=freq_mod_func)

        # Create ADSR with gate trigger
        adsr = ADSREnvelope(
            attack_duration=0.01,
            decay_duration=0.05,
            sustain_level=0.7,
            release_duration=0.1,
        )
        gate_adsr = GateTriggeredADSR(adsr, gate_output)

        # Initially no sound
        osc_samples = mod_osc.get_samples(10)
        env_samples = gate_adsr.get_samples(10)
        assert np.mean(np.abs(env_samples)) < 0.1  # Envelope should be quiet

        # Trigger note
        cv.process_message(NoteOnMessage(0.0, 0, 69, 100))  # A4 = 440 Hz

        # Generate audio
        osc_samples = mod_osc.get_samples(1000)
        env_samples = gate_adsr.get_samples(1000)

        # Multiply oscillator by envelope (this is what Volume (Mod) does)
        output = osc_samples * env_samples

        assert len(output) == 1000
        # Should have sound (envelope is active)
        assert np.mean(np.abs(output)) > 0.01

        # Release note
        cv.process_message(NoteOffMessage(0.1, 0, 69))

        # Generate more audio - should fade out
        osc_samples = mod_osc.get_samples(1000)
        env_samples = gate_adsr.get_samples(1000)
        output = osc_samples * env_samples

        # Envelope should be decreasing
        first_half = np.mean(np.abs(output[:500]))
        second_half = np.mean(np.abs(output[500:]))
        assert first_half > second_half  # Fading out
