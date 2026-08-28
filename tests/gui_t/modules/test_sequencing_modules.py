"""GUI runtime coverage for sequencing modules."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from sonicrack.gui.modules.sequencing.accent import AccentModule
from sonicrack.gui.modules.sequencing.arpeggiator import ArpeggiatorModule, _parse_notes
from sonicrack.gui.modules.sequencing.behringer_182 import Behringer182Module
from sonicrack.gui.modules.sequencing.clock import ClockModule
from sonicrack.gui.modules.sequencing.slide import SlideModule
from sonicrack.gui.modules.sequencing.step_sequencer import StepSequencerModule
from sonicrack.gui.widgets import ImagePushButton
from sonicrack.patching.registry import discover_modules, get_registry
from tests.gui_t.helpers import connect_signal as _connect_signal


def test_sequencing_modules_are_discoverable(qapp: Any):
    del qapp
    discover_modules("sonicrack.gui.modules", recursive=True)
    registered = get_registry().list_modules()

    assert "Clock" in registered
    assert "Step Sequencer" in registered
    assert "Arpeggiator" in registered
    assert "Slide" in registered
    assert "Accent" in registered
    assert "Behringer 182" in registered
    assert "Acid Filter" in registered
    assert "TB-303 Voice" in registered


def test_clock_module_writes_pulses(qapp: Any):
    del qapp
    module = ClockModule()

    module.process_runtime(8, {"bpm": 60.0, "division": "1/4", "swing": 0.0})

    output = np.asarray(module.clock_port.value)
    assert output.shape == (8,)
    assert output[0] == 1.0
    # Clock is a trigger stream (cable color / compatibility), not a gate.
    from sonicrack.patching.port import PortSignal

    assert module.clock_port.signal == PortSignal.TRIGGER


def test_clock_module_stretches_triggers_for_gate_consumers(qapp: Any):
    """Single-sample StepClock edges become multi-sample trigger pulses."""
    del qapp
    module = ClockModule()
    # Force a known sample rate on the clock component for deterministic width.
    module.component.sample_rate = 1000.0
    module.component.reset()
    module._pulse_hold = 0
    module._previous_level = 0.0

    # 60 BPM, 1/4 → 1 step per second → 1000 samples/step at sr=1000.
    # Min trigger is 2 ms → 2 samples; cap is step_samples - 1.
    module.process_runtime(
        16, {"bpm": 60.0, "division": "1/4", "swing": 0.0, "running": True}
    )
    output = np.asarray(module.clock_port.value)
    assert output[0] == 1.0
    assert output[1] == 1.0  # stretched beyond a single sample
    # Still returns low before the next step so sequencers see rising edges.
    assert np.any(output[2:] == 0.0)


def test_clock_can_trigger_adsr_gate(qapp: Any):
    del qapp
    from sonicrack.gui.modules.modulated_source.envelope_adsr import ADSRModule

    clock = ClockModule()
    adsr = ADSRModule()
    clock.clock_port.connect(adsr.gate_input)

    n = 512
    clock.process_runtime(
        n, {"bpm": 120.0, "division": "1/4", "swing": 0.0, "running": True}
    )
    # Without a render context, the ADSR reads the connected port's last write.
    adsr.process_runtime(
        n,
        {
            "attack_duration": 0.01,
            "decay_duration": 0.05,
            "sustain_level": 0.7,
            "release_duration": 0.1,
        },
    )
    out = np.asarray(adsr.out_port.value)
    assert float(out.max()) > 0.05


def test_step_sequencer_module_writes_all_cv_outputs(qapp: Any):
    del qapp
    module = StepSequencerModule()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "notes": "36,48",
            "accents": "1,0",
            "slides": "0,1",
            "gate_length": 1.0,
            "bpm": 120.0,
            "division": "1/16",
        },
    )

    assert np.asarray(module.freq_port.value).shape == (4,)
    np.testing.assert_allclose(module.freq_port.value, [-2.0, -2.0, -1.0, -1.0])
    np.testing.assert_allclose(module.gate_port.value, [1, 1, 1, 1])
    np.testing.assert_allclose(module.accent_port.value, [1, 1, 0, 0])
    np.testing.assert_allclose(module.slide_port.value, [0, 0, 1, 1])


def test_step_sequencer_advances_once_per_wide_clock_pulse(qapp: Any):
    """Clock module emits multi-sample triggers; sequencer must advance once each."""
    del qapp
    module = StepSequencerModule()
    # Wide pulse (as from ensure_min_pulse_width), then idle, then another pulse.
    clock_out = _connect_signal(
        module.clock_input,
        np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    params = {
        "notes": "36,48,60,72",
        "accents": "0,0,0,0",
        "slides": "0,0,0,0",
        "gate_length": 1.0,
        "bpm": 120.0,
        "division": "1/16",
    }
    module.process_runtime(8, params)
    # First wide pulse → one advance (step 0 = MIDI 36 → -2.0 V/oct).
    assert module.component._active_step == 0
    np.testing.assert_allclose(module.freq_port.value, np.full(8, -2.0))

    # Second buffer: still low (no advance), then rising edge mid-buffer.
    clock_out.write(
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    module.process_runtime(8, params)
    assert module.component._active_step == 1
    # Samples before edge stay on step 0; after edge on step 1 (MIDI 48 → -1.0).
    freq = np.asarray(module.freq_port.value)
    np.testing.assert_allclose(freq[:2], [-2.0, -2.0])
    np.testing.assert_allclose(freq[2:], np.full(6, -1.0))


def test_clock_module_drives_step_sequencer_through_pattern(qapp: Any):
    """End-to-end: Clock → Step Sequencer advances through successive steps."""
    del qapp
    clock = ClockModule()
    seq = StepSequencerModule()
    clock.clock_port.connect(seq.clock_input)

    # Fast step rate so several steps fit in a short render.
    clock.component.sample_rate = 1000.0
    clock.component.reset()
    clock._pulse_hold = 0
    clock._previous_level = 0.0
    seq.component.sample_rate = 1000.0
    seq.component.clock.sample_rate = 1000.0
    seq.component.reset()
    seq._previous_clock = 0.0

    steps_seen: list[int] = []
    n = 64
    # 60 BPM 1/4 → 1 step/sec → 1000 samples/step; with n=64 we need many buffers.
    # Use 240 BPM 1/4 → 4 steps/sec → 250 samples/step → ~1 step every 4 buffers.
    for _ in range(24):
        clock.process_runtime(
            n, {"bpm": 240.0, "division": "1/4", "swing": 0.0, "running": True}
        )
        seq.process_runtime(
            n,
            {
                "notes": "36,48,60,72,37,49,61,73",
                "accents": "0,0,0,0,0,0,0,0",
                "slides": "0,0,0,0,0,0,0,0",
                "gate_length": 1.0,
                "bpm": 120.0,
                "division": "1/16",
            },
        )
        steps_seen.append(int(seq.component._active_step))

    # Should walk multiple distinct steps rather than sticking on one.
    unique = set(steps_seen)
    assert len(unique) >= 4, f"expected multi-step advance, saw {steps_seen}"
    # Consecutive clock ticks should advance sequentially (mod 8).
    advances = [s for i, s in enumerate(steps_seen) if i == 0 or s != steps_seen[i - 1]]
    assert len(advances) >= 2, f"expected at least two advances, saw {steps_seen}"
    for prev, cur in zip(advances[:-1], advances[1:], strict=True):
        assert cur == (prev + 1) % 8


def test_step_sequencer_gate_length_change_does_not_reset_step(qapp: Any):
    """Gate knob scrubbing must update live without reset() or UI work on audio path."""
    del qapp
    module = StepSequencerModule()
    clock_out = _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    base_params = {
        "notes": "36,48,60,72",
        "accents": "0,0,0,0",
        "slides": "0,0,0,0",
        "gate_length": 0.8,
        "bpm": 120.0,
        "division": "1/16",
    }
    module.process_runtime(4, base_params)
    assert module.component._active_step == 1

    # Advance further, then change only gate length — playhead must not jump back.
    clock_out.write(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    module.process_runtime(4, {**base_params, "gate_length": 0.8})
    step_before = int(module.component._active_step)
    assert step_before == 3

    clock_out.write(np.zeros(4, dtype=np.float32))
    module.process_runtime(4, {**base_params, "gate_length": 0.25})
    assert module.component._active_step == step_before
    assert all(
        abs(event.gate_length - 0.25) < 1e-9 for event in module.component.pattern
    )
    assert abs(module.component._current_event.gate_length - 0.25) < 1e-9


def test_step_sequencer_gate_length_affects_gate_output(qapp: Any):
    """Shorter gate_length should release the gate within a long step."""
    del qapp
    module = StepSequencerModule()
    module.component.sample_rate = 100.0
    module.component.clock.sample_rate = 100.0
    module.component.reset()
    module._previous_structure_key = None
    module._previous_gate_length = None
    module._previous_clock = 0.0

    # Internal clock: 60 BPM 1/4 → 1 step/sec → 100 samples/step at sr=100.
    params = {
        "notes": "36",
        "accents": "0",
        "slides": "0",
        "gate_length": 0.2,
        "bpm": 60.0,
        "division": "1/4",
    }
    module.process_runtime(100, params)
    gate = np.asarray(module.gate_port.value)
    # 0.2 * 100 samples → gate high for first 20 samples, then low.
    assert float(gate[0]) == 1.0
    assert float(gate[19]) == 1.0
    assert float(gate[20]) == 0.0
    assert float(gate[-1]) == 0.0


def test_step_sequencer_process_runtime_does_not_mutate_note_buttons(qapp: Any):
    """Audio path must not rewrite note button widgets (thread-safety)."""
    del qapp
    module = StepSequencerModule()
    labels_before = [button.text() for button in module.note_buttons]

    module.process_runtime(
        4,
        {
            "notes": "60,61,62,63,64,65,66,67",
            "accents": "0,0,0,0,0,0,0,0",
            "slides": "0,0,0,0,0,0,0,0",
            "gate_length": 0.5,
            "bpm": 120.0,
            "division": "1/16",
        },
    )

    assert [button.text() for button in module.note_buttons] == labels_before


def test_step_sequencer_step_toggles_follow_set_parameters(qapp: Any):
    del qapp
    module = StepSequencerModule()

    module.set_accents("0,1,0,1,0,1,0,1")
    module.set_slides("1,0,1,0,1,0,1,0")

    assert [button.isChecked() for button in module.accent_buttons] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert [led.is_on() for led in module.slide_leds] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_step_sequencer_step_toggles_update_parameters(qapp: Any):
    del qapp
    module = StepSequencerModule()

    module.accent_buttons[1].setChecked(True)
    module.slide_buttons[0].setChecked(True)

    assert module.get_accents() == "1,1,0,1,0,0,1,0"
    assert module.get_slides() == "1,0,1,0,0,0,1,0"
    assert module.accent_leds[1].is_on()
    assert module.slide_leds[0].is_on()


def test_step_sequencer_randomize_updates_pattern_and_toggles(qapp: Any):
    del qapp
    module = StepSequencerModule()
    original_notes = module.notes_edit.text()

    module.randomize_pattern(random.Random(42))

    notes = [
        item.strip() for item in module.notes_edit.text().split(",") if item.strip()
    ]
    accents = module.get_accents().split(",")
    slides = module.get_slides().split(",")

    assert len(notes) == module.step_toggle_count
    assert len(accents) == module.step_toggle_count
    assert len(slides) == module.step_toggle_count
    assert module.notes_edit.text() != original_notes
    assert all(item == "-" or item.lstrip("-").isdigit() for item in notes)
    assert all(flag in {"0", "1"} for flag in accents)
    assert all(flag in {"0", "1"} for flag in slides)
    assert [button.isChecked() for button in module.accent_buttons] == [
        flag == "1" for flag in accents
    ]
    assert [button.isChecked() for button in module.slide_buttons] == [
        flag == "1" for flag in slides
    ]

    # Rests should not carry accent or slide flags.
    for index, note in enumerate(notes):
        if note == "-":
            assert accents[index] == "0"
            assert slides[index] == "0"


def test_step_sequencer_randomize_button_triggers_pattern_change(qapp: Any):
    del qapp
    module = StepSequencerModule()
    module.notes_edit.setText("36,36,36,36,36,36,36,36")
    module.set_accents("0,0,0,0,0,0,0,0")
    module.set_slides("0,0,0,0,0,0,0,0")
    before = (
        module.notes_edit.text(),
        module.get_accents(),
        module.get_slides(),
    )

    module.randomize_button.click()

    after = (
        module.notes_edit.text(),
        module.get_accents(),
        module.get_slides(),
    )
    assert after != before
    assert module.randomize_button.text() == "Randomize"
    assert not hasattr(module, "accent_edit")
    assert not hasattr(module, "slide_edit")


def test_behringer_182_module_writes_dual_cv_and_gate_outputs(qapp: Any):
    del qapp
    module = Behringer182Module()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "cv_a": "0,0.5",
            "cv_b": "1,0.25",
            "gates": "1,0",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
        },
    )

    np.testing.assert_allclose(module.cv_a_port.value, [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(module.cv_b_port.value, [2.0, 2.0, 0.5, 0.5])
    np.testing.assert_allclose(module.gate_port.value, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(module.trigger_port.value, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(module.end_port.value, [0.0, 0.0, 0.0, 0.0])


def test_behringer_182_uses_compact_panel_controls(qapp: Any):
    del qapp
    module = Behringer182Module()

    assert isinstance(module.run_button, ImagePushButton)
    assert module.run_button.text() == "Stop"
    assert module.run_button.width() == 54
    assert all(knob.knob_size == 28 for knob in module.cv_a_knobs)
    assert all(knob.knob_size == 28 for knob in module.cv_b_knobs)
    assert all(knob.label == "" for knob in module.cv_a_knobs)
    assert all(knob.label == "" for knob in module.cv_b_knobs)
    assert all(
        knob.knob_size == 34
        for knob in (
            module.bpm_knob,
            module.gate_length_knob,
            module.range_a_knob,
            module.range_b_knob,
        )
    )
    assert len(module.step_leds) == 8
    assert len(module.gate_buttons) == 8
    assert module.get_gates() == "1,1,1,1,1,1,1,1"
    assert not hasattr(module, "gates_edit")
    assert module.randomize_button.text() == "Randomize"

    module.run_button.setChecked(False)

    assert module.run_button.text() == "Start"


def test_behringer_182_gate_toggles_update_parameters(qapp: Any):
    del qapp
    module = Behringer182Module()

    module.gate_buttons[0].setChecked(False)
    module.gate_buttons[2].setChecked(False)

    assert module.get_gates() == "0,1,0,1,1,1,1,1"

    module.set_gates("1,0,1,0,1,0,1,0")
    assert module.get_gates() == "1,0,1,0,1,0,1,0"
    assert [button.isChecked() for button in module.gate_buttons] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_behringer_182_randomize_updates_cv_and_gates(qapp: Any):
    del qapp
    module = Behringer182Module()
    before_a = [knob.get_value() for knob in module.cv_a_knobs]
    before_b = [knob.get_value() for knob in module.cv_b_knobs]
    before_gates = module.get_gates()

    module.randomize_pattern(random.Random(42))

    after_a = [knob.get_value() for knob in module.cv_a_knobs]
    after_b = [knob.get_value() for knob in module.cv_b_knobs]
    after_gates = module.get_gates()
    gate_flags = after_gates.split(",")

    assert after_a != before_a
    assert after_b != before_b
    assert after_gates != before_gates
    assert len(gate_flags) == module.step_count
    assert all(flag in {"0", "1"} for flag in gate_flags)
    assert any(flag == "1" for flag in gate_flags)
    assert all(0.0 <= value <= 1.0 for value in after_a)
    assert all(0.0 <= value <= 1.0 for value in after_b)
    assert [button.isChecked() for button in module.gate_buttons] == [
        flag == "1" for flag in gate_flags
    ]


def test_behringer_182_randomize_button_triggers_pattern_change(qapp: Any):
    del qapp
    module = Behringer182Module()
    for knob in module.cv_a_knobs:
        knob.set_value(0.0)
    for knob in module.cv_b_knobs:
        knob.set_value(0.0)
    module.set_gates("0,0,0,0,0,0,0,0")
    before = (
        [knob.get_value() for knob in module.cv_a_knobs],
        [knob.get_value() for knob in module.cv_b_knobs],
        module.get_gates(),
    )

    module.randomize_button.click()

    after = (
        [knob.get_value() for knob in module.cv_a_knobs],
        [knob.get_value() for knob in module.cv_b_knobs],
        module.get_gates(),
    )
    assert after != before
    assert module.randomize_button.text() == "Randomize"
    assert not hasattr(module, "gates_edit")


def test_behringer_182_step_leds_follow_active_step(qapp: Any):
    del qapp
    module = Behringer182Module()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": True,
        },
    )

    assert [led.is_on() for led in module.step_leds] == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_behringer_182_module_stops_gate_outputs(qapp: Any):
    del qapp
    module = Behringer182Module()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "cv_a_1": 0.25,
            "cv_b_1": 0.75,
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": False,
        },
    )

    np.testing.assert_allclose(module.cv_a_port.value, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(module.cv_b_port.value, [1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(module.gate_port.value, 0.0)
    np.testing.assert_allclose(module.trigger_port.value, 0.0)
    np.testing.assert_allclose(module.end_port.value, 0.0)


def test_behringer_182_cv_knob_changes_without_resetting_step(qapp: Any):
    del qapp
    module = Behringer182Module()
    clock_output = _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )

    module.process_runtime(
        3,
        {
            "cv_a_1": 0.0,
            "cv_a_2": 0.25,
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": True,
        },
    )

    clock_output.write(np.zeros(2, dtype=np.float32))
    module.process_runtime(
        2,
        {
            "cv_a_1": 0.0,
            "cv_a_2": 0.75,
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": True,
        },
    )

    np.testing.assert_allclose(module.cv_a_port.value, [3.0, 3.0])


def test_slide_module_processes_frequency_cv(qapp: Any):
    del qapp
    module = SlideModule()
    _connect_signal(
        module.freq_input,
        np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    _connect_signal(
        module.slide_input,
        np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )

    module.process_runtime(4, {"time": 1.0, "always_on": False})

    output = np.asarray(module.freq_output.value)
    assert output.shape == (4,)
    assert output[0] == 0.0
    assert 0.0 < output[1] < 1.0


def test_accent_module_writes_scaled_cv_outputs(qapp: Any):
    del qapp
    module = AccentModule()
    # soniclab AccentProcessor rises over ~2 ms on a rising accent so a short
    # buffer only captures the start of the ramp. Render long enough to settle.
    n = 256
    accent = np.zeros(n, dtype=np.float32)
    accent[:128] = 1.0
    _connect_signal(module.accent_input, accent)

    module.process_runtime(
        n,
        {
            "amount": 1.0,
            "decay": 0.0,
            "amp_depth": 0.25,
            "cutoff_depth": 0.5,
            "envelope_depth": 0.75,
        },
    )

    amp = np.asarray(module.amp_port.value, dtype=np.float32)
    cutoff = np.asarray(module.cutoff_port.value, dtype=np.float32)
    env = np.asarray(module.env_port.value, dtype=np.float32)
    assert amp.shape == (n,)
    # After the ~2 ms accent rise (≈88 samples @ 44.1 kHz), depths settle.
    np.testing.assert_allclose(amp[100:120], 0.25, atol=1e-2)
    np.testing.assert_allclose(cutoff[100:120], 0.5, atol=1e-2)
    np.testing.assert_allclose(env[100:120], 0.75, atol=1e-2)
    # After accent falls (and decay=0), outputs return toward zero.
    assert float(amp[-1]) < 0.05
    assert float(cutoff[-1]) < 0.05
    assert float(env[-1]) < 0.05


def test_parse_notes_skips_rests_and_clamps() -> None:
    assert _parse_notes("60, 64, -, 67, rest, 200") == [60, 64, 67, 127]
    assert _parse_notes("") == []


def test_arpeggiator_module_writes_cv_outputs(qapp: Any):
    del qapp
    module = ArpeggiatorModule()
    # External clock: two steps across the buffer.
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "notes": "60,72",
            "pattern": "Up",
            "octaves": 1,
            "gate_length": 1.0,
            "transpose": 0,
            "swing": 0.0,
            "bpm": 120.0,
            "division": "1/16",
            "latch": False,
            "running": True,
        },
    )

    freq = np.asarray(module.freq_port.value)
    gate = np.asarray(module.gate_port.value)
    trig = np.asarray(module.trigger_port.value)
    vel = np.asarray(module.vel_port.value)

    assert freq.shape == (4,)
    # MIDI 60 → 0V, MIDI 72 → +1V (1V/oct, C4 reference).
    np.testing.assert_allclose(freq, [0.0, 0.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(gate, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(trig, [1.0, 0.0, 1.0, 0.0])
    assert np.all(vel >= 0.0)


def test_arpeggiator_apply_chord_updates_notes(qapp: Any):
    del qapp
    module = ArpeggiatorModule()

    text = module.apply_chord("A", 3, "Min7")

    assert text == "57,60,64,67"
    assert module.notes_edit.text() == text


def test_arpeggiator_pattern_and_octaves_expand_sequence(qapp: Any):
    del qapp
    module = ArpeggiatorModule()
    module.notes_edit.setText("60,64,67")
    module.set_octaves(2)
    module.pattern_combo.setCurrentText("Down")
    module._refresh_sequence_preview()

    # Two octaves of C-E-G expanded then reversed: G5 … C4.
    notes = module.component.sequence_notes
    assert notes == [79, 76, 72, 67, 64, 60]


def test_arpeggiator_parameters_round_trip(qapp: Any):
    del qapp
    module = ArpeggiatorModule()
    module.set_parameters(
        {
            "notes": "48,55,60",
            "pattern": "Random",
            "octaves": 3,
            "gate_length": 0.33,
            "transpose": -5,
            "bpm": 140.0,
            "division": "1/8",
            "latch": True,
            "running": False,
            "chord": "Power",
            "root": "G",
            "root_octave": "2",
        }
    )
    params = module.get_parameters()

    assert params["notes"] == "48,55,60"
    assert params["pattern"] == "Random"
    assert params["octaves"] == 3
    assert abs(float(params["gate_length"]) - 0.33) < 1e-6
    assert params["transpose"] == -5
    assert params["division"] == "1/8"
    assert params["latch"] is True
    assert params["running"] is False
