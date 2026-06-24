import contextlib
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

try:
    from audio_io import AudioOutput
except (ImportError, OSError):
    AudioOutput = None

from engine import (
    Oscillator,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)

CLASS_MAP = {
    "Sine": SineOscillator,
    "Saw": SawtoothOscillator,
    "Square": SquareOscillator,
    "Triangle": TriangleOscillator,
}

MIN_FREQ_HZ = 1.0
MAX_FREQ_HZ = 500.0
FREQ_SLIDER_STEPS = 500
AUDIO_SAMPLE_RATE = 44100
AUDIO_BUFFER_SIZE = 512


def _available_modes(oscillator_cls: type[Oscillator]) -> list[str]:
    get_available_modes = getattr(oscillator_cls, "get_available_modes", None)
    if get_available_modes is None:
        return []
    return list(get_available_modes())


def _slider_to_frequency(value: int) -> float:
    normalized = value / FREQ_SLIDER_STEPS
    return MIN_FREQ_HZ * ((MAX_FREQ_HZ / MIN_FREQ_HZ) ** normalized)


def _frequency_to_slider(frequency: float) -> int:
    frequency = min(max(frequency, MIN_FREQ_HZ), MAX_FREQ_HZ)
    normalized = np.log(frequency / MIN_FREQ_HZ) / np.log(MAX_FREQ_HZ / MIN_FREQ_HZ)
    return int(round(normalized * FREQ_SLIDER_STEPS))


class WaveformViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Oscillator Viewer")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)

        # Plot
        self.plot = pg.PlotWidget(title="Waveform")
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen=pg.mkPen(color=(30, 144, 255), width=2))
        layout.addWidget(self.plot)

        # Controls row
        ctrl = QtWidgets.QWidget()
        ctrl_l = QtWidgets.QHBoxLayout(ctrl)

        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(CLASS_MAP.keys())
        ctrl_l.addWidget(QtWidgets.QLabel("Wave:"))
        ctrl_l.addWidget(self.wave_combo)

        self.mode_combo = QtWidgets.QComboBox()
        ctrl_l.addWidget(QtWidgets.QLabel("Mode:"))
        ctrl_l.addWidget(self.mode_combo)

        ctrl_l.addWidget(QtWidgets.QLabel("Freq (Hz):"))

        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, FREQ_SLIDER_STEPS)
        self.freq_slider.setValue(_frequency_to_slider(110.0))
        self.freq_slider.setMinimumWidth(180)
        ctrl_l.addWidget(self.freq_slider)

        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(MIN_FREQ_HZ, MAX_FREQ_HZ)
        self.freq_spin.setDecimals(0)
        self.freq_spin.setSingleStep(1)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.setValue(110.0)
        self.freq_spin.setMinimumWidth(75)
        ctrl_l.addWidget(self.freq_spin)

        self.amp_spin = QtWidgets.QDial()
        self.amp_spin.setRange(0, 100)
        self.amp_spin.setValue(100)
        self.amp_spin.setSingleStep(1)
        ctrl_l.addWidget(QtWidgets.QLabel("Amp:"))
        ctrl_l.addWidget(self.amp_spin)

        self.phase_spin = QtWidgets.QDial()
        self.phase_spin.setRange(0, 360)
        self.phase_spin.setValue(0)
        ctrl_l.addWidget(QtWidgets.QLabel("Phase (°):"))
        ctrl_l.addWidget(self.phase_spin)

        self.sr_spin = QtWidgets.QSpinBox()
        self.sr_spin.setRange(100, 192000)
        self.sr_spin.setValue(1024)
        ctrl_l.addWidget(QtWidgets.QLabel("Sample Rate:"))
        ctrl_l.addWidget(self.sr_spin)

        self.points_spin = QtWidgets.QSpinBox()
        self.points_spin.setRange(64, 16384)
        self.points_spin.setValue(1024)
        ctrl_l.addWidget(QtWidgets.QLabel("Points:"))
        ctrl_l.addWidget(self.points_spin)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        ctrl_l.addWidget(self.play_btn)

        layout.addWidget(ctrl)

        # Internal state
        self._osc: Oscillator | None = None
        self._audio_osc: Oscillator | None = None
        self._audio_output = None
        self._x = None

        # Connections
        self.wave_combo.currentTextChanged.connect(self._change_waveform)
        self.mode_combo.currentTextChanged.connect(self._change_mode)
        # Update oscillator properties when widgets change
        for widget in (
            self.amp_spin,
            self.phase_spin,
            self.sr_spin,
            self.points_spin,
        ):
            widget.valueChanged.connect(self._update_osc_params)
        self.freq_slider.valueChanged.connect(self._update_frequency_from_slider)
        self.freq_spin.valueChanged.connect(self._update_frequency_from_spin)
        self.play_btn.toggled.connect(self._toggle_playback)

        # Timer for continuous redraw (non-advancing snapshot mode)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(30)  # ~33 FPS

        # Create oscillator once
        self._populate_mode_combo()
        self._create_initial_osc()
        self._update_x()  # initialize x axis and y-range

    def _current_oscillator_class(self) -> type[Oscillator]:
        cls = CLASS_MAP[self.wave_combo.currentText()]
        return cls

    def _current_mode(self) -> str | None:
        if self.mode_combo.count() == 0:
            return None
        return self.mode_combo.currentText()

    def _oscillator_kwargs(self, sample_rate: int | None = None) -> dict:
        freq = float(self.freq_spin.value())
        amp = float(self.amp_spin.value() / 100)
        phase = float(self.phase_spin.value())
        kwargs = {
            "frequency": freq,
            "amplitude": amp,
            "phase": phase,
            "sample_rate": int(sample_rate or self.sr_spin.value()),
        }
        mode = self._current_mode()
        if mode:
            kwargs["mode"] = mode
        return kwargs

    def _create_oscillator(self, sample_rate: int | None = None) -> Oscillator:
        cls = self._current_oscillator_class()
        return cls(**self._oscillator_kwargs(sample_rate))

    def _create_audio_oscillator(self) -> Oscillator:
        return self._create_oscillator(AUDIO_SAMPLE_RATE)

    def _replace_oscillator(self):
        osc = self._create_oscillator()
        with contextlib.suppress(TypeError):
            iter(osc)
        self._osc = osc

    def _replace_audio_oscillator(self):
        self._audio_osc = self._create_audio_oscillator()

    def _create_initial_osc(self):
        self._replace_oscillator()

    def _populate_mode_combo(self):
        cls = self._current_oscillator_class()
        current_mode = self._current_mode()
        modes = _available_modes(cls)

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(modes)
        if current_mode in modes:
            self.mode_combo.setCurrentText(current_mode)
        self.mode_combo.setEnabled(bool(modes))
        self.mode_combo.blockSignals(False)

    def _change_waveform(self, _):
        self._populate_mode_combo()
        self._replace_oscillator()
        self._replace_audio_oscillator_if_needed()

    def _change_mode(self, mode: str):
        if self._osc is None or not mode:
            return
        self._osc.mode = mode
        if self._audio_osc is not None:
            self._audio_osc.mode = mode

    def _update_frequency_from_slider(self, value: int):
        frequency = _slider_to_frequency(value)
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(frequency)
        self.freq_spin.blockSignals(False)
        self._update_osc_params()

    def _update_frequency_from_spin(self, value: float):
        self.freq_slider.blockSignals(True)
        self.freq_slider.setValue(_frequency_to_slider(value))
        self.freq_slider.blockSignals(False)
        self._update_osc_params()

    def _update_osc_params(self):
        if self._osc is None:
            return
        freq = float(self.freq_spin.value())
        amp = float(self.amp_spin.value() / 100)
        phase = float(self.phase_spin.value())
        sr = int(self.sr_spin.value())

        self._osc.frequency = freq
        self._osc.amplitude = amp
        self._osc.phase = phase
        self._osc.sample_rate = sr
        self._update_audio_osc_params(freq, amp, phase)

        # Update plot range and x axis
        self._update_x()

    def _update_audio_osc_params(self, freq: float, amp: float, phase: float):
        if self._audio_osc is None:
            return
        self._audio_osc.frequency = freq
        self._audio_osc.amplitude = amp
        self._audio_osc.phase = phase
        self._audio_osc.sample_rate = AUDIO_SAMPLE_RATE

    def _replace_audio_oscillator_if_needed(self):
        if self._audio_osc is not None or self._is_playing_audio():
            self._replace_audio_oscillator()

    def _is_playing_audio(self) -> bool:
        return bool(
            self._audio_output is not None
            and getattr(self._audio_output, "is_playing", False)
        )

    def _toggle_playback(self, checked: bool):
        if checked:
            self.play_btn.setText("Stop")
            self._start_playback()
        else:
            self.play_btn.setText("Play")
            self._stop_playback()

    def _start_playback(self):
        if AudioOutput is None:
            self.play_btn.blockSignals(True)
            self.play_btn.setChecked(False)
            self.play_btn.setText("Play")
            self.play_btn.blockSignals(False)
            QtWidgets.QMessageBox.warning(
                self,
                "Audio playback unavailable",
                "Install the audio-io extra to enable sound playback.",
            )
            return

        self._replace_audio_oscillator()
        self._audio_output = AudioOutput(
            sample_rate=AUDIO_SAMPLE_RATE,
            buffer_size=AUDIO_BUFFER_SIZE,
            audio_callback=self._audio_callback,
        )
        self._audio_output.set_master_volume(0.25)

        try:
            self._audio_output.start_playback()
        except Exception as exc:
            self.play_btn.blockSignals(True)
            self.play_btn.setChecked(False)
            self.play_btn.setText("Play")
            self.play_btn.blockSignals(False)
            self._audio_output = None
            self._audio_osc = None
            QtWidgets.QMessageBox.warning(
                self,
                "Audio playback unavailable",
                f"Could not start audio playback:\n{exc}",
            )

    def _stop_playback(self):
        if self._audio_output is not None:
            self._audio_output.cleanup(graceful=True)
            self._audio_output = None
        self._audio_osc = None

    def _audio_callback(self, frames: int) -> np.ndarray:
        if self._audio_osc is None:
            return np.zeros(frames, dtype=np.float32)
        return self._audio_osc.get_samples(frames, mode="vectorized")

    def closeEvent(self, event):
        self._stop_playback()
        super().closeEvent(event)

    def _update_x(self):
        pts = int(self.points_spin.value())
        sr = max(1, int(self.sr_spin.value()))
        self._x = np.linspace(0, pts / sr, pts, endpoint=False)
        amp = float(self.amp_spin.value() / 100)
        self.plot.setYRange(-max(1.0, amp) * 1.1, max(1.0, amp) * 1.1)

    def _get_samples_no_advance(self, pts: int):
        """Return a non-advancing snapshot of `pts` samples."""
        if self._osc is None:
            return np.zeros(int(pts), dtype=float)

        return self._osc.get_samples(int(pts))

    def _update_plot(self):
        if self._osc is None:
            return
        pts = int(self.points_spin.value())
        # Always use non-advancing snapshot
        y = self._get_samples_no_advance(pts)

        # # recompute x if points or sr changed
        if self._x is None or len(self._x) != len(y):
            self._update_x()

        self.curve.setData(self._x, y)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = WaveformViewer()
    win.resize(900, 500)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
