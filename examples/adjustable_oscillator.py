import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
from engine import (
    SineOscillator,
    SawtoothOscillator,
    SquareOscillator,
    TriangleOscillator,
    Oscillator,
)

CLASS_MAP = {
    "Sine": SineOscillator,
    "Saw": SawtoothOscillator,
    "Square": SquareOscillator,
    "Triangle": TriangleOscillator,
}


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

        self.freq_spin = QtWidgets.QDial()
        self.freq_spin.setRange(10, 2000)
        self.freq_spin.setValue(10)
        self.freq_spin.setSingleStep(1)
        ctrl_l.addWidget(QtWidgets.QLabel("Freq (Hz):"))
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

        # Snapshot-only button (disabled, informational)
        self.toggle_btn = QtWidgets.QPushButton("Snapshot (no advance)")
        self.toggle_btn.setEnabled(False)
        ctrl_l.addWidget(self.toggle_btn)

        layout.addWidget(ctrl)

        # Internal state
        self._osc: Oscillator | None = None
        self._x = None

        # Connections
        self.wave_combo.currentTextChanged.connect(self._change_waveform)
        # Update oscillator properties when widgets change
        for widget in (
            self.freq_spin,
            self.amp_spin,
            self.phase_spin,
            self.sr_spin,
            self.points_spin,
        ):
            widget.valueChanged.connect(self._update_osc_params)

        # Timer for continuous redraw (non-advancing snapshot mode)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(30)  # ~33 FPS

        # Create oscillator once
        self._create_initial_osc()
        self._update_x()  # initialize x axis and y-range

    def _create_initial_osc(self):
        cls = CLASS_MAP[self.wave_combo.currentText()]
        freq = float(self.freq_spin.value())
        amp = float(self.amp_spin.value() / 100)
        phase = float(self.phase_spin.value())
        sr = int(self.sr_spin.value())

        try:
            osc = cls(freq=freq, amplitude=amp, phase=phase, sample_rate=sr)
        except TypeError:
            osc = cls(freq, amp, phase, sr)
        try:
            iter(osc)
        except TypeError:
            pass
        self._osc = osc

    def _change_waveform(self, _):
        cls = CLASS_MAP[self.wave_combo.currentText()]
        freq = float(self.freq_spin.value())
        amp = float(self.amp_spin.value() / 100)
        phase = float(self.phase_spin.value())
        sr = int(self.sr_spin.value())
        try:
            osc = cls(freq=freq, amplitude=amp, phase=phase, sample_rate=sr)
        except TypeError:
            osc = cls(freq, amp, phase, sr)
        try:
            iter(osc)
        except TypeError:
            pass
        self._osc = osc

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

        # Update plot range and x axis
        self._update_x()

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
