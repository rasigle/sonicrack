import json
import sys
import threading

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QDial,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QFileDialog,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from envelopes import generate_adsr_envelope


# --- Helper functions ----------------------------------------------------------


def generate_waveform(waveform_type, t, freq):
    if waveform_type == "Sine":
        return np.sin(2 * np.pi * freq * t)
    elif waveform_type == "Square":
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform_type == "Triangle":
        return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    elif waveform_type == "Sawtooth":
        return 2 * (t * freq - np.floor(t * freq + 0.5))
    return np.zeros_like(t)


# --- Knob widget ---------------------------------------------------------------


class Knob(QWidget):
    """Small rotary knob with a label and value display."""

    def __init__(self, name, minv, maxv, step, default, callback):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.dial = QDial()
        self.dial.setFixedSize(60, 60)
        self.dial.setMinimum(int(minv / step))
        self.dial.setMaximum(int(maxv / step))
        self.dial.setValue(int(default / step))
        self.dial.setNotchesVisible(True)
        self.step = step
        self.minv = minv
        self.maxv = maxv
        self.callback = callback

        self.label_name = QLabel(name)
        self.label_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_val = QLabel(f"{default:.2f}")
        self.label_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_val.setStyleSheet("color: #888; font-size: 10px;")

        layout.addWidget(self.label_name)
        layout.addWidget(self.dial)
        layout.addWidget(self.label_val)

        self.dial.valueChanged.connect(self._on_change)

    def _on_change(self, val):
        real_val = val * self.step
        real_val = min(max(real_val, self.minv), self.maxv)
        self.label_val.setText(f"{real_val:.2f}")
        self.callback(real_val)

    def value(self):
        return self.dial.value() * self.step

    def setValue(self, val):
        self.dial.setValue(int(val / self.step))
        self.label_val.setText(f"{val:.2f}")


# --- Main Synth Dialog ---------------------------------------------------------


class ADSRDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎛 ADSR Synthesizer – Compact Knob Edition")
        self.resize(1400, 850)

        self.fs = 44100
        self.duration = 1.5
        self.loop_enabled = False
        self.loop_thread = None
        self.stop_flag = threading.Event()

        main_layout = QVBoxLayout(self)

        # === PLOTS ===
        plot_layout = QHBoxLayout()
        self.time_fig = Figure()
        self.time_canvas = FigureCanvas(self.time_fig)
        self.time_ax = self.time_fig.add_subplot(111)
        plot_layout.addWidget(self.time_canvas, stretch=3)

        self.freq_fig = Figure()
        self.freq_canvas = FigureCanvas(self.freq_fig)
        self.freq_ax = self.freq_fig.add_subplot(111)
        plot_layout.addWidget(self.freq_canvas, stretch=3)
        main_layout.addLayout(plot_layout, stretch=5)

        # === CONTROL PANEL ===
        ctrl_box = QGroupBox("Synth Controls")
        ctrl_layout = QGridLayout(ctrl_box)
        ctrl_layout.setSpacing(10)

        # --- Waveform selection ---
        self.waveform_checks = []
        for i, w in enumerate(["Sine", "Square", "Triangle", "Sawtooth"]):
            cb = QCheckBox(w)
            cb.setChecked(w == "Sine")
            cb.stateChanged.connect(self.update_plot)
            self.waveform_checks.append(cb)
            ctrl_layout.addWidget(cb, 0, i)

        # --- ADSR knobs ---
        self.attack_knob = Knob("A", 0.0, 1.0, 0.01, 0.1, lambda _: self.update_plot())
        self.decay_knob = Knob("D", 0.0, 1.0, 0.01, 0.2, lambda _: self.update_plot())
        self.sustain_knob = Knob("S", 0.0, 1.0, 0.01, 0.7, lambda _: self.update_plot())
        self.release_knob = Knob("R", 0.0, 1.0, 0.01, 0.3, lambda _: self.update_plot())

        # --- Main knobs ---
        self.freq_knob = Knob("Freq", 20, 2000, 5, 440, lambda _: self.update_plot())
        self.amp_knob = Knob("Amp", 0.1, 2.0, 0.05, 1.0, lambda _: self.update_plot())

        # --- LFO knobs ---
        self.lfo_enable = QCheckBox("Enable LFO")
        self.lfo_enable.stateChanged.connect(self.update_plot)
        self.lfo_freq_knob = Knob(
            "LFO Freq", 0.1, 20, 0.1, 2, lambda _: self.update_plot()
        )
        self.lfo_depth_knob = Knob(
            "LFO Depth", 0.0, 1.0, 0.01, 0.3, lambda _: self.update_plot()
        )
        self.lfo_target_box = QComboBox()
        self.lfo_target_box.addItems(["Amplitude", "Frequency"])
        self.lfo_target_box.currentIndexChanged.connect(self.update_plot)

        # --- VCO knobs ---
        self.vco_enable = QCheckBox("Enable VCO")
        self.vco_enable.stateChanged.connect(self.update_plot)
        self.vco_freq_knob = Knob(
            "VCO Freq", 0.1, 1000, 1, 100, lambda _: self.update_plot()
        )
        self.vco_depth_knob = Knob(
            "VCO Depth", 0.0, 100.0, 1.0, 20, lambda _: self.update_plot()
        )
        self.vco_wave_box = QComboBox()
        self.vco_wave_box.addItems(["Sine", "Triangle", "Sawtooth", "Square"])
        self.vco_wave_box.currentIndexChanged.connect(self.update_plot)

        # --- Playback controls ---
        self.loop_toggle = QCheckBox("Loop Playback")
        self.loop_toggle.stateChanged.connect(self.toggle_loop)
        self.play_btn = QPushButton("▶ Play Once")
        self.stop_btn = QPushButton("⏹ Stop")
        self.save_btn = QPushButton("💾 Save Preset")
        self.load_btn = QPushButton("📂 Load Preset")
        self.close_btn = QPushButton("❌ Close")
        self.play_btn.clicked.connect(self.play_once)
        self.stop_btn.clicked.connect(self.stop_loop)
        self.save_btn.clicked.connect(self.save_preset)
        self.load_btn.clicked.connect(self.load_preset)
        self.close_btn.clicked.connect(self.close)

        # --- Layout organization ---
        knob_row1 = [
            self.attack_knob,
            self.decay_knob,
            self.sustain_knob,
            self.release_knob,
            self.freq_knob,
            self.amp_knob,
            self.lfo_freq_knob,
            self.lfo_depth_knob,
            self.vco_freq_knob,
            self.vco_depth_knob,
        ]
        for i, knob in enumerate(knob_row1):
            ctrl_layout.addWidget(knob, 1, i)

        # Row 2: toggles + combo boxes
        ctrl_layout.addWidget(self.lfo_enable, 2, 0)
        ctrl_layout.addWidget(QLabel("LFO Target:"), 2, 1)
        ctrl_layout.addWidget(self.lfo_target_box, 2, 2)
        ctrl_layout.addWidget(self.vco_enable, 2, 3)
        ctrl_layout.addWidget(QLabel("VCO Wave:"), 2, 4)
        ctrl_layout.addWidget(self.vco_wave_box, 2, 5)
        ctrl_layout.addWidget(self.loop_toggle, 2, 7)

        # Row 3: buttons
        ctrl_layout.addWidget(self.play_btn, 3, 0)
        ctrl_layout.addWidget(self.stop_btn, 3, 1)
        ctrl_layout.addWidget(self.save_btn, 3, 2)
        ctrl_layout.addWidget(self.load_btn, 3, 3)
        ctrl_layout.addWidget(self.close_btn, 3, 4)

        main_layout.addWidget(ctrl_box, stretch=0)
        self.update_plot()

    # --- Core synthesis ---
    def synthesize(self):
        t = np.linspace(0, self.duration, int(self.fs * self.duration))
        freq = self.freq_knob.value()
        amp = self.amp_knob.value()

        waves = [
            generate_waveform(cb.text(), t, freq)
            for cb in self.waveform_checks
            if cb.isChecked()
        ]
        base_wave = np.mean(waves, axis=0) if waves else np.zeros_like(t)

        # LFO
        amp_mod = np.ones_like(t)
        if self.lfo_enable.isChecked():
            lfo = np.sin(2 * np.pi * self.lfo_freq_knob.value() * t)
            if self.lfo_target_box.currentText() == "Amplitude":
                amp_mod = 1 + lfo * self.lfo_depth_knob.value()
            else:
                freq += lfo * freq * self.lfo_depth_knob.value()

        # VCO
        if self.vco_enable.isChecked():
            mod = generate_waveform(
                self.vco_wave_box.currentText(), t, self.vco_freq_knob.value()
            )
            vco = np.sin(2 * np.pi * (freq + mod * self.vco_depth_knob.value()) * t)
        else:
            vco = np.sin(2 * np.pi * freq * t)

        adsr = generate_adsr_envelope(
            t,
            self.attack_knob.value(),
            self.decay_knob.value(),
            self.sustain_knob.value(),
            self.release_knob.value(),
            self.duration,
        )
        signal = (base_wave + 0.5 * vco) * adsr * amp * amp_mod
        return signal, base_wave * amp, adsr * amp, t

    # --- Plot updates ---
    def update_plot(self):
        signal, base_wave, adsr_env, t = self.synthesize()

        # Time domain
        self.time_ax.clear()
        self.time_ax.plot(
            t, base_wave, color="grey", linestyle="--", alpha=0.5, label="Original Wave"
        )
        self.time_ax.plot(
            t, adsr_env, color="orange", linestyle=":", alpha=0.7, label="ADSR Envelope"
        )
        self.time_ax.plot(t, signal, color="tab:blue", label="Final Signal")
        self.time_ax.set_title("Time Domain (ADSR, LFO, VCO)")
        self.time_ax.set_xlabel("Time [s]")
        self.time_ax.set_ylabel("Amplitude")
        self.time_ax.legend()
        self.time_canvas.draw()

        # Frequency domain
        self.freq_ax.clear()
        freqs = np.fft.rfftfreq(len(signal), 1 / self.fs)
        self.freq_ax.plot(freqs, np.abs(np.fft.rfft(signal)), color="tab:blue")
        self.freq_ax.set_xlim(0, 5000)
        self.freq_ax.set_title("Frequency Domain")
        self.freq_ax.set_xlabel("Frequency [Hz]")
        self.freq_ax.set_ylabel("Magnitude")
        self.freq_canvas.draw()

    # --- Playback ---
    def play_once(self):
        signal, *_ = self.synthesize()
        sd.stop()
        signal /= np.max(np.abs(signal) + 1e-9)
        sd.play(signal, self.fs)

    def toggle_loop(self):
        self.loop_enabled = self.loop_toggle.isChecked()
        if self.loop_enabled:
            self.start_loop()
        else:
            self.stop_loop()

    def start_loop(self):
        if self.loop_thread and self.loop_thread.is_alive():
            return
        self.stop_flag.clear()
        self.loop_thread = threading.Thread(target=self._loop_playback, daemon=True)
        self.loop_thread.start()

    def _loop_playback(self):
        while not self.stop_flag.is_set():
            signal, *_ = self.synthesize()
            signal /= np.max(np.abs(signal) + 1e-9)
            sd.play(signal, self.fs, blocking=True)

    def stop_loop(self):
        self.stop_flag.set()
        sd.stop()

    # --- Preset handling ---
    def save_preset(self):
        data = {
            "attack": self.attack_knob.value(),
            "decay": self.decay_knob.value(),
            "sustain": self.sustain_knob.value(),
            "release": self.release_knob.value(),
            "freq": self.freq_knob.value(),
            "amp": self.amp_knob.value(),
            "lfo_enabled": self.lfo_enable.isChecked(),
            "lfo_freq": self.lfo_freq_knob.value(),
            "lfo_depth": self.lfo_depth_knob.value(),
            "lfo_target": self.lfo_target_box.currentText(),
            "vco_enabled": self.vco_enable.isChecked(),
            "vco_freq": self.vco_freq_knob.value(),
            "vco_depth": self.vco_depth_knob.value(),
            "vco_wave": self.vco_wave_box.currentText(),
            "waveforms": [cb.text() for cb in self.waveform_checks if cb.isChecked()],
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", "", "ADSR Preset (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", "", "ADSR Preset (*.json)"
        )
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        for cb in self.waveform_checks:
            cb.setChecked(cb.text() in data.get("waveforms", []))
        self.attack_knob.setValue(data.get("attack", 0.1))
        self.decay_knob.setValue(data.get("decay", 0.2))
        self.sustain_knob.setValue(data.get("sustain", 0.7))
        self.release_knob.setValue(data.get("release", 0.3))
        self.freq_knob.setValue(data.get("freq", 440))
        self.amp_knob.setValue(data.get("amp", 1.0))
        self.lfo_enable.setChecked(data.get("lfo_enabled", False))
        self.lfo_freq_knob.setValue(data.get("lfo_freq", 2))
        self.lfo_depth_knob.setValue(data.get("lfo_depth", 0.3))
        self.lfo_target_box.setCurrentText(data.get("lfo_target", "Amplitude"))
        self.vco_enable.setChecked(data.get("vco_enabled", False))
        self.vco_freq_knob.setValue(data.get("vco_freq", 100))
        self.vco_depth_knob.setValue(data.get("vco_depth", 20))
        self.vco_wave_box.setCurrentText(data.get("vco_wave", "Sine"))
        self.update_plot()


# --- Run App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = ADSRDialog()
    dlg.show()
    sys.exit(app.exec())
