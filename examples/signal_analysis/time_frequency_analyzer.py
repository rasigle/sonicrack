import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy import signal


# --- Matplotlib Canvas ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


# --- Waveform Widget (inside a group) ---
class WaveformWidget(QFrame):
    def __init__(self, remove_callback, update_callback, parent=None):
        super().__init__(parent)
        self.remove_callback = remove_callback
        self.update_callback = update_callback
        self.setFrameShape(QFrame.Shape.Box)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.waveform = QComboBox()
        self.waveform.addItems(["sine", "square", "triangle", "saw"])

        self.freq = QDoubleSpinBox()
        self.freq.setRange(0.1, 1000)
        self.freq.setValue(5)
        self.freq.setSuffix(" Hz")

        self.amp = QDoubleSpinBox()
        self.amp.setRange(0.1, 10.0)
        self.amp.setValue(1.0)
        self.amp.setPrefix("A=")

        self.phase = QDoubleSpinBox()
        self.phase.setRange(0.0, 2 * np.pi)
        self.phase.setValue(0.0)
        self.phase.setPrefix("ϕ=")

        remove_btn = QPushButton("❌ Remove")
        remove_btn.clicked.connect(lambda: self.remove_callback(self))

        # Connect all value changes to parent update
        for widget in [self.waveform, self.freq, self.amp, self.phase]:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.update_callback)
            else:
                widget.valueChanged.connect(self.update_callback)

        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.waveform)
        layout.addWidget(QLabel("Freq:"))
        layout.addWidget(self.freq)
        layout.addWidget(QLabel("Amp:"))
        layout.addWidget(self.amp)
        layout.addWidget(QLabel("Phase:"))
        layout.addWidget(self.phase)
        layout.addWidget(remove_btn)

    def get_params(self):
        return {
            "waveform": self.waveform.currentText(),
            "freq": self.freq.value(),
            "amp": self.amp.value(),
            "phase": self.phase.value(),
        }


# --- Signal Group (independent signal made of multiple waveforms) ---
class SignalGroup(QGroupBox):
    def __init__(self, remove_callback, update_callback, parent=None):
        super().__init__("Signal Group")
        self.remove_callback = remove_callback
        self.update_callback = update_callback
        self.waveforms = []

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        add_waveform_btn = QPushButton("➕ Add Waveform")
        remove_group_btn = QPushButton("🗑 Remove Group")

        # NEW: toggles
        self.show_time = QCheckBox("Show in Time")
        self.show_fft = QCheckBox("Show in FFT")
        self.show_time.setChecked(True)
        self.show_fft.setChecked(True)

        self.show_time.stateChanged.connect(self.update_callback)
        self.show_fft.stateChanged.connect(self.update_callback)

        add_waveform_btn.clicked.connect(self.add_waveform)
        remove_group_btn.clicked.connect(lambda: self.remove_callback(self))

        header.addWidget(add_waveform_btn)
        header.addWidget(remove_group_btn)
        header.addStretch()
        header.addWidget(self.show_time)
        header.addWidget(self.show_fft)

        self.waveform_layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addLayout(self.waveform_layout)

        self.add_waveform()

    def get_visibility(self):
        """Return (show_time, show_fft)"""
        return self.show_time.isChecked(), self.show_fft.isChecked()

    def add_waveform(self):
        widget = WaveformWidget(
            remove_callback=self.remove_waveform, update_callback=self.update_callback
        )
        self.waveforms.append(widget)
        self.waveform_layout.addWidget(widget)
        self.update_callback()

    def remove_waveform(self, widget):
        self.waveforms.remove(widget)
        widget.setParent(None)
        self.update_callback()

    def get_combined_signal(self, t):
        """Sum all waveforms to create one signal"""
        y = np.zeros_like(t)
        for wf in self.waveforms:
            p = wf.get_params()
            if p["waveform"] == "sine":
                wave = np.sin(2 * np.pi * p["freq"] * t + p["phase"])
            elif p["waveform"] == "square":
                wave = signal.square(2 * np.pi * p["freq"] * t + p["phase"])
            elif p["waveform"] == "triangle":
                wave = signal.sawtooth(
                    2 * np.pi * p["freq"] * t + p["phase"], width=0.5
                )
            elif p["waveform"] == "saw":
                wave = signal.sawtooth(2 * np.pi * p["freq"] * t + p["phase"])
            y += p["amp"] * wave
        return y


class CompositeSignalWidget(QFrame):
    def __init__(self, get_group_names, remove_callback, update_callback, parent=None):
        super().__init__(parent)
        self.get_group_names = get_group_names
        self.remove_callback = remove_callback
        self.update_callback = update_callback

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.operation = QComboBox()
        self.operation.addItems(["Add", "Subtract", "Multiply", "Average"])

        self.group_checks = []
        self.group_layout = QHBoxLayout()

        self.refresh_group_checks()

        # NEW: visibility checkboxes
        self.show_time = QCheckBox("Show in Time")
        self.show_fft = QCheckBox("Show in FFT")
        self.show_time.setChecked(True)
        self.show_fft.setChecked(True)

        for cb in [self.show_time, self.show_fft]:
            cb.stateChanged.connect(self.update_callback)

        self.remove_btn = QPushButton("❌ Remove")
        self.remove_btn.clicked.connect(lambda: self.remove_callback(self))

        layout.addWidget(QLabel("Operation:"))
        layout.addWidget(self.operation)
        layout.addLayout(self.group_layout)
        layout.addWidget(self.show_time)
        layout.addWidget(self.show_fft)
        layout.addWidget(self.remove_btn)

        self.operation.currentIndexChanged.connect(self.update_callback)

    def get_visibility(self):
        """Return (show_time, show_fft)"""
        return self.show_time.isChecked(), self.show_fft.isChecked()

    def refresh_group_checks(self):
        """Refresh available groups (in case groups were added/removed)"""
        # clear old checkboxes
        for cb in self.group_checks:
            cb.setParent(None)
        self.group_checks.clear()

        for name in self.get_group_names():
            cb = QCheckBox(name)
            cb.stateChanged.connect(self.update_callback)
            self.group_checks.append(cb)
            self.group_layout.addWidget(cb)

    def get_selected_groups(self):
        """Return indices of selected groups"""
        return [i for i, cb in enumerate(self.group_checks) if cb.isChecked()]

    def get_operation(self):
        return self.operation.currentText()


# --- Main Window ---
class TimeFreqDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hierarchical Time–Frequency Signal Analyzer")
        self.resize(1300, 700)

        main_layout = QVBoxLayout(self)
        plot_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        # Plot canvases
        self.time_canvas = MplCanvas(self)
        self.freq_canvas = MplCanvas(self)
        plot_layout.addWidget(self.time_canvas)
        plot_layout.addWidget(self.freq_canvas)

        # Groups container
        self.groups = []
        self.group_container = QVBoxLayout()

        # Scrollable area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.group_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        # Add Group button
        add_group_btn = QPushButton("➕ Add Signal Group")
        add_group_btn.clicked.connect(self.add_group)

        control_layout.addWidget(add_group_btn)
        control_layout.addWidget(scroll_area)

        main_layout.addLayout(plot_layout)
        main_layout.addLayout(control_layout)

        # --- Composite signals section ---
        self.composites = []
        self.composite_container = QVBoxLayout()

        add_comp_btn = QPushButton("➕ Add Composite Signal")
        add_comp_btn.clicked.connect(self.add_composite_signal)

        control_layout.addWidget(QLabel("Composite Signals:"))
        control_layout.addWidget(add_comp_btn)
        control_layout.addLayout(self.composite_container)

        # Start with one group
        self.add_group()

        # Real-time update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(150)  # refresh every 150 ms

    def get_group_names(self):
        return [f"Signal {i+1}" for i in range(len(self.groups))]

    def add_composite_signal(self):
        widget = CompositeSignalWidget(
            get_group_names=self.get_group_names,
            remove_callback=self.remove_composite_signal,
            update_callback=self.update_plots,
        )
        self.composites.append(widget)
        self.composite_container.addWidget(widget)
        self.update_plots()

    def remove_composite_signal(self, widget):
        self.composites.remove(widget)
        widget.setParent(None)
        self.update_plots()

    def add_group(self):
        group = SignalGroup(
            remove_callback=self.remove_group, update_callback=self.update_plots
        )
        self.groups.append(group)
        self.group_container.addWidget(group)
        self.update_plots()

    def remove_group(self, group):
        self.groups.remove(group)
        group.setParent(None)
        self.update_plots()

    def update_plots(self):
        t = np.linspace(0, 1, 2000)
        self.time_canvas.axes.clear()
        self.freq_canvas.axes.clear()

        if not self.groups:
            self.time_canvas.draw()
            self.freq_canvas.draw()
            return

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        signals = []

        # --- Plot individual signal groups ---
        for i, group in enumerate(self.groups):
            y = group.get_combined_signal(t)
            signals.append(y)
            color = colors[i % len(colors)]
            show_time, show_fft = group.get_visibility()

            # Time-domain
            if show_time:
                style = "-" if show_fft else ":"  # dashed if hidden in FFT
                self.time_canvas.axes.plot(
                    t, y, label=f"Signal {i+1}", color=color, linestyle=style
                )
            else:
                # Dimmed version if hidden in time
                self.time_canvas.axes.plot(t, y, color="grey", linestyle=":", alpha=0.3)

            # Frequency-domain
            if show_fft:
                freqs = np.fft.rfftfreq(len(t), d=t[1] - t[0])
                Y = np.abs(np.fft.rfft(y))
                self.freq_canvas.axes.plot(freqs, Y, label=f"Signal {i+1}", color=color)

        # --- Plot composite signals ---
        for c_idx, comp in enumerate(self.composites):
            selected = comp.get_selected_groups()
            if len(selected) < 2:
                continue

            op = comp.get_operation()
            combo = signals[selected[0]].copy()
            for idx in selected[1:]:
                if op == "Add":
                    combo += signals[idx]
                elif op == "Subtract":
                    combo -= signals[idx]
                elif op == "Multiply":
                    combo *= signals[idx]
                elif op == "Average":
                    combo = (combo + signals[idx]) / 2.0

            color = "black"
            style = "--"
            label = f"Composite {c_idx+1} ({op})"

            show_time, show_fft = comp.get_visibility()

            if show_time:
                style = "--" if show_fft else ":"
                self.time_canvas.axes.plot(
                    t, combo, color=color, linestyle=style, label=label
                )
            else:
                self.time_canvas.axes.plot(
                    t, combo, color="grey", linestyle=":", alpha=0.3
                )

            if show_fft:
                freqs = np.fft.rfftfreq(len(t), d=t[1] - t[0])
                Y = np.abs(np.fft.rfft(combo))
                self.freq_canvas.axes.plot(
                    freqs, Y, color=color, linestyle=style, label=label
                )

        # Axes titles, legends, and redraw
        self.time_canvas.axes.set_title("Time Domain Signals")
        self.time_canvas.axes.set_xlabel("Time [s]")
        self.time_canvas.axes.set_ylabel("Amplitude")
        self.time_canvas.axes.legend()

        self.freq_canvas.axes.set_title("Frequency Domain (FFT)")
        self.freq_canvas.axes.set_xlabel("Frequency [Hz]")
        self.freq_canvas.axes.set_ylabel("Magnitude")
        self.freq_canvas.axes.set_xlim(0, 100)
        self.freq_canvas.axes.legend()

        self.time_canvas.draw()
        self.freq_canvas.draw()


# --- Run App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = TimeFreqDialog()
    dlg.show()
    sys.exit(app.exec())
