"""PyQt6 Audio FFT Analyser

This is a version 2 of an audio spectrogram analyser application that uses
Matplotlib for spectrogram rendering instead of pyqtgraph's ImageItem.
For the spectragram, it uses librosa's specshow for better visualization.

However, performance is war lower than the pyqtgraph version, especially for large
audio files, due to the overhead of Matplotlib rendering.
"""

import contextlib
import sys
import threading
from pathlib import Path

import librosa
import numpy as np
import pyaudio
import pyqtgraph as pg
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets

# Ensure OpenGL if available for performance
pg.setConfigOptions(useOpenGL=True)


def read_audio_file(path: str):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    max_abs = np.max(np.abs(data)) if data.size else 1.0
    if max_abs > 0:
        data = data / max_abs
    return sr, data


def compute_spectrogram(data, sr, nfft=2048, hop=None, scale="dB"):
    """Compute a spectrogram of audio data.

    Args:
        data (np.ndarray): 1D array of audio samples.
        sr (int): Sample rate of the audio data.
        nfft (int): Number of FFT points.
        hop (int): Hop size between frames. If None, defaults to nfft // 4.
        scale (str): "dB" for decibel scale, "linear" for linear magnitude.

    Returns:
        S_out (np.ndarray): Spectrogram array (freq_bins x time_frames).
        freqs (np.ndarray): Frequency bins in Hz.
        times (np.ndarray): Time frames in seconds.
    """
    if hop is None:
        hop = nfft // 4

    # pad data to at least nfft
    if len(data) < nfft:
        data = np.pad(data, (0, nfft - len(data)))

    # number of frames
    n_frames = 1 + (len(data) - nfft) // hop
    window = np.hanning(nfft)
    # build frames
    frames = np.lib.stride_tricks.sliding_window_view(data, nfft)[::hop][:n_frames]

    # apply window and fft
    S = np.fft.rfft(frames * window[None, :], axis=1)
    S_mag = np.abs(S).T  # shape (freq_bins, time_frames)
    S_out = 20.0 * np.log10(S_mag + 1e-12) if scale == "dB" else S_mag
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    times = np.arange(S_out.shape[1]) * hop / sr
    return S_out, freqs, times


class SpectrogramWidget(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # Create figure and axes for embedding
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # Store reference to colorbar for updates
        self.colorbar = None

        # Playback position line
        self.play_line = None

        # Set tight layout for better appearance
        self.fig.tight_layout()

    def set_spectrogram(self, y, nfft, sr, cmap="inferno"):
        """Plot spectrogram using librosa inside the widget."""
        # Clear previous plot
        self.ax.clear()
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        # Compute STFT and convert to dB
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=nfft)), ref=np.max)

        # Display spectrogram using librosa
        img = librosa.display.specshow(
            D, y_axis="log", x_axis="time", sr=sr, ax=self.ax, cmap=cmap
        )

        # Add colorbar
        self.colorbar = self.fig.colorbar(img, ax=self.ax, format="%+2.f dB")

        # Set labels
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_title("Spectrogram")

        # Re-add playback line if it exists
        if self.play_line is not None:
            x_pos = self.play_line.get_xdata()[0]
            self.play_line = self.ax.axvline(
                x=x_pos, color="red", linewidth=2, alpha=0.7
            )

        # Adjust layout and redraw
        self.fig.tight_layout()
        self.draw()

    def update_play_line(self, time_sec):
        """Update the playback position line.

        Args:
            time_sec: Current playback time in seconds
        """
        if self.play_line is None:
            # Create the line
            self.play_line = self.ax.axvline(
                x=time_sec, color="red", linewidth=2, alpha=0.7
            )
        else:
            # Update existing line position
            self.play_line.set_xdata([time_sec, time_sec])

        self.draw_idle()  # Use draw_idle for better performance during updates

    def reset_view(self):
        """Reset the view to auto-range."""
        self.ax.autoscale()
        self.draw()

    def set_colormap(self, cmap_name):
        """Update the colormap of the current spectrogram."""
        # This would require re-plotting with the new colormap
        # For now, just store it for next set_spectrogram call
        pass


class FFTAnalyserWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio FFT Analyser")
        self.resize(1400, 940)
        self.setAcceptDrops(True)

        std_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
        )
        self.setWindowIcon(std_icon)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)

        # Top controls row
        top_row = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_row)

        self.file_label = QtWidgets.QLabel("Drop an audio file (.wav/.flac/.mp3)")
        top_row.addWidget(self.file_label)

        top_row.addStretch()

        self.device_combo = QtWidgets.QComboBox()
        top_row.addWidget(QtWidgets.QLabel("Output device:"))
        top_row.addWidget(self.device_combo)

        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_playback)
        top_row.addWidget(self.play_button)

        self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(180)
        top_row.addWidget(QtWidgets.QLabel("Volume:"))
        top_row.addWidget(self.volume_slider)

        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        top_row.addWidget(self.time_label)

        # Second row controls
        row2 = QtWidgets.QHBoxLayout()
        main_layout.addLayout(row2)

        self.fft_size_spin = QtWidgets.QComboBox()
        sizes = (512, 1024, 2048, 4096, 8192, 16384)
        for v in sizes:
            self.fft_size_spin.addItem(str(v), userData=v)
        self.fft_size_spin.setCurrentIndex(sizes.index(2048))
        self.fft_size_spin.valueChanged = self.fft_size_spin.currentIndexChanged
        row2.addWidget(QtWidgets.QLabel("FFT size (points:"))
        row2.addWidget(self.fft_size_spin)

        self.freq_scale_combo = QtWidgets.QComboBox()
        self.freq_scale_combo.addItems(["Linear freq", "Log freq"])
        row2.addWidget(QtWidgets.QLabel("FFT scale:"))
        row2.addWidget(self.freq_scale_combo)

        self.spec_colormap_combo = QtWidgets.QComboBox()
        self.spec_colormap_combo.addItems(["plasma", "inferno", "viridis", "turbo"])
        row2.addWidget(QtWidgets.QLabel("Colormap:"))
        row2.addWidget(self.spec_colormap_combo)

        self.spec_scale_combo = QtWidgets.QComboBox()
        self.spec_scale_combo.addItems(["dB", "linear"])
        row2.addWidget(QtWidgets.QLabel("Spectrogram scale:"))
        row2.addWidget(self.spec_scale_combo)

        export_spec_btn = QtWidgets.QPushButton("Export Spectrogram PNG")
        export_spec_btn.clicked.connect(self.export_spectrogram_png)
        row2.addWidget(export_spec_btn)

        export_fft_btn = QtWidgets.QPushButton("Export FFT CSV")
        export_fft_btn.clicked.connect(self.export_fft_csv)
        row2.addWidget(export_fft_btn)

        row2.addStretch()

        # Position slider with seeking
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderPressed.connect(self._position_pressed)
        self.position_slider.sliderReleased.connect(self._position_released)
        self.position_slider.valueChanged.connect(self._position_changed_by_user)
        main_layout.addWidget(self.position_slider)

        # Splitter: left plots, right spectrogram
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left_frame = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_frame)

        # waveform and FFT plots
        self.plot_wf = pg.PlotWidget(title="Waveform")
        self.plot_wf.showGrid(x=True, y=True)
        self.plot_wf.setLabel(axis="left", text="Amplitude")
        self.plot_wf.setLabel(axis="bottom", text="Time [s]")

        self.plot_fft = pg.PlotWidget(title="FFT (dB)")
        self.plot_fft.setLabel(axis="left", text="Amplitude [dB]")
        self.plot_fft.setLabel(axis="bottom", text="Frequency [Hz]")
        self.plot_fft.showGrid(x=True, y=True)
        left_layout.addWidget(self.plot_wf, stretch=1)
        left_layout.addWidget(self.plot_fft, stretch=1)

        self.wf_curve = self.plot_wf.plot(
            pen=pg.mkPen(color=(60, 120, 200), width=1)
        )  # soft blue
        self.fft_curve = self.plot_fft.plot(
            pen=pg.mkPen(color=(255, 140, 0), width=2)
        )  # orange

        splitter.addWidget(left_frame)

        # right: pyqtgraph spectrogram
        right_frame = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_frame)

        self.spec_widget = SpectrogramWidget()
        right_layout.addWidget(self.spec_widget)
        splitter.addWidget(right_frame)

        # Adjust splitter stretch: spectrogram bigger
        splitter.setStretchFactor(0, 1)  # left panel (waveform+FFT)
        splitter.setStretchFactor(1, 3)  # right panel (spectrogram)

        # status
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Audio / playback
        self.sr = None
        self.data = None
        self.pyaudio_inst = pyaudio.PyAudio()
        self.stream = None
        self.stream_lock = threading.Lock()
        self.play_pos = 0  # sample index
        self.chunk = 1024
        self.is_playing = False
        self.user_is_seeking = False
        self.volume = 1.0
        self.selected_device_index = None

        # UI timer
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.setInterval(100)
        self.ui_timer.timeout.connect(self._ui_timer_tick)

        # Spectrogram playback line update timer (less frequent)
        self.spec_line_timer = QtCore.QTimer()
        self.spec_line_timer.setInterval(
            200
        )  # Update every 200ms instead of every frame
        self.spec_line_timer.timeout.connect(self._update_spec_playback_line)

        # connect controls
        self.fft_size_spin.valueChanged.connect(self._spec_nfft_changed)
        self.freq_scale_combo.currentIndexChanged.connect(self.update_display)
        self.spec_colormap_combo.currentIndexChanged.connect(
            self._spec_colormap_changed
        )
        self.spec_scale_combo.currentIndexChanged.connect(self._spec_scale_changed)
        self.volume_slider.valueChanged.connect(self._volume_changed)
        self.device_combo.currentIndexChanged.connect(self._device_changed)

        # populate device list and default
        self._populate_output_devices()

    # Device enumeration
    def _populate_output_devices(self):
        self.device_combo.clear()
        try:
            default_info = self.pyaudio_inst.get_default_output_device_info()
            default_index = default_info.get("index", None)
        except OSError:
            default_index = None

        for i in range(self.pyaudio_inst.get_device_count()):
            info = self.pyaudio_inst.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) > 0:
                label = f"{info.get('name')} (id={i})"
                self.device_combo.addItem(label, userData=i)
                if default_index is not None and info.get("index") == default_index:
                    # later we'll set current index
                    pass

        # try to set to system default if possible
        if default_index is not None:
            # find item with that index
            for j in range(self.device_combo.count()):
                if self.device_combo.itemData(j) == default_index:
                    self.device_combo.setCurrentIndex(j)
                    self.selected_device_index = default_index
                    break
        else:
            if self.device_combo.count() > 0:
                self.device_combo.setCurrentIndex(0)
                self.selected_device_index = self.device_combo.itemData(0)

    # Drag & drop
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if Path(path).suffix.lower() in [".wav", ".flac", ".mp3"]:
                event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        path = event.mimeData().urls()[0].toLocalFile()
        self.load_file(path)

    # Load audio and compute spectrogram
    def load_file(self, path: str):
        try:
            self.sr, self.data = read_audio_file(path)
            self.file_label.setText(
                f"Loaded: {Path(path).name} — {self.sr} Hz — {len(self.data)} samples"
            )
            self.status.showMessage("Computing spectrogram...", 2000)
            QtWidgets.QApplication.processEvents()
            self._compute_and_set_spectrogram()
            self.position_slider.setValue(0)
            self._update_time_label_from_slider()
            self.play_pos = 0
            self.update_display()
            self.status.showMessage("Loaded and spectrogram ready", 3000)
        except Exception as e:
            self.status.showMessage(f"Error loading file: {e}", 8000)

    def _compute_and_set_spectrogram(self):
        if self.data is None:
            return

        nfft = int(self.fft_size_spin.currentData())
        cmap = self.spec_colormap_combo.currentText()
        self.spec_widget.set_spectrogram(self.data, nfft=nfft, sr=self.sr, cmap=cmap)

    # PyAudio callback
    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        with self.stream_lock:
            if self.data is None:
                out = np.zeros(frame_count, dtype=np.float32)
                flag = pyaudio.paContinue
            else:
                end = self.play_pos + frame_count
                if end >= len(self.data):
                    chunk = self.data[self.play_pos : len(self.data)]
                    out = np.zeros(frame_count, dtype=np.float32)
                    out[: len(chunk)] = chunk
                    flag = pyaudio.paComplete
                else:
                    out = self.data[self.play_pos : end]
                    flag = pyaudio.paContinue
                self.play_pos = min(len(self.data), end)

        # apply volume (immediate)
        vol = getattr(self, "volume", 1.0)
        if vol != 1.0:
            out = out * vol

        # prevent clipping: if any sample >1.0, scale
        max_abs = np.max(np.abs(out)) if out.size else 0.0
        if max_abs > 1.0:
            out = out / max_abs
        return out.astype(np.float32).tobytes(), flag

    # Playback controls
    def start_playback(self):
        if self.data is None:
            return

        # if at end and clicked play, restart from beginning
        if self.play_pos == len(self.data):
            self.play_pos = 0

        # prepare device
        device_index = self.selected_device_index
        try:
            if self.stream is not None:
                self.stop_playback()
            # start stream
            self.stream = self.pyaudio_inst.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=int(self.sr),
                output=True,
                frames_per_buffer=self.chunk,
                output_device_index=device_index,
                stream_callback=self._pyaudio_callback,
            )
            self.is_playing = True
            self.play_button.setText("⏸ Pause")
            self.stream.start_stream()
            self.ui_timer.start()
            self.spec_line_timer.start()
        except Exception as e:
            self.status.showMessage(f"Failed to start playback: {e}", 6000)

    def stop_playback(self):
        self.is_playing = False
        self.play_button.setText("▶ Play")
        self.ui_timer.stop()
        self.spec_line_timer.stop()
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            # ensure selected device is set
            idx = self.device_combo.currentIndex()
            if idx >= 0:
                self.selected_device_index = self.device_combo.itemData(idx)
            self.start_playback()

    # Seeking behavior: immediate even while playing
    def _position_pressed(self):
        self.user_is_seeking = True

    def _position_released(self):
        self.user_is_seeking = False
        self._apply_slider_position()

    def _position_changed_by_user(self, value):
        if self.user_is_seeking:
            # show time update while dragging
            self._update_time_label_from_slider(value)
        else:
            # immediate seek while playing or not
            self._apply_slider_position()

    def _apply_slider_position(self):
        if self.data is None:
            return

        pct = self.position_slider.value() / self.position_slider.maximum()
        new_pos = int(pct * len(self.data))
        with self.stream_lock:
            self.play_pos = max(0, min(len(self.data), new_pos))

        # self.spec_widget.update_play_line(self.play_pos / self.sr)
        self._update_time_label_from_slider()

        # if playing, the callback will pick up the new play_pos immediately
        self.update_display()

    def _update_time_label_from_slider(self, slider_value=None):
        if self.data is None:
            self.time_label.setText("00:00 / 00:00")
            return

        if slider_value is None:
            slider_value = self.position_slider.value()
        pct = slider_value / self.position_slider.maximum()
        cur = int(pct * len(self.data))
        self.time_label.setText(
            f"{self._format_time(cur / self.sr)} / "
            f"{self._format_time(len(self.data) / self.sr)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    # UI timer tick
    def _ui_timer_tick(self):
        # update position slider from play_pos
        if self.data is None:
            return

        pct = int((self.play_pos / len(self.data)) * self.position_slider.maximum())
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(pct)
        self.position_slider.blockSignals(False)

        # update time label
        self.time_label.setText(
            f"{self._format_time(self.play_pos / self.sr)} / "
            f"{self._format_time(len(self.data) / self.sr)}"
        )

        self.update_display()

        # update spectrogram play line (time in seconds)
        # self.spec_widget.update_play_line(self.play_pos / self.sr)

        # stop if finished
        if self.stream is None:
            return
        try:
            if not self.stream.is_active() and self.play_pos >= len(self.data):
                self.stop_playback()
        except OSError as e:
            self.status.showMessage(f"Audio stream error: {e}", 6000)
            self.stop_playback()

    def _update_spec_playback_line(self):
        """Update spectrogram playback line position (called less frequently than
        UI timer)."""
        if self.data is None or self.sr is None:
            return
        # Update spectrogram play line (time in seconds)
        self.spec_widget.update_play_line(self.play_pos / self.sr)

    # Controls handlers
    def _volume_changed(self, v):
        # v is 0..200, map to 0.0..2.0
        self.volume = float(v) / 100.0

    def _device_changed(self, idx):
        # set selected device index; if playing, restart playback on new device
        if idx < 0:
            return
        self.selected_device_index = self.device_combo.itemData(idx)
        if self.is_playing:
            # restart stream to use new device
            cur_pos = None
            with self.stream_lock:
                cur_pos = self.play_pos
            self.stop_playback()
            with self.stream_lock:
                self.play_pos = cur_pos
            self.start_playback()

    def _spec_nfft_changed(self, _):
        # recompute spectrogram display with chosen nfft
        self._compute_and_set_spectrogram()

    def _spec_colormap_changed(self, _):
        # recompute spectrogram display with chosen colormap
        self._compute_and_set_spectrogram()

    def _spec_scale_changed(self, _):
        # recompute spectrogram display with chosen scale
        self._compute_and_set_spectrogram()

    # Display update (waveform + FFT)
    def update_display(self):
        if self.data is None:
            return

        self._update_fft()

    def _update_fft(self):
        n = len(self.data)
        t = np.arange(n) / float(self.sr)
        max_points = 20000
        if n > max_points:
            step = int(np.ceil(n / max_points))
            self.wf_curve.setData(t[::step], self.data[::step])
        else:
            self.wf_curve.setData(t, self.data)

        fft_size = int(self.fft_size_spin.currentData())
        start_pct = self.position_slider.value() / self.position_slider.maximum()
        start_idx = int(start_pct * max(0, n - fft_size))
        start_idx = max(0, min(start_idx, n - fft_size))
        segment = self.data[start_idx : start_idx + fft_size]
        if len(segment) < 2:
            return

        window = np.hanning(len(segment))
        fft_vals = np.fft.rfft(segment * window)
        freqs = np.fft.rfftfreq(len(segment), 1.0 / self.sr)
        mag_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)

        if self.freq_scale_combo.currentIndex() == 1:
            mask = freqs > 0
            self.plot_fft.setLogMode(x=True, y=False)
            self.fft_curve.setData(freqs[mask], mag_db[mask])
        else:
            self.plot_fft.setLogMode(x=False, y=False)
            self.fft_curve.setData(freqs, mag_db)

    # Export functions
    def export_fft_csv(self):
        if self.data is None:
            return
        fft_size = int(self.fft_size_spin.currentData())
        segment = self.data[:fft_size]
        fft_vals = np.fft.rfft(segment * np.hanning(len(segment)))
        freqs = np.fft.rfftfreq(len(segment), 1.0 / self.sr)
        mag_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)
        fname = "fft_spectrum.csv"
        np.savetxt(
            fname,
            np.column_stack((freqs, mag_db)),
            delimiter=",",
            header="freq(Hz),magnitude(dB)",
        )
        self.status.showMessage(f"Exported FFT CSV → {fname}", 4000)

    def export_spectrogram_png(self):
        if self.data is None:
            return
        fname = "spectrogram.png"
        self.status.showMessage(f"Exported spectrogram → {fname}", 4000)

    def closeEvent(self, event):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
        except (OSError, RuntimeError):
            pass
        with contextlib.suppress(OSError, RuntimeError):
            self.pyaudio_inst.terminate()
        super().closeEvent(event)


def _debug_hook():
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook


def main():
    _debug_hook()

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(background="w", foreground="k")
    win = FFTAnalyserWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
