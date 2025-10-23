import numpy as np
import sounddevice as sd
import threading
import tkinter as tk
from tkinter import ttk


class SmoothSequencer:
    def __init__(self, freqs, sample_rate=44100, gain=0.2, freq_ramp_ms=10.0):
        self.freqs = np.array(freqs, dtype=float)
        self.nsteps = len(freqs)
        self.sr = sample_rate
        self.gain = gain
        self.phase = 0.0
        self.two_pi = 2 * np.pi
        self.current_step = 0
        self.samples_generated = 0
        self.samples_in_step = 0
        self.tempo = 120.0
        self.waveform = "Sine"  # Default waveform
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.freq_ramp_samps = int(freq_ramp_ms * 0.001 * self.sr)
        self.f_curr = self.freqs[0]
        self.f_next = self.freqs[1]
        self.step_callback = None  # GUI callback to update step display

    def _step_duration_samples(self):
        with self.lock:
            bpm = self.tempo
        dur_sec = 60.0 / bpm
        return int(dur_sec * self.sr)

    def _generate_wave(self, phases):
        """Return samples for current waveform."""
        wf = self.waveform.lower()
        if wf == "sine":
            return np.sin(phases)
        elif wf == "saw":
            # sawtooth: -1 to 1
            return 2.0 * (phases / (2 * np.pi)) - 1.0
        elif wf == "triangle":
            # triangle wave (symmetric)
            return (
                2.0
                * np.abs(
                    2.0 * (phases / (2 * np.pi) - np.floor(phases / (2 * np.pi) + 0.5))
                )
                - 1.0
            )
        else:
            return np.sin(phases)

    def _audio_callback(self, outdata, frames, time, status):
        if not self.running:
            outdata[:] = np.zeros((frames, 1))
            return

        buf = np.zeros(frames, dtype=np.float32)
        idx = 0

        while idx < frames:
            if self.samples_in_step == 0:
                # new step
                self.samples_in_step = self._step_duration_samples()
                self.samples_generated = 0
                self.f_curr = self.f_next
                self.current_step = (self.current_step + 1) % self.nsteps
                self.f_next = self.freqs[self.current_step]
                # notify GUI of new step
                if self.step_callback:
                    try:
                        self.step_callback(self.current_step)
                    except Exception:
                        pass

            remain = self.samples_in_step - self.samples_generated
            n = min(frames - idx, remain)
            t = np.arange(n)

            # frequency ramp near end of note
            if remain < self.freq_ramp_samps:
                f = np.linspace(self.f_curr, self.f_next, n, dtype=np.float32)
            else:
                f = np.full(n, self.f_curr, dtype=np.float32)

            incr = self.two_pi * f / self.sr
            phases = self.phase + np.cumsum(incr)
            self.phase = float(phases[-1] % self.two_pi)

            wave = self._generate_wave(phases).astype(np.float32)
            buf[idx : idx + n] = wave * self.gain

            idx += n
            self.samples_generated += n
            if self.samples_generated >= self.samples_in_step:
                self.samples_in_step = 0

        outdata[:, 0] = buf

    def _audio_thread(self):
        """Thread that runs sounddevice stream."""
        try:
            with sd.OutputStream(
                channels=1,
                samplerate=self.sr,
                blocksize=512,
                callback=self._audio_callback,
            ):
                while self.running:
                    sd.sleep(100)
        except Exception as e:
            print("Audio thread error:", e)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._audio_thread, daemon=True)
        self.thread.start()
        print("Sequencer started")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("Sequencer stopped")

    def set_tempo(self, bpm):
        with self.lock:
            self.tempo = bpm

    def set_waveform(self, name):
        with self.lock:
            self.waveform = name


class TempoGUI:
    def __init__(self, sequencer):
        self.seq = sequencer
        self.root = tk.Tk()
        self.root.title("Interactive Sequencer")
        self.root.geometry("420x320")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        ttk.Label(self.root, text="Tempo (BPM)", font=("Arial", 12)).pack(pady=5)
        self.slider = ttk.Scale(
            self.root, from_=60, to=200, orient="horizontal", command=self.on_slider
        )
        self.slider.set(120)
        self.slider.pack(fill="x", padx=20)
        self.value_label = ttk.Label(self.root, text="120 BPM")
        self.value_label.pack(pady=5)

        # Waveform selector
        ttk.Label(self.root, text="Waveform:", font=("Arial", 12)).pack(pady=5)
        self.wave_select = ttk.Combobox(
            self.root, values=["Sine", "Saw", "Triangle"], state="readonly"
        )
        self.wave_select.current(0)
        self.wave_select.bind("<<ComboboxSelected>>", self.on_waveform)
        self.wave_select.pack(pady=5)

        # Step indicator (LED-like labels)
        self.step_labels = []
        steps_frame = ttk.Frame(self.root)
        steps_frame.pack(pady=10)
        for i in range(self.seq.nsteps):
            lbl = tk.Label(
                steps_frame,
                text=str(i + 1),
                width=2,
                height=1,
                bg="#333333",
                fg="white",
                font=("Arial", 10, "bold"),
                relief="flat",
            )
            lbl.grid(row=0, column=i, padx=2)
            self.step_labels.append(lbl)

        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Start", command=self.on_start).pack(
            side="left", padx=10
        )
        ttk.Button(btn_frame, text="Stop", command=self.on_stop).pack(
            side="left", padx=10
        )

        # Link callback
        self.seq.step_callback = self.update_step

    def on_slider(self, val):
        bpm = float(val)
        self.seq.set_tempo(bpm)
        self.value_label.config(text=f"{bpm:.1f} BPM")

    def on_waveform(self, event):
        name = self.wave_select.get()
        self.seq.set_waveform(name)

    def update_step(self, idx):
        """Called from audio thread; schedule safely in main thread."""
        self.root.after(0, self._update_step_ui, idx)

    def _update_step_ui(self, idx):
        for i, lbl in enumerate(self.step_labels):
            lbl.config(bg="#FF4444" if i == idx else "#333333")

    def on_start(self):
        self.seq.start()

    def on_stop(self):
        self.seq.stop()
        # Reset step lights
        for lbl in self.step_labels:
            lbl.config(bg="#333333")

    def on_close(self):
        self.seq.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    freqs = [
        261.63,
        0,
        392.00,
        0,
        440.00,
        392.00,
        329.63,
        261.63,
        196.00,
        246.94,
        293.66,
        349.23,
        392.00,
        440.00,
        493.88,
        523.25,
    ]
    seq = SmoothSequencer(freqs)
    gui = TempoGUI(seq)
    gui.run()
