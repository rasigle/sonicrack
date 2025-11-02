import numpy as np
import sounddevice as sd
import threading
import tkinter as tk
from tkinter import ttk


class SequencerAudio:
    def __init__(self, freqs, sample_rate=44100, gain=0.2, attack_ms=5, release_ms=5):
        self.freqs = np.array(freqs, dtype=float)
        self.nsteps = len(freqs)
        self.sr = sample_rate
        self.gain = gain
        self.attack_samps = int(attack_ms * 0.001 * self.sr)
        self.release_samps = int(release_ms * 0.001 * self.sr)
        self.phase = 0.0
        self.two_pi = 2 * np.pi
        self.current_step = 0
        self.samples_in_step = 0
        self.samples_generated = 0
        self.tempo = 120.0  # BPM (modifiable from GUI)
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        self.stream = sd.OutputStream(
            channels=1, callback=self.audio_callback, samplerate=self.sr, blocksize=512
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

    def set_tempo(self, bpm):
        with self.lock:
            self.tempo = bpm

    def _step_duration_samples(self):
        # 1 beat per step => 60 / bpm seconds
        with self.lock:
            bpm = self.tempo
        dur_sec = 60.0 / bpm
        return int(dur_sec * self.sr)

    def audio_callback(self, outdata, frames, time, status):
        if not self.running:
            outdata[:] = np.zeros((frames, 1))
            return
        buf = np.zeros(frames, dtype=np.float32)
        idx = 0
        while idx < frames:
            if self.samples_in_step == 0:
                # start new step
                self.samples_in_step = self._step_duration_samples()
                self.samples_generated = 0
                self.freq = self.freqs[self.current_step]
                self.current_step = (self.current_step + 1) % self.nsteps

            # generate up to end of this step
            remain_step = self.samples_in_step - self.samples_generated
            n = min(frames - idx, remain_step)
            t = np.arange(n)
            incr = self.two_pi * self.freq / self.sr
            phases = self.phase + incr * t
            wave = np.sin(phases).astype(np.float32)
            self.phase = float((phases[-1] + incr) % self.two_pi)

            # simple linear attack/release envelope
            env = np.ones(n, dtype=np.float32)
            if self.samples_generated < self.attack_samps:
                a = min(self.attack_samps - self.samples_generated, n)
                env[:a] = np.linspace(
                    self.samples_generated / self.attack_samps, 1.0, a, endpoint=False
                )
            if remain_step - n < self.release_samps:
                r = min(self.release_samps, remain_step)
                env[-r:] = np.linspace(1.0, 0.0, r, endpoint=False)[-n:]
            buf[idx : idx + n] = wave * env * self.gain
            idx += n
            self.samples_generated += n
            if self.samples_generated >= self.samples_in_step:
                self.samples_in_step = 0
        outdata[:, 0] = buf


class TempoGUI:
    def __init__(self, sequencer):
        self.seq = sequencer
        self.root = tk.Tk()
        self.root.title("Live Tempo Sequencer")
        self.root.geometry("300x150")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        label = ttk.Label(self.root, text="Tempo (BPM)", font=("Arial", 12))
        label.pack(pady=10)

        self.slider = ttk.Scale(
            self.root, from_=60, to=200, orient="horizontal", command=self.on_slider
        )
        self.slider.set(120)
        self.slider.pack(fill="x", padx=20)

        self.value_label = ttk.Label(self.root, text="120 BPM")
        self.value_label.pack(pady=10)

        # start audio thread
        threading.Thread(target=self.seq.start, daemon=True).start()

    def on_slider(self, val):
        bpm = float(val)
        self.seq.set_tempo(bpm)
        self.value_label.config(text=f"{bpm:.1f} BPM")

    def on_close(self):
        self.seq.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Example 16-step pattern (C major arpeggio)
    freqs = [
        261.63,
        329.63,
        392.00,
        523.25,
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
    seq = SequencerAudio(freqs)
    gui = TempoGUI(seq)
    gui.run()
