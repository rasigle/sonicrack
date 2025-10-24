import threading

import numpy as np
import pyaudio
from pynput import keyboard

from src.engine.generators import generate_waveform, WaveForm
from envelopes import envelope

# 🎧 Audio settings
fs = 44100
volume = 0.4
SAMPLE_SIZE = 1024

# 🎵 Keyboard → Frequency mapping (one octave + sharps)
KEY_FREQUENCIES = {
    "a": 261.63,  # C4
    "w": 277.18,  # C#4
    "s": 293.66,  # D4
    "e": 311.13,  # D#4
    "d": 329.63,  # E4
    "f": 349.23,  # F4
    "t": 369.99,  # F#4
    "g": 392.00,  # G4
    "y": 415.30,  # G#4
    "h": 440.00,  # A4
    "u": 466.16,  # A#4
    "j": 493.88,  # B4
    "k": 523.25,  # C5
}


# 🎛️ Waveform selection
waveform_type = WaveForm.SINE  # Default
WAVEFORMS = [WaveForm.SINE, WaveForm.SQUARE, WaveForm.SAW, WaveForm.TRIANGLE]

# Internal state
active_notes = {}  # key -> {'freq', 'phase', 'time', 'released'}
running = True


# ---------- SYNTH ENGINE ----------
def audio_thread():
    """Audio generation loop."""
    global running
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=fs,
        output=True,
        frames_per_buffer=SAMPLE_SIZE,
    )

    while running:
        samples = np.zeros(SAMPLE_SIZE, dtype=np.float32)
        dt = 1.0 / fs

        for key, note in list(active_notes.items()):
            freq = note["freq"]
            phase = note["phase"]
            t0 = note["time"]
            released = note["released"]
            release_start = note.get("release_start", None)

            # Envelope + waveform
            t_vals = np.arange(SAMPLE_SIZE) * dt
            amps = np.array(
                [envelope(t0 + t, released, release_start or 0) for t in t_vals]
            )
            wave = generate_waveform(waveform_type, freq, phase, SAMPLE_SIZE, fs)
            samples += wave * amps

            # Update note state
            note["time"] += SAMPLE_SIZE * dt
            note["phase"] += SAMPLE_SIZE
            if released and envelope(note["time"], True, release_start) <= 0:
                del active_notes[key]

        # Normalize
        if active_notes:
            samples /= len(active_notes)
        stream.write(
            (volume * samples).astype(np.float32).tobytes(),
            exception_on_underflow=False,
        )

    stream.stop_stream()
    stream.close()
    p.terminate()


# ---------- KEYBOARD HANDLERS ----------


def on_press(key):
    global waveform_type
    try:
        k = key.char.lower()

        # Switch waveforms
        if k in ["1", "2", "3", "4"]:
            waveform_type = WAVEFORMS[int(k) - 1]
            print(f"🎛️ Waveform: {waveform_type}")
            return

        # Play notes
        if k in KEY_FREQUENCIES and k not in active_notes:
            active_notes[k] = {
                "freq": KEY_FREQUENCIES[k],
                "phase": 0,
                "time": 0,
                "released": False,
            }
            print(f"🎵 Pressed: {k}")
    except AttributeError:
        pass


def on_release(key):
    try:
        k = key.char.lower()
        if k in active_notes and not active_notes[k]["released"]:
            active_notes[k]["released"] = True
            active_notes[k]["release_start"] = active_notes[k]["time"]
            print(f"Released: {k}")
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        global running
        running = False
        return False


# ---------- RUN SYNTH ----------

thread = threading.Thread(target=audio_thread, daemon=True)
thread.start()

print("🎹 Synth with ADSR + Waveform Selector")
print("Play notes: A–K (white) + W, E, T, Y, U (black)")
print("Switch waveforms:")
print("  [1] Sine   [2] Square   [3] Sawtooth   [4] Triangle")
print("Press ESC to quit.\n")

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

thread.join()
print("✅ Synth stopped.")
