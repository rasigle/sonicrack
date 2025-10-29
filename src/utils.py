import librosa
import numpy as np
from scipy.io import wavfile

from constants import DEFAULT_SAMPLE_RATE

to_16 = lambda wav, amp: np.int16(wav * amp * (2 ** 15 - 1))


def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1):
    if not fname.endswith(".wav"):
        fname += ".wav"
    wav = np.array(wav)
    wav = to_16(wav, amp)
    if wav2 is not None:
        wav2 = np.array(wav2)
        wav2 = to_16(wav2, amp)
        wav = np.stack([wav, wav2]).T

    wavfile.write(fname, DEFAULT_SAMPLE_RATE, wav)


def hz(note: str) -> float:
    return float(librosa.note_to_hz(note))
