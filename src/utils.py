import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf

from constants import DEFAULT_SAMPLE_RATE


def to_16(wav, amp):
    return np.int16(wav * amp * (2**15 - 1))


def play_wave(wav, sr: float = DEFAULT_SAMPLE_RATE, amp=0.5):
    import sounddevice as sd

    wav = to_16(np.array(wav), amp)
    sd.play(wav, samplerate=sr)
    sd.wait()


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


def read_wave_file(path: str):
    """Read a WAV file and return sample_rate, numpy array (mono)

    Args:
        path (str): Path to the WAV file.

    Returns:
        sample_rate (int): Sample rate of the audio file.
        data (np.ndarray): Mono audio data as a numpy array of type float32.

    """
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data.astype(np.float32)


def hz(note: str) -> float:
    return float(librosa.note_to_hz(note))
