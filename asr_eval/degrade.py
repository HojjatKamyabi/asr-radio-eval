# Simple radio channel: band-pass filter plus white noise.
import numpy as np
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 16000
CONDITIONS = ["clean", "band", "band_snr20", "band_snr10", "band_snr5", "band_snr0"]
SOS = butter(4, [300, 3000], btype="bandpass", fs=SAMPLE_RATE, output="sos")


def bandpass(audio):
    return sosfiltfilt(SOS, audio)


def add_noise(audio, snr_db, seed):
    noise = np.random.default_rng(seed).standard_normal(len(audio))
    return audio + noise * np.sqrt(np.mean(audio**2) / 10 ** (snr_db / 10))


def degrade(audio, condition, seed):
    if condition == "clean":
        return audio
    out = bandpass(audio)
    if condition != "band":
        snr = float(condition.removeprefix("band_snr"))
        out = add_noise(out, snr, seed)
    return np.clip(out, -1, 1).astype(np.float32)


def noise_only(audio, seed):
    # band-passed noise with the same length and power as the message, but no speech
    speech = bandpass(audio)
    noise = bandpass(np.random.default_rng(seed).standard_normal(len(audio)))
    noise *= np.sqrt(np.mean(speech**2) / np.mean(noise**2))
    return np.clip(noise, -1, 1).astype(np.float32)
