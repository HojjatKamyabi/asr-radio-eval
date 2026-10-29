import numpy as np
import pytest

from asr_eval.degrade import SAMPLE_RATE, add_noise, bandpass, degrade

T = np.arange(SAMPLE_RATE) / SAMPLE_RATE


def tone(freq):
    return 0.1 * np.sin(2 * np.pi * freq * T)


def rms(x):
    return np.sqrt(np.mean(x**2))


def test_same_length():
    for condition in ["clean", "band", "band_snr0"]:
        assert len(degrade(tone(1000), condition, seed=1)) == SAMPLE_RATE


def test_bandpass_removes_100hz_keeps_1khz():
    assert rms(bandpass(tone(100))) < 0.01 * rms(tone(100))
    assert rms(bandpass(tone(1000))) == pytest.approx(rms(tone(1000)), rel=0.05)


@pytest.mark.parametrize("snr", [20, 10, 5, 0])
def test_measured_snr_matches_target(snr):
    signal = bandpass(tone(1000))
    noise = add_noise(signal, snr, seed=3) - signal
    measured = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    assert abs(measured - snr) < 0.5


def test_same_seed_same_output():
    a = degrade(tone(1000), "band_snr5", seed=7)
    assert np.array_equal(a, degrade(tone(1000), "band_snr5", seed=7))
    assert not np.array_equal(a, degrade(tone(1000), "band_snr5", seed=8))
