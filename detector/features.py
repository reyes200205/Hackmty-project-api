"""Extraccion de features acusticas para deteccion de voz sintetica (A2).

Todo se calcula solo a partir del canal del caller, sin depender de los
turns/*.json del dataset (esos no estan disponibles en /detect, donde solo
llega el audio), para que el pipeline de entrenamiento y el de inferencia
usen exactamente la misma señal.
"""
from __future__ import annotations

import functools

import numpy as np
import scipy.fft
from scipy.fftpack import dct

SAMPLE_RATE_DEFAULT = 8000
FRAME_MS = 25.0
HOP_MS = 10.0
N_MELS = 26
N_MFCC = 13
PITCH_MIN_HZ = 75.0
PITCH_MAX_HZ = 400.0
VOICED_ENERGY_PERCENTILE = 40.0
VOICED_STRENGTH_THRESHOLD = 0.25
DIGITAL_SILENCE_THRESHOLD = 1e-5

FEATURE_NAMES: list[str] = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + [
        "pitch_mean",
        "pitch_std",
        "jitter",
        "shimmer",
        "voiced_fraction",
        "spectral_flatness",
        "noise_floor_db",
        "digital_silence_ratio",
        "rms_db_mean",
        "rms_db_std",
    ]
)


def _frame_signal(x: np.ndarray, sample_rate: int, frame_ms: float = FRAME_MS, hop_ms: float = HOP_MS) -> np.ndarray:
    frame_len = max(int(sample_rate * frame_ms / 1000), 1)
    hop_len = max(int(sample_rate * hop_ms / 1000), 1)
    if len(x) < frame_len:
        return np.zeros((0, frame_len), dtype=x.dtype)
    n_frames = 1 + (len(x) - frame_len) // hop_len
    shape = (n_frames, frame_len)
    strides = (x.strides[0] * hop_len, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@functools.lru_cache(maxsize=8)
def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    low_mel = _hz_to_mel(np.array(0.0))
    high_mel = _hz_to_mel(np.array(sample_rate / 2.0))
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        f_center = max(f_center, f_left + 1)
        f_right = max(f_right, f_center + 1)
        for k in range(f_left, f_center):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


def compute_mfcc(
    x_or_frames: np.ndarray,
    sample_rate: int,
    n_mfcc: int = N_MFCC,
    n_mels: int = N_MELS,
) -> np.ndarray:
    if x_or_frames.ndim == 1:
        frames = _frame_signal(x_or_frames, sample_rate)
    else:
        frames = x_or_frames
    if frames.shape[0] == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)

    emphasized = np.empty_like(frames, dtype=np.float32)
    emphasized[:, 0] = frames[:, 0]
    emphasized[:, 1:] = frames[:, 1:] - np.float32(0.97) * frames[:, :-1]
    window = np.hamming(frames.shape[1]).astype(np.float32)
    windowed = emphasized * window

    n_fft = frames.shape[1]
    mag = np.abs(scipy.fft.rfft(windowed, n=n_fft, axis=1, workers=1))
    power = (mag ** 2) * np.float32(1.0 / n_fft)

    fbank = _mel_filterbank(sample_rate, n_fft, n_mels)
    mel_energy = power @ fbank.T
    np.maximum(mel_energy, 1e-12, out=mel_energy)
    log_mel = np.log(mel_energy)

    mfcc = dct(log_mel.astype(np.float64), type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfcc


def spectral_flatness(x: np.ndarray, frame: int = 1024, hop: int = 512) -> float:
    if len(x) < frame:
        return 0.0
    n_frames = 1 + (len(x) - frame) // hop
    shape = (n_frames, frame)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    window = np.hanning(frame).astype(np.float32)
    mag = np.abs(scipy.fft.rfft(frames * window, axis=1, workers=1)) + np.float32(1e-10)
    gm = np.exp(np.mean(np.log(mag), axis=1))
    am = np.mean(mag, axis=1)
    return float(np.mean(gm / am))


def _autocorr_pitch(frames: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Pitch (F0) y fuerza de voicing por frame via autocorrelacion (Wiener-Khinchin, vectorizado en float32)."""
    n_frames, frame_len = frames.shape
    f0 = np.zeros(n_frames, dtype=np.float32)
    strength = np.zeros(n_frames, dtype=np.float32)

    min_lag = int(sample_rate / PITCH_MAX_HZ)
    max_lag = min(int(sample_rate / PITCH_MIN_HZ), frame_len - 1)
    if n_frames == 0 or max_lag <= min_lag:
        return f0, strength

    window = np.hamming(frame_len).astype(np.float32)
    seg = frames * window
    seg -= seg.mean(axis=1, keepdims=True)
    energy = np.sum(seg ** 2, axis=1)

    n_fft = 1
    while n_fft < 2 * frame_len:
        n_fft *= 2
    spec = scipy.fft.rfft(seg, n=n_fft, axis=1, workers=1)
    ac_full = scipy.fft.irfft(spec * np.conj(spec), n=n_fft, axis=1, workers=1)
    ac = ac_full[:, :frame_len]

    segment = ac[:, min_lag:max_lag]
    peak_idx = np.argmax(segment, axis=1)
    peak_val = segment[np.arange(n_frames), peak_idx]
    norm_peak = peak_val / (energy + np.float32(1e-12))
    lag = peak_idx + min_lag

    good = (energy > 1e-9) & (norm_peak > VOICED_STRENGTH_THRESHOLD) & (lag > 0)
    f0[good] = sample_rate / lag[good]
    strength[good] = norm_peak[good]
    return f0, strength


def _jitter_shimmer(f0: np.ndarray, voiced: np.ndarray, amplitudes: np.ndarray) -> tuple[float, float]:
    idx = np.where(voiced)[0]
    if len(idx) < 3:
        return 0.0, 0.0

    periods = 1.0 / f0[idx]
    amps = amplitudes[idx]
    consecutive = np.diff(idx) == 1

    period_diffs = np.abs(np.diff(periods))[consecutive]
    amp_diffs = np.abs(np.diff(amps))[consecutive]

    mean_period = np.mean(periods)
    mean_amp = np.mean(amps) + 1e-12

    jitter = float(np.mean(period_diffs) / mean_period) if len(period_diffs) else 0.0
    shimmer = float(np.mean(amp_diffs) / mean_amp) if len(amp_diffs) else 0.0
    return jitter, shimmer


def extract_features(x: np.ndarray, sample_rate: int, max_seconds: float | None = 60.0) -> dict[str, float]:
    if max_seconds is not None and len(x) > int(max_seconds * sample_rate):
        x = x[:int(max_seconds * sample_rate)]
    x = np.asarray(x, dtype=np.float32)
    frames = _frame_signal(x, sample_rate)

    if frames.shape[0] == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    energy_threshold = np.percentile(rms, VOICED_ENERGY_PERCENTILE)
    voiced_energy_mask = rms > energy_threshold

    # Calculo de pitch solo sobre frames candidatos con energia suficiente (ahorro de ~60% computo)
    n_frames = frames.shape[0]
    f0 = np.zeros(n_frames, dtype=np.float32)
    if np.any(voiced_energy_mask):
        sub_frames = frames[voiced_energy_mask]
        sub_f0, _ = _autocorr_pitch(sub_frames, sample_rate)
        f0[voiced_energy_mask] = sub_f0

    voiced_mask = voiced_energy_mask & (f0 > 0)
    jitter, shimmer = _jitter_shimmer(f0, voiced_mask, rms)

    mfcc = compute_mfcc(frames, sample_rate)
    if mfcc.shape[0] > 0:
        mfcc_mean = mfcc.mean(axis=0)
        mfcc_std = mfcc.std(axis=0)
    else:
        mfcc_mean = np.zeros(N_MFCC)
        mfcc_std = np.zeros(N_MFCC)

    pitch_values = f0[voiced_mask]
    pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) else 0.0
    pitch_std = float(np.std(pitch_values)) if len(pitch_values) else 0.0

    features: dict[str, float] = {}
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
        features[f"mfcc_{i}_std"] = float(mfcc_std[i])

    features.update({
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "voiced_fraction": float(np.mean(voiced_mask)),
        "spectral_flatness": spectral_flatness(x),
        "noise_floor_db": float(np.percentile(rms_db, 10)),
        "digital_silence_ratio": float(np.mean(np.abs(x) < DIGITAL_SILENCE_THRESHOLD)),
        "rms_db_mean": float(np.mean(rms_db)),
        "rms_db_std": float(np.std(rms_db)),
    })
    return features


def feature_vector(x: np.ndarray, sample_rate: int, max_seconds: float | None = 60.0) -> np.ndarray:
    features = extract_features(x, sample_rate, max_seconds=max_seconds)
    vec = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

