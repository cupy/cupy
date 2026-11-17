# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-channel sensor pipeline: band-pass filter, PSD, peak frequency. NumPy/SciPy version."""

import time

import numpy as np
from scipy import signal

FS = 10_000
N_CHANNELS = 32
N_SAMPLES = 1_000_000


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    freqs = rng.uniform(800, 1500, size=(N_CHANNELS, 1))
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * freqs * t)
    x += 0.3 * rng.standard_normal((N_CHANNELS, N_SAMPLES))
    return x, freqs.ravel()


def process(x):
    sos = signal.butter(8, [500, 2000], btype="bandpass", fs=FS, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=-1)
    f, psd = signal.welch(y, fs=FS, nperseg=8192, axis=-1)
    peaks = f[np.argmax(psd, axis=-1)]
    rms = np.sqrt(np.mean(y ** 2, axis=-1))
    return peaks, rms


if __name__ == "__main__":
    x, true_freqs = make_data()
    start = time.time()
    peaks, rms = process(x)
    elapsed = time.time() - start
    err = np.abs(peaks - true_freqs).max()
    print(f"max peak error={err:.2f} Hz  mean rms={rms.mean():.4f}  time={elapsed:.2f}s")
    np.save("peaks.npy", peaks)
