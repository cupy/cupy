# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whiten batches of float32 sensor frames on the GPU: FIR-normalize, decorrelate, spectrum."""

import time

import cupy as cp
import numpy as np
from scipy import signal

N_FRAMES = 2048
FRAME = 2048


def design():
    taps = signal.firwin(63, 0.25)            # designed once on the CPU
    gain = np.linalg.norm(taps)               # normalization constant
    return taps, gain


def whiten(frames, mix, gain):
    x = frames / gain                         # normalize
    x = x - x.mean(axis=1, keepdims=True)
    y = x @ mix                               # decorrelate channels
    spec = cp.fft.rfft(y, axis=1)
    return cp.abs(spec).mean(axis=0)


if __name__ == "__main__":
    rng = cp.random.default_rng(0)
    frames = rng.standard_normal((N_FRAMES, FRAME), dtype=cp.float32)
    mix = rng.standard_normal((FRAME, FRAME), dtype=cp.float32) / np.float32(FRAME ** 0.5)
    _, gain = design()

    whiten(frames, mix, gain)                 # warm-up
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = whiten(frames, mix, gain)
    cp.cuda.Device().synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1e3
    print(f"{ms:.2f} ms per batch  checksum={float(out.sum()):.3f}")
