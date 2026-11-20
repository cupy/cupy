# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gray-Scott reaction-diffusion on a small 192x192 grid, many time steps (CuPy)."""

import time

import cupy as cp
import numpy as np

N = 192
STEPS = 2_000
DU, DV, F, K = 0.16, 0.08, 0.035, 0.065


def laplacian(z):
    return (cp.roll(z, 1, 0) + cp.roll(z, -1, 0) + cp.roll(z, 1, 1)
            + cp.roll(z, -1, 1) - 4 * z)


def run():
    u = cp.ones((N, N), dtype=cp.float32)
    v = cp.zeros((N, N), dtype=cp.float32)
    c = N // 2
    u[c - 8:c + 8, c - 8:c + 8] = 0.5
    v[c - 8:c + 8, c - 8:c + 8] = 0.25
    for _ in range(STEPS):
        uvv = u * v * v
        u = u + DU * laplacian(u) - uvv + F * (1 - u)
        v = v + DV * laplacian(v) + uvv - (F + K) * v
    return u, v


if __name__ == "__main__":
    run()  # warm-up
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    u, v = run()
    cp.cuda.Device().synchronize()
    print(f"time={time.perf_counter() - t0:.2f}s  "
          f"u_mean={float(u.mean()):.6f} v_mean={float(v.mean()):.6f}")
    np.save("v_final.npy", v.get())
