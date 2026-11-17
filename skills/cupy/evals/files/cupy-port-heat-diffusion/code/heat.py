# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""2D heat diffusion (Jacobi iteration) on a square plate, NumPy version."""

import time

import numpy as np


def solve(n=1024, tol=1e-3, max_iter=300):
    u = np.zeros((n, n), dtype=np.float64)
    u[0, :] = 100.0  # hot top edge
    for it in range(max_iter):
        new = u.copy()
        new[1:-1, 1:-1] = 0.25 * (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
        )
        diff = np.abs(new - u).max()
        u = new
        if diff < tol:
            break
    return u, it


def row_profile(u):
    profile = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        profile[i] = u[i].mean()
    return profile


if __name__ == "__main__":
    start = time.time()
    u, iters = solve()
    profile = row_profile(u)
    elapsed = time.time() - start
    print(f"iters={iters} mean={u.mean():.6f} profile[10]={profile[10]:.6f} "
          f"time={elapsed:.2f}s")
    np.save("profile.npy", profile)
