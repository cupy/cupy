# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regrid scattered station measurements onto a regular grid. NumPy/SciPy version."""

import time

import numpy as np
from scipy.interpolate import griddata

N_STATIONS = 150_000
GRID = 1024


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((N_STATIONS, 2))
    vals = np.sin(6.0 * pts[:, 0]) * np.cos(4.0 * pts[:, 1]) + 0.1 * pts[:, 0]
    return pts, vals


def regrid(pts, vals):
    axis = np.linspace(0.02, 0.98, GRID)
    gx, gy = np.meshgrid(axis, axis)
    return griddata(pts, vals, (gx, gy), method="linear")


if __name__ == "__main__":
    pts, vals = make_data()
    start = time.perf_counter()
    field = regrid(pts, vals)
    elapsed = time.perf_counter() - start
    print(f"mean={field.mean():.8f} max={field.max():.8f} time={elapsed:.2f}s")
    np.save("field.npy", field)
