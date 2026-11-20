# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-device summary of a time-ordered event log. NumPy version."""

import numpy as np

N_EVENTS = 2_000_000
N_DEVICES = 5_000


def make_events(seed=0):
    rng = np.random.default_rng(seed)
    device = rng.integers(0, N_DEVICES, N_EVENTS)          # events are in time order
    drift = np.linspace(0.0, 3.0, N_EVENTS)
    value = 20.0 + drift + rng.standard_normal(N_EVENTS)
    return device, value


def summarize(device, value):
    last = np.full(N_DEVICES, np.nan)
    last[device] = value                                   # latest reading per device
    last_seen = np.full(N_DEVICES, -1)
    last_seen[device] = np.arange(device.size)             # index of the latest event
    total = np.zeros(N_DEVICES)
    np.add.at(total, device, value)
    count = np.bincount(device, minlength=N_DEVICES)
    return last, last_seen, total / count


if __name__ == "__main__":
    device, value = make_events()
    last, last_seen, mean = summarize(device, value)
    print(f"last_sum={last.sum():.6f} last_seen_sum={int(last_seen.sum())} "
          f"mean_sum={mean.sum():.6f}")
    print(f"devices above their own mean at the end: {int((last > mean).sum())}")
