"""Single-kernel device iteration with the private bytecode frontend.

Run with ``python -m examples.fusion.bytecode_iteration`` from the checkout.
Each decorated call launches one kernel and returns a scalar device array.
Large arrays use a cooperative grid; small arrays use a single block.
"""
from __future__ import annotations

import cupy as cp
from cupy._core._fusion_bytecode import fuse


def filter_agg(data):
    total = 0
    for x in data:
        if x % 2 == 0:
            total += 1
    return total


def sigma_clipped_mean(data, sigma=3.0, max_iter=5):
    mask = cp.zeros(data.shape, dtype=bool)
    for _ in range(max_iter):
        valid_data = data[~mask]
        if len(valid_data) == 0:
            break
        mean = cp.mean(valid_data)
        std = cp.std(valid_data)
        new_mask = cp.abs(data - mean) > (sigma * std)
        if cp.all(mask == (mask | new_mask)):
            break
        mask = mask | new_mask
    return cp.mean(data[~mask])


if __name__ == '__main__':
    count = fuse(filter_agg)
    clipped = fuse(sigma_clipped_mean)
    for size in (2048, 262144):
        data = cp.arange(-size // 2, size // 2, dtype=cp.float64)
        cp.testing.assert_array_equal(count(data),
                                      cp.count_nonzero(data % 2 == 0))
        cp.testing.assert_allclose(clipped(data, sigma=1.5),
                                   sigma_clipped_mean(data, sigma=1.5),
                                   atol=1e-5)
        for compiled in (count, clipped):
            print(compiled.__name__, size, compiled.diagnostics['backend'],
                  'launches:', compiled.diagnostics['kernel_launches'],
                  'schedule:', compiled.diagnostics['schedule'])
