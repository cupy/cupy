"""Focused cooperative-kernel memory probe; invocation is in README.txt."""
from __future__ import annotations

import numpy as np
import cupy as cp

from cupy._core._fusion_bytecode import fuse
from examples.fusion.bytecode_iteration import filter_agg, sigma_clipped_mean


def main():
    host = np.ones(65537, dtype='float64')
    host[-1] = 100000
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        data = cp.asarray(host)
        count = fuse(filter_agg)
        clipped = fuse(sigma_clipped_mean)
        a = count(data)
        b = clipped(data)
    stream.synchronize()
    assert a.item() == 1
    assert b.item() == 1
    assert count.diagnostics['schedule']['blocks'] > 1
    assert clipped.diagnostics['schedule']['blocks'] > 1
    print('multi-block count and iterative mask update passed')


if __name__ == '__main__':
    main()
