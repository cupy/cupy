
import pytest

import cupy
from cupy import testing
from cupyx.scipy.signal._iir_utils import apply_iir

import numpy as np


class TestIIRUtils:
    def _sequential_impl(self, x, b):
        t = x.copy()
        y = cupy.empty_like(x, dtype=cupy.float64)

        n = t.size
        k = b.size

        for i in range(n):
            y[i] = t[i]
            for j in range(1, min(i, k) + 1):
                y[i] += b[j - 1] * y[i - j]
        return y

    @pytest.mark.parametrize('size', [11, 20, 51, 120, 128, 250])
    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
    def test_order(self, size, order):
        signs = cupy.tile([1, -1], int(2 * np.ceil(size / 4)))
        x = cupy.arange(3, size + 3, dtype=cupy.float64) * signs[:size]
        b = testing.shaped_random((order,))
        seq = self._sequential_impl(x, b)
        par = apply_iir(x, b)
        testing.assert_allclose(seq, par)
