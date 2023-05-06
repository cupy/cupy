
from math import sqrt, pi

import cupy
import cupyx.scipy.signal as signal
from cupy import testing
from cupy.testing import assert_array_almost_equal, assert_allclose

import numpy as np

import pytest
from pytest import raises as assert_raises


@testing.with_requires("scipy")
class TestBilinear_zpk:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3

        z_d, p_d, k_d = scp.signal.bilinear_zpk(z, p, k, 10)
        return z_d, p_d, k_d

        """
        assert_allclose(sort(z_d), sort([(20-2j)/(20+2j), (20+2j)/(20-2j),
                                         -1]))
        assert_allclose(sort(p_d), sort([77/83,
                                         (1j/2 + 39/2) / (41/2 - 1j/2),
                                         (39/2 - 1j/2) / (1j/2 + 41/2), ]))
        assert_allclose(k_d, 9696/69803)
        """


@testing.with_requires("scipy")
class TestBilinear:

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic(self, xp, scp):
        b = [0.14879732743343033]
        a = [1, 0.54552236880522209, 0.14879732743343033]
        b_z, a_z = scp.signal.bilinear(b, a, 0.5)
        return b_z, a_z

#        assert_array_almost_equal(b_z, [0.087821, 0.17564, 0.087821],
#                                  decimal=5)
#        assert_array_almost_equal(a_z, [1, -1.0048, 0.35606], decimal=4)

    @testing.numpy_cupy_allclose(scipy_name="scp")
    def test_basic_2(self, xp, scp):
        b = [1, 0, 0.17407467530697837]
        a = [1, 0.18460575326152251, 0.17407467530697837]
        b_z, a_z = scp.signal.bilinear(b, a, 0.5)
        return b_z, a_z

#        assert_array_almost_equal(b_z, [0.86413, -1.2158, 0.86413],
#                                  decimal=4)
#        assert_array_almost_equal(a_z, [1, -1.2158, 0.72826],
#                                  decimal=4)
