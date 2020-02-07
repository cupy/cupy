import itertools
import unittest

import cupy
import numpy

from cupy import testing
import cupyx.scipy.special


@testing.gpu
@testing.with_requires('scipy')
class TestSpecialConvex(unittest.TestCase):

    def test_huber_basic(self):
        assert cupyx.scipy.special.huber(-1, 1.5) == cupy.inf
        testing.assert_allclose(cupyx.scipy.special.huber(2, 1.5),
                                0.5 * 1.5**2)
        testing.assert_allclose(cupyx.scipy.special.huber(2, 2.5),
                                2 * (2.5 - 0.5 * 2))

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_huber(self, xp, scp, dtype):
        import scipy.special  # NOQA
        z = testing.shaped_random((10, 2), xp=xp, dtype=dtype)
        return scp.special.huber(z[:, 0], z[:, 1])

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_entr(self, xp, scp, dtype):
        import scipy.special  # NOQA
        values = (0, 0.5, 1.0, cupy.inf)
        signs = [-1, 1]
        arr = []
        for sgn, v in itertools.product(signs, values):
            arr.append(sgn * v)
        z = xp.asarray(arr, dtype=dtype)
        return scp.special.entr(z)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rel_entr(self, xp, scp, dtype):
        import scipy.special  # NOQA
        values = (0, 0.5, 1.0)
        signs = [-1, 1]
        arr = []
        for sgna, va, sgnb, vb in itertools.product(signs, values, signs,
                                                    values):
            arr.append((sgna*va, sgnb*vb))
        z = xp.asarray(numpy.array(arr, dtype=dtype))
        return scp.special.kl_div(z[:, 0], z[:, 1])

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pseudo_huber(self, xp, scp, dtype):
        import scipy.special  # NOQA
        z = testing.shaped_random((10, 2), xp=numpy, dtype=dtype).tolist()
        z = xp.asarray(z + [[0, 0.5], [0.5, 0]], dtype=dtype)
        return scp.special.pseudo_huber(z[:, 0], z[:, 1])
