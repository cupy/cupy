import unittest

import numpy

import cupy
from cupy import cuda
from cupy import testing


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestEigenvalue(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigh(self, xp, dtype):
        a = xp.array([[1, 0, 3], [0, 5, 0], [3, 0, 5]], dtype)
        w, v = xp.linalg.eigh(a)

        # Order of eigen values is not defined.
        # They must be sorted to compare them.
        if xp is numpy:
            inds = numpy.argsort(w)
        else:
            inds = cupy.array(numpy.argsort(w.get()))
        w = w[inds]
        v = v[inds]
        return xp.concatenate([w[None], v])

    def test_eigh_float16(self):
        # NumPy's eigh deos not support float16
        a = cupy.array([[1, 0, 3], [0, 5, 0], [3, 0, 5]], 'e')
        w, v = cupy.linalg.eigh(a)

        self.assertEqual(w.dtype, numpy.float16)
        self.assertEqual(v.dtype, numpy.float16)

        a = a.astype('d')
        w = w.astype('d')
        v = v.astype('d')

        testing.assert_allclose(a.dot(v), w * v, rtol=1e-3, atol=1e-4)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh(self, xp, dtype):
        a = xp.array([[1, 0, 3], [0, 5, 0], [3, 0, 5]], dtype)
        w = xp.linalg.eigvalsh(a)

        # Order of eigen values is not defined.
        # They must be sorted to compare them.
        if xp is numpy:
            inds = numpy.argsort(w)
        else:
            inds = cupy.array(numpy.argsort(w.get()))
        w = w[inds]
        return w
