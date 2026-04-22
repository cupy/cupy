from __future__ import annotations

import unittest

import numpy
import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(3, 4), (4, 3), (2, 3, 4)],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
class TestSvdvals(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_svdvals(self, xp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        if xp is numpy and not hasattr(numpy.linalg, 'svdvals'):
            # Fallback for older numpy
            return numpy.linalg.svd(a, compute_uv=False)
        return xp.linalg.svdvals(a)


@testing.parameterize(*testing.product({
    'shape': [(3, 4), (4, 3), (2, 3, 4)],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'ord': [None, 'fro', 'nuc', 1, -1, 2, -2, numpy.inf, -numpy.inf],
}))
class TestMatrixNorm(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_matrix_norm(self, xp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        if xp is numpy and not hasattr(numpy.linalg, 'matrix_norm'):
            # Fallback for older numpy
            return numpy.linalg.norm(a, axis=(-2, -1), ord=self.ord)
        return xp.linalg.matrix_norm(a, ord=self.ord)


@testing.parameterize(*testing.product({
    'shape': [(5,), (3, 4)],
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'ord': [2, 1, -1, numpy.inf, -numpy.inf, 0, 3],
    'axis': [None, 0, -1],
}))
class TestVectorNorm(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_vector_norm(self, xp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        if self.axis is not None and a.ndim <= self.axis:
            return xp.zeros(())  # Skip invalid axis

        if xp is numpy and not hasattr(numpy.linalg, 'vector_norm'):
            # Fallback for older numpy
            return numpy.linalg.norm(a, axis=self.axis, ord=self.ord)
        return xp.linalg.vector_norm(a, axis=self.axis, ord=self.ord)


@testing.parameterize(*testing.product({
    'shape': [(3, 3), (2, 4, 4)],
    'dtype': [numpy.float32, numpy.float64],
    'offset': [0, 1, -1],
}))
class TestLinalgTrace(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_trace(self, xp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        if xp is numpy and not hasattr(numpy.linalg, 'trace'):
            return numpy.trace(a, offset=self.offset, axis1=-2, axis2=-1)
        return xp.linalg.trace(a, offset=self.offset)


@testing.parameterize(*testing.product({
    'shape': [(3, 3), (2, 4, 4)],
    'dtype': [numpy.float32, numpy.float64],
    'offset': [0, 1, -1],
}))
class TestLinalgDiagonal(unittest.TestCase):

    @testing.numpy_cupy_allclose()
    def test_diagonal(self, xp):
        a = testing.shaped_random(self.shape, xp, self.dtype)
        if xp is numpy and not hasattr(numpy.linalg, 'diagonal'):
            return numpy.diagonal(a, offset=self.offset, axis1=-2, axis2=-1)
        return xp.linalg.diagonal(a, offset=self.offset)


class TestMultiDot(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_multi_dot_2(self, xp):
        a = testing.shaped_random((3, 4), xp)
        b = testing.shaped_random((4, 5), xp)
        return xp.linalg.multi_dot([a, b])

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_multi_dot_3(self, xp):
        a = testing.shaped_random((3, 4), xp)
        b = testing.shaped_random((4, 5), xp)
        c = testing.shaped_random((5, 2), xp)
        return xp.linalg.multi_dot([a, b, c])

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_multi_dot_4(self, xp):
        a = testing.shaped_random((10, 100), xp)
        b = testing.shaped_random((100, 5), xp)
        c = testing.shaped_random((5, 50), xp)
        d = testing.shaped_random((50, 20), xp)
        return xp.linalg.multi_dot([a, b, c, d])

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_multi_dot_vectors(self, xp):
        a = testing.shaped_random((10,), xp)
        b = testing.shaped_random((10, 5), xp)
        c = testing.shaped_random((5,), xp)
        return xp.linalg.multi_dot([a, b, c])

    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_multi_dot_out(self, xp):
        dtype = numpy.float64
        a = testing.shaped_random((3, 4), xp, dtype)
        b = testing.shaped_random((4, 5), xp, dtype)
        c = testing.shaped_random((5, 2), xp, dtype)
        out = xp.empty((3, 2), dtype=dtype)
        res = xp.linalg.multi_dot([a, b, c], out=out)
        self.assertIs(res, out)
        return out


class TestLinalgExports(unittest.TestCase):

    def test_exports(self):
        self.assertTrue(hasattr(cupy.linalg, 'matmul'))
        self.assertTrue(hasattr(cupy.linalg, 'outer'))
        self.assertTrue(hasattr(cupy.linalg, 'tensordot'))

    @testing.numpy_cupy_allclose()
    def test_matmul(self, xp):
        a = testing.shaped_random((3, 4), xp)
        b = testing.shaped_random((4, 5), xp)
        return xp.linalg.matmul(a, b)

    @testing.numpy_cupy_allclose()
    def test_outer(self, xp):
        a = testing.shaped_random((3,), xp)
        b = testing.shaped_random((4,), xp)
        return xp.linalg.outer(a, b)

    @testing.numpy_cupy_allclose()
    def test_tensordot(self, xp):
        a = testing.shaped_random((3, 4), xp)
        b = testing.shaped_random((4, 5), xp)
        return xp.linalg.tensordot(a, b, axes=1)
