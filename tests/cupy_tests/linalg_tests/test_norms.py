import unittest

import numpy
import pytest

import cupy
from cupy import testing
import cupyx


@testing.gpu
class TestTrace(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_trace(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.trace(1, 3, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_trace(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return xp.trace(a, 1, 3, 2)


@testing.parameterize(*testing.product({
    'shape': [(1,), (2,)],
    'ord': [-numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf],
    'axis': [0, None],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(1, 2), (2, 2)],
    'ord': [-numpy.Inf, -2, -1, 1, 2, numpy.Inf, 'fro', 'nuc'],
    'axis': [(0, 1), None],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(2, 2, 2)],
    'ord': [-numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf],
    'axis': [0, 1, 2],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(2, 2, 2)],
    'ord': [-numpy.Inf, -1, 1, numpy.Inf, 'fro'],
    'axis': [(0, 1), (0, 2), (1, 2)],
    'keepdims': [True, False],
})
)
@testing.gpu
class TestNorm(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_norm(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        res = xp.linalg.norm(a, self.ord, self.axis, self.keepdims)
        if xp == numpy and not isinstance(res, numpy.ndarray):
            real_dtype = a.real.dtype
            if issubclass(real_dtype.type, numpy.inexact):
                # Avoid numpy bug. See numpy/numpy#10667
                res = res.astype(a.real.dtype)
        return res


@testing.parameterize(*testing.product({
    'array': [
        [[1, 2], [3, 4]],
        [[1, 2], [1, 2]],
        [[0, 0], [0, 0]],
        [1, 2],
        [0, 1],
        [0, 0],
    ],
    'tol': [None, 1]
}))
@testing.gpu
class TestMatrixRank(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_array_equal(type_check=True)
    def test_matrix_rank(self, xp, dtype):
        a = xp.array(self.array, dtype=dtype)
        y = xp.linalg.matrix_rank(a, tol=self.tol)
        if xp is cupy:
            assert isinstance(y, cupy.ndarray)
            assert y.shape == ()
        else:
            # Note numpy returns numpy scalar or python int
            y = xp.array(y)
        return y


@testing.gpu
class TestDet(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det(self, xp, dtype):
        a = testing.shaped_arange((2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_3(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_4(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_batch(self, xp, dtype):
        a = xp.empty((2, 0, 3, 3), dtype)
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_matrix(self, xp, dtype):
        a = xp.empty((0, 0), dtype)
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_matrices(self, xp, dtype):
        a = xp.empty((2, 3, 0, 0), dtype)
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_different_last_two_dims(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 2), xp, dtype)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_different_last_two_dims_empty_batch(self, dtype):
        for xp in (numpy, cupy):
            a = xp.empty((0, 3, 2), dtype)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_one_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2,), xp, dtype)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_singular(self, xp, dtype):
        a = xp.zeros((2, 3, 3), dtype)
        return xp.linalg.det(a)


@testing.gpu
class TestSlogdet(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet(self, xp, dtype):
        a = testing.shaped_arange((2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_3(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_4(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_singular(self, xp, dtype):
        a = xp.zeros((3, 3), dtype)
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_singular_errstate(self, xp, dtype):
        a = xp.zeros((3, 3), dtype)
        with cupyx.errstate(linalg='raise'):
            # `cupy.linalg.slogdet` internally catches `dev_info < 0` from
            # cuSOLVER, which should not affect `dev_info > 0` cases.
            sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes('fdFD')
    def test_slogdet_one_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2,), xp, dtype)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.slogdet(a)
