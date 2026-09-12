from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing
import cupyx


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
    'ord': [-numpy.inf, -2, -1, 0, 1, 2, 3, numpy.inf],
    'axis': [0, None],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(1, 2), (2, 2)],
    'ord': [-numpy.inf, -2, -1, 1, 2, numpy.inf, 'fro', 'nuc'],
    'axis': [(0, 1), None],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(2, 2, 2)],
    'ord': [-numpy.inf, -2, -1, 0, 1, 2, 3, numpy.inf],
    'axis': [0, 1, 2],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(2, 2, 2)],
    'ord': [-numpy.inf, -1, 1, numpy.inf, 'fro'],
    'axis': [(0, 1), (0, 2), (1, 2)],
    'keepdims': [True, False],
})
)
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


@testing.parameterize(
    *testing.product({"ord": [-numpy.inf, -2, -1, 1, 2, numpy.inf, "fro"]})
)
class TestCond(unittest.TestCase):
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_singular_zeros(self, xp, dtype):
        A = xp.zeros(shape=(2, 2), dtype=dtype)
        result = xp.linalg.cond(A, self.ord)

        # singular matrices don't always hit infinity.
        result = xp.asarray(result)  # numpy is scalar and can't be replaced
        large_number = 1.0 / (xp.finfo(dtype).eps)
        result[result >= large_number] = xp.inf

        return result

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_singular_ones(self, xp, dtype):
        A = xp.ones(shape=(2, 2), dtype=dtype)
        result = xp.linalg.cond(A, self.ord)

        # singular matrices don't always hit infinity.
        result = xp.asarray(result)  # numpy is scalar and can't be replaced
        large_number = 1.0 / (xp.finfo(dtype).eps)
        result[result >= large_number] = xp.inf

        return result

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_stacked_singular(self, xp, dtype):
        # Check behavior when only some of the stacked matrices are
        # singular

        A = xp.arange(16, dtype=dtype).reshape((2, 2, 2, 2))
        A[0, 0] = 0
        A[1, 1] = 0

        res = xp.linalg.cond(A, self.ord)
        return res

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_default(self, xp, dtype):
        A = testing.shaped_arange((2, 2), xp, dtype=dtype)
        return xp.linalg.cond(A)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_basic(self, xp, dtype):
        A = testing.shaped_arange((2, 2), xp, dtype=dtype)
        return xp.linalg.cond(A, self.ord)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_generalized_1(self, xp, dtype):
        A = testing.shaped_arange((2, 2), xp, dtype=dtype)
        A = xp.array([A, 2 * A, 3 * A])
        return xp.linalg.cond(A, self.ord)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_generalized_2(self, xp, dtype):
        A = testing.shaped_arange((2, 2), xp, dtype=dtype)
        A = xp.array([A, 2 * A, 3 * A])
        A = xp.array([A]*2*3).reshape((3, 2)+A.shape)

        return xp.linalg.cond(A, self.ord)

    @testing.for_float_dtypes(no_float16=True)
    def test_0x0(self, dtype):
        for xp in (numpy, cupy):
            A = xp.empty((0, 0), dtype=dtype)
            with pytest.raises(numpy.linalg.LinAlgError,
                               match="cond is not defined on empty arrays"):
                xp.linalg.cond(A, self.ord)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_1x1(self, xp, dtype):
        A = xp.ones((1, 1), dtype=dtype)
        return xp.linalg.cond(A, self.ord)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_8x8(self, xp, dtype):
        A = testing.shaped_arange(
            (8, 8), xp, dtype=dtype)+xp.diag(xp.ones(8, dtype=dtype))
        return xp.linalg.cond(A, self.ord)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_nonarray(self, xp):
        A = [[1., 2.], [3., 4.]]
        return xp.linalg.cond(A, self.ord)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_hermitian(self, xp, dtype):
        A = xp.array([[1., 2.], [2., 1.]], dtype=dtype)
        return xp.linalg.cond(A, self.ord)


class TestCondBasicNonSVD(unittest.TestCase):
    def test_basic_nonsvd(self):
        # Smoketest the non-svd norms
        A = cupy.array([[1.0, 0, 1], [0, -2.0, 0], [0, 0, 3.0]])
        testing.assert_array_almost_equal(cupy.linalg.cond(A, cupy.inf), 4)
        testing.assert_array_almost_equal(
            cupy.linalg.cond(A, -cupy.inf), 2 / 3
        )
        testing.assert_array_almost_equal(cupy.linalg.cond(A, 1), 4)
        testing.assert_array_almost_equal(cupy.linalg.cond(A, -1), 0.5)
        testing.assert_array_almost_equal(
            cupy.linalg.cond(A, "fro"), cupy.sqrt(265 / 12)
        )


class TestLinalgMatrixNorm:

    def test_matrix_norm_exists(self):
        assert hasattr(cupy.linalg, 'matrix_norm')

    @pytest.mark.parametrize('shape', [(3, 3), (4, 5), (5, 4)])
    @pytest.mark.parametrize('ord', [
        None, 'fro', 'nuc', 1, -1, 2, -2, numpy.inf, -numpy.inf,
    ])
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_matrix_norm_matches_numpy_2d(self, shape, ord, keepdims):
        size = shape[0] * shape[1]
        a_np = (numpy.arange(size, dtype=numpy.float64) + 1).reshape(shape)
        a_cp = cupy.asarray(a_np)
        kwargs = {} if ord is None else {'ord': ord}

        expected = numpy.linalg.matrix_norm(a_np, keepdims=keepdims, **kwargs)
        actual = cupy.linalg.matrix_norm(a_cp, keepdims=keepdims, **kwargs)

        testing.assert_allclose(actual, cupy.asarray(expected),
                                rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('shape', [(2, 3, 3), (3, 2, 4, 5)])
    @pytest.mark.parametrize('ord', ['fro', 1, -1, numpy.inf, -numpy.inf])
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_matrix_norm_matches_numpy_batched(self, shape, ord, keepdims):
        size = 1
        for s in shape:
            size *= s
        a_np = (numpy.arange(size, dtype=numpy.float64) + 1).reshape(shape)
        a_cp = cupy.asarray(a_np)

        expected = numpy.linalg.matrix_norm(a_np, keepdims=keepdims, ord=ord)
        actual = cupy.linalg.matrix_norm(a_cp, keepdims=keepdims, ord=ord)

        assert actual.shape == expected.shape
        testing.assert_allclose(actual, cupy.asarray(expected),
                                rtol=1e-5, atol=1e-6)

    def test_matrix_norm_default_ord_is_fro(self):
        a_np = (numpy.arange(9, dtype=numpy.float64) + 1).reshape(3, 3)
        a_cp = cupy.asarray(a_np)
        default = cupy.linalg.matrix_norm(a_cp)
        explicit = cupy.linalg.matrix_norm(a_cp, ord='fro')
        testing.assert_array_equal(default, explicit)

    def test_matrix_norm_keepdims_shape(self):
        a_cp = (cupy.arange(24, dtype=cupy.float64) + 1).reshape(2, 3, 4)
        kept = cupy.linalg.matrix_norm(a_cp, keepdims=True)
        not_kept = cupy.linalg.matrix_norm(a_cp, keepdims=False)
        assert kept.shape == (2, 1, 1)
        assert not_kept.shape == (2,)

    @pytest.mark.parametrize('shape', [(), (4,), (5,)])
    def test_matrix_norm_raises_for_low_ndim(self, shape):
        # Guard: forces this test to fail at the base commit (missing
        # attribute) rather than being incidentally "satisfied" by
        # pytest.raises catching an AttributeError.
        assert hasattr(cupy.linalg, 'matrix_norm')

        if shape:
            size = 1
            for s in shape:
                size *= s
            a_cp = cupy.arange(size, dtype=cupy.float64).reshape(shape)
        else:
            a_cp = cupy.asarray(1.0, dtype=cupy.float64)

        with pytest.raises(Exception):
            cupy.linalg.matrix_norm(a_cp)
