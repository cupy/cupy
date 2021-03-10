import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing import _condition
import cupyx
from cupy.cublas import get_batched_gesv_limit, set_batched_gesv_limit


@testing.parameterize(*testing.product({
    'batched_gesv_limit': [None, 0],
    'order': ['C', 'F'],
}))
@testing.gpu
@testing.fix_random()
class TestSolve(unittest.TestCase):

    def setUp(self):
        if self.batched_gesv_limit is not None:
            self.old_limit = get_batched_gesv_limit()
            set_batched_gesv_limit(self.batched_gesv_limit)

    def tearDown(self):
        if self.batched_gesv_limit is not None:
            set_batched_gesv_limit(self.old_limit)

    @testing.for_dtypes('ifdFD')
    # TODO(kataoka): Fix contiguity
    @testing.numpy_cupy_allclose(atol=1e-3, contiguous_check=False)
    def check_x(self, a_shape, b_shape, xp, dtype):
        a = testing.shaped_random(a_shape, xp, dtype=dtype, seed=0, scale=20)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, seed=1)
        a = a.copy(order=self.order)
        b = b.copy(order=self.order)
        a_copy = a.copy()
        b_copy = b.copy()
        result = xp.linalg.solve(a, b)
        cupy.testing.assert_array_equal(a_copy, a)
        cupy.testing.assert_array_equal(b_copy, b)
        return result

    def test_solve(self):
        self.check_x((4, 4), (4,))
        self.check_x((5, 5), (5, 2))
        self.check_x((2, 4, 4), (2, 4,))
        self.check_x((2, 5, 5), (2, 5, 2))
        self.check_x((2, 3, 2, 2), (2, 3, 2,))
        self.check_x((2, 3, 3, 3), (2, 3, 3, 2))

    def check_shape(self, a_shape, b_shape, error_type):
        for xp in (numpy, cupy):
            a = xp.random.rand(*a_shape)
            b = xp.random.rand(*b_shape)
            with pytest.raises(error_type):
                xp.linalg.solve(a, b)

    def test_invalid_shape(self):
        self.check_shape((2, 3), (4,), numpy.linalg.LinAlgError)
        self.check_shape((3, 3), (2,), ValueError)
        self.check_shape((3, 3), (2, 2), ValueError)
        self.check_shape((3, 3, 4), (3,), numpy.linalg.LinAlgError)
        self.check_shape((2, 3, 3), (3,), ValueError)


@testing.parameterize(*testing.product({
    'a_shape': [(2, 3, 6), (3, 4, 4, 3)],
    'axes': [None, (0, 2)],
}))
@testing.fix_random()
@testing.gpu
class TestTensorSolve(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose(atol=0.02)
    def test_tensorsolve(self, xp, dtype):
        a_shape = self.a_shape
        b_shape = self.a_shape[:2]
        a = testing.shaped_random(a_shape, xp, dtype=dtype, seed=0)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, seed=1)
        return xp.linalg.tensorsolve(a, b, axes=self.axes)


@testing.parameterize(*testing.product({
    'order': ['C', 'F'],
}))
@testing.gpu
class TestInv(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    @_condition.retry(10)
    def check_x(self, a_shape, dtype):
        a_cpu = numpy.random.randint(0, 10, size=a_shape)
        a_cpu = a_cpu.astype(dtype, order=self.order)
        a_gpu = cupy.asarray(a_cpu, order=self.order)
        a_gpu_copy = a_gpu.copy()
        result_cpu = numpy.linalg.inv(a_cpu)
        result_gpu = cupy.linalg.inv(a_gpu)
        assert result_cpu.dtype == result_gpu.dtype
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)
        cupy.testing.assert_array_equal(a_gpu_copy, a_gpu)

    def check_shape(self, a_shape):
        a = cupy.random.rand(*a_shape)
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.inv(a)

    def test_inv(self):
        self.check_x((3, 3))
        self.check_x((4, 4))
        self.check_x((5, 5))
        self.check_x((2, 5, 5))
        self.check_x((3, 4, 4))
        self.check_x((4, 2, 3, 3))

    def test_invalid_shape(self):
        self.check_shape((2, 3))
        self.check_shape((4, 1))
        self.check_shape((4, 3, 2))
        self.check_shape((2, 4, 3))


@testing.gpu
class TestInvInvalid(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    def test_inv(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array([[1, 2], [2, 4]]).astype(dtype)
            with cupyx.errstate(linalg='raise'):
                with pytest.raises(numpy.linalg.LinAlgError):
                    xp.linalg.inv(a)

    @testing.for_dtypes('ifdFD')
    def test_batched_inv(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array([[[1, 2], [2, 4]]]).astype(dtype)
            assert a.ndim >= 3  # CuPy internally uses a batched function.
            with cupyx.errstate(linalg='raise'):
                with pytest.raises(numpy.linalg.LinAlgError):
                    xp.linalg.inv(a)


@testing.gpu
class TestPinv(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    @_condition.retry(10)
    def check_x(self, a_shape, rcond, dtype):
        a_gpu = testing.shaped_random(a_shape, dtype=dtype)
        a_cpu = cupy.asnumpy(a_gpu)
        a_gpu_copy = a_gpu.copy()
        if not isinstance(rcond, float):
            rcond = numpy.asarray(rcond)
        result_cpu = numpy.linalg.pinv(a_cpu, rcond=rcond)
        if not isinstance(rcond, float):
            rcond = cupy.asarray(rcond)
        result_gpu = cupy.linalg.pinv(a_gpu, rcond=rcond)

        assert result_cpu.dtype == result_gpu.dtype
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)
        cupy.testing.assert_array_equal(a_gpu_copy, a_gpu)

    def test_pinv(self):
        self.check_x((3, 3), rcond=1e-15)
        self.check_x((2, 4), rcond=1e-15)
        self.check_x((3, 2), rcond=1e-15)

        self.check_x((4, 4), rcond=0.3)
        self.check_x((2, 5), rcond=0.5)
        self.check_x((5, 3), rcond=0.6)

    def test_pinv_batched(self):
        self.check_x((2, 3, 4), rcond=1e-15)
        self.check_x((2, 3, 4, 5), rcond=1e-15)

    def test_pinv_batched_vector_rcond(self):
        self.check_x((2, 3, 4), rcond=[0.2, 0.8])
        self.check_x((2, 3, 4, 5),
                     rcond=[[0.2, 0.9, 0.1],
                            [0.7, 0.2, 0.5]])

    def test_pinv_size_0(self):
        self.check_x((3, 0), rcond=1e-15)
        self.check_x((0, 3), rcond=1e-15)
        self.check_x((0, 0), rcond=1e-15)
        self.check_x((0, 2, 3), rcond=1e-15)
        self.check_x((2, 0, 3), rcond=1e-15)


@testing.gpu
class TestLstsq(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose(atol=1e-3)
    def check_lstsq_solution(self, a_shape, b_shape, seed, rcond, xp, dtype,
                             singular=False):
        if singular:
            m, n = a_shape
            rank = min(m, n) - 1
            a = xp.matmul(
                testing.shaped_random(
                    (m, rank), xp, dtype=dtype, scale=3, seed=seed),
                testing.shaped_random(
                    (rank, n), xp, dtype=dtype, scale=3, seed=seed+42),
            )
        else:
            a = testing.shaped_random(a_shape, xp, dtype=dtype, seed=seed)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, seed=seed+37)
        a_copy = a.copy()
        b_copy = b.copy()
        results = xp.linalg.lstsq(a, b, rcond)
        if xp is cupy:
            testing.assert_array_equal(a_copy, a)
            testing.assert_array_equal(b_copy, b)
        return results

    def check_invalid_shapes(self, a_shape, b_shape):
        a = cupy.random.rand(*a_shape)
        b = cupy.random.rand(*b_shape)
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.lstsq(a, b, rcond=None)

    def test_lstsq_solutions(self):
        # Comapres numpy.linalg.lstsq and cupy.linalg.lstsq solutions for:
        #   a shapes range from (3, 3) to (5, 3) and (3, 5)
        #   b shapes range from (i, 3) to (i, )
        #   sets a random seed for deterministic testing
        for i in range(3, 6):
            for j in range(3, 6):
                for k in range(2, 4):
                    seed = i + j + k
                    # check when b has shape (i, k)
                    self.check_lstsq_solution((i, j), (i, k), seed,
                                              rcond=-1)
                    self.check_lstsq_solution((i, j), (i, k), seed,
                                              rcond=None)
                    self.check_lstsq_solution((i, j), (i, k), seed,
                                              rcond=0.5)
                    self.check_lstsq_solution((i, j), (i, k), seed,
                                              rcond=1e-6, singular=True)
                # check when b has shape (i, )
                self.check_lstsq_solution((i, j), (i, ), seed+1, rcond=-1)
                self.check_lstsq_solution((i, j), (i, ), seed+1, rcond=None)
                self.check_lstsq_solution((i, j), (i, ), seed+1, rcond=0.5)
                self.check_lstsq_solution((i, j), (i, ), seed+1, rcond=1e-6,
                                          singular=True)

    def test_invalid_shapes(self):
        self.check_invalid_shapes((4, 3), (3, ))
        self.check_invalid_shapes((3, 3, 3), (2, 2))
        self.check_invalid_shapes((3, 3, 3), (3, 3))
        self.check_invalid_shapes((3, 3), (3, 3, 3))
        self.check_invalid_shapes((2, 2), (10, ))
        self.check_invalid_shapes((3, 3), (2, 2))
        self.check_invalid_shapes((4, 3), (10, 3, 3))

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-3)
    def test_warn_rcond(self, xp, dtype):
        a = testing.shaped_random((3, 3), xp, dtype)
        b = testing.shaped_random((3,), xp, dtype)
        with testing.assert_warns(FutureWarning):
            return xp.linalg.lstsq(a, b)


@testing.gpu
class TestTensorInv(unittest.TestCase):

    @testing.for_dtypes('ifdFD')
    @_condition.retry(10)
    def check_x(self, a_shape, ind, dtype):
        a_cpu = numpy.random.randint(0, 10, size=a_shape).astype(dtype)
        a_gpu = cupy.asarray(a_cpu)
        a_gpu_copy = a_gpu.copy()
        result_cpu = numpy.linalg.tensorinv(a_cpu, ind=ind)
        result_gpu = cupy.linalg.tensorinv(a_gpu, ind=ind)
        assert result_cpu.dtype == result_gpu.dtype
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)
        cupy.testing.assert_array_equal(a_gpu_copy, a_gpu)

    def check_shape(self, a_shape, ind):
        a = cupy.random.rand(*a_shape)
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.tensorinv(a, ind=ind)

    def check_ind(self, a_shape, ind):
        a = cupy.random.rand(*a_shape)
        with self.assertRaises(ValueError):
            cupy.linalg.tensorinv(a, ind=ind)

    def test_tensorinv(self):
        self.check_x((12, 3, 4), ind=1)
        self.check_x((3, 8, 24), ind=2)
        self.check_x((18, 3, 3, 2), ind=1)
        self.check_x((1, 4, 2, 2), ind=2)
        self.check_x((2, 3, 5, 30), ind=3)
        self.check_x((24, 2, 2, 3, 2), ind=1)
        self.check_x((3, 4, 2, 3, 2), ind=2)
        self.check_x((1, 2, 3, 2, 3), ind=3)
        self.check_x((3, 2, 1, 2, 12), ind=4)

    def test_invalid_shape(self):
        self.check_shape((2, 3, 4), ind=1)
        self.check_shape((1, 2, 3, 4), ind=3)

    def test_invalid_index(self):
        self.check_ind((12, 3, 4), ind=-1)
        self.check_ind((18, 3, 3, 2), ind=0)
