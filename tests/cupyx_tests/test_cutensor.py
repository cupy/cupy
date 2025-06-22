from __future__ import annotations

import gc

import numpy
import pytest

import cupy
import cupyx
from cupy._core import _routines_linalg as _linalg
from cupy import testing
from cupy.cuda import device

from cupy.cuda import cutensor as ct

if ct.available:
    from cupyx import cutensor


@testing.parameterize(
    {'dtype': numpy.float16, 'tol': 3e-3},
    {'dtype': numpy.float32, 'tol': 1e-6},
    {'dtype': numpy.float64, 'tol': 1e-12},
    {'dtype': numpy.complex64, 'tol': 1e-6},
    {'dtype': numpy.complex128, 'tol': 1e-12},
)
@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestCuTensor:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.a = testing.shaped_random(
            (20, 40, 30), cupy, self.dtype, seed=0)
        self.b = testing.shaped_random(
            (40, 30, 20), cupy, self.dtype, seed=1)
        self.c = testing.shaped_random(
            (30, 20, 40), cupy, self.dtype, seed=2)
        self.mode_a = ('y', 'z', 'x')
        self.mode_b = ('z', 'x', 'y')
        self.mode_c = ('x', 'y', 'z')
        self.alpha = 1.1
        self.beta = 1.2
        self.gamma = 1.3
        self.a_transposed = self.a.transpose(2, 0, 1).copy()
        self.b_transposed = self.b.transpose(1, 2, 0).copy()
        self.c_transposed = self.c.copy()

    def test_elementwise_trinary(self):
        d = cutensor.elementwise_trinary(
            self.alpha, self.a, self.mode_a,
            self.beta,  self.b, self.mode_b,
            self.gamma, self.c, self.mode_c
        )

        assert d.dtype == self.dtype

        testing.assert_allclose(
            self.alpha * self.a_transposed +
            self.beta * self.b_transposed +
            self.gamma * self.c_transposed,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_elementwise_trinary_out(self):
        out = testing.shaped_random(
            (30, 20, 40), cupy, self.dtype, seed=3)

        d = cutensor.elementwise_trinary(
            self.alpha, self.a, self.mode_a,
            self.beta,  self.b, self.mode_b,
            self.gamma, self.c, self.mode_c, out=out
        )

        assert d is out
        testing.assert_allclose(
            self.alpha * self.a_transposed +
            self.beta * self.b_transposed +
            self.gamma * self.c,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_elementwise_binary(self):
        d = cutensor.elementwise_binary(
            self.alpha, self.a, self.mode_a,
            self.gamma, self.c, self.mode_c
        )

        assert d.dtype == self.dtype

        testing.assert_allclose(
            self.alpha * self.a_transposed +
            self.gamma * self.c_transposed,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_elementwise_binary_out(self):
        out = testing.shaped_random(
            (30, 20, 40), cupy, self.dtype, seed=3)

        d = cutensor.elementwise_binary(
            self.alpha, self.a, self.mode_a,
            self.gamma, self.c, self.mode_c, out=out
        )

        assert d is out
        testing.assert_allclose(
            self.alpha * self.a_transposed +
            self.gamma * self.c_transposed,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_contraction(self):
        compute_capability = int(device.get_compute_capability())
        if compute_capability < 70 and self.dtype == numpy.float16:
            pytest.skip('Not supported.')

        d = cutensor.contraction(
            self.alpha, self.a, self.mode_a,
            self.b, self.mode_b,
            self.beta, self.c, self.mode_c
        )

        assert self.c is d
        testing.assert_allclose(
            self.alpha * self.a_transposed * self.b_transposed +
            self.beta * self.c_transposed,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_reduction(self):
        if self.dtype == numpy.float16:
            pytest.skip('Not supported.')

        c = testing.shaped_random((30,), cupy, self.dtype, seed=2)
        c_orig = c.copy()

        d = cutensor.reduction(
            self.alpha, self.a, self.mode_a,
            self.beta, c, ('x',)
        )

        assert c is d
        testing.assert_allclose(
            self.alpha * self.a_transposed.sum(axis=(1, 2)) +
            self.beta * c_orig,
            d,
            rtol=self.tol, atol=self.tol
        )


@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestMode:

    def test_create_mode_int(self):
        m = cutensor.create_mode(10, 11, 12)
        assert m.ndim == 3
        assert repr(m) == 'mode(10, 11, 12)'

    def test_create_mode_ascii(self):
        m = cutensor.create_mode('x', 'y')
        assert m.ndim == 2
        assert repr(m) == 'mode(120, 121)'

    def test_mode_compare(self):
        m1 = cutensor.create_mode(10, 11, 12)
        m2 = cutensor.create_mode(10, 11, 12)
        assert m1 == m2
        assert m1.data == m2.data  # cached

        m2 = cutensor.create_mode(12, 11, 10)
        assert m1 != m2
        assert m1.data != m2.data


@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestScalar:

    def test_create(self):
        s = cutensor._Scalar(10, cupy.float32)
        assert repr(s) == 'scalar(10.0, dtype=float32)'


@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestCuTensorDescriptor:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.a = testing.shaped_random(
            (20, 40, 30), cupy, numpy.float32, seed=0)
        self.b = testing.shaped_random(
            (40, 30, 20), cupy, numpy.float32, seed=1)
        self.c = testing.shaped_random(
            (30, 20, 40), cupy, numpy.float32, seed=2)
        self.mode_a = ('y', 'z', 'x')
        self.mode_b = ('z', 'x', 'y')
        self.mode_c = ('x', 'y', 'z')
        self.alpha = 1.1
        self.beta = 1.2
        self.gamma = 1.3
        self.a_transposed = self.a.transpose(2, 0, 1).copy()
        self.b_transposed = self.b.transpose(1, 2, 0).copy()
        self.c_transposed = self.c.copy()

    def test_elementwise_trinary(self):
        d = cutensor.elementwise_trinary(
            self.alpha, self.a, self.mode_a,
            self.beta,  self.b, self.mode_b,
            self.gamma, self.c, self.mode_c,
            op_A=ct.OP_SQRT, op_B=ct.OP_TANH, op_C=ct.OP_COS,
            op_AB=ct.OP_ADD, op_ABC=ct.OP_MUL
        )

        testing.assert_allclose(
            (self.alpha * cupy.sqrt(self.a_transposed) +
             self.beta * cupy.tanh(self.b_transposed)) *
            self.gamma * cupy.cos(self.c),
            d,
            rtol=1e-6, atol=1e-6
        )

    def test_elementwise_binary(self):
        d = cutensor.elementwise_binary(
            self.alpha, self.a, self.mode_a,
            self.gamma, self.c, self.mode_c,
            op_A=ct.OP_SIGMOID, op_C=ct.OP_ABS, op_AC=ct.OP_MUL
        )

        testing.assert_allclose(
            self.alpha * (1 / (1 + cupy.exp(-self.a_transposed))) *
            self.gamma * cupy.abs(self.c),
            d,
            rtol=1e-6, atol=1e-6
        )

    def test_reduction(self):
        c = testing.shaped_random((30,), cupy, numpy.float32, seed=2)
        c_orig = c.copy()

        d = cutensor.reduction(
            self.alpha, self.a, self.mode_a,
            self.beta, c, ('x',),
            op_A=ct.OP_COS, op_C=ct.OP_TANH,
            op_reduce=ct.OP_MAX
        )

        assert c is d
        testing.assert_allclose(
            self.alpha * cupy.cos(self.a_transposed).max(axis=(1, 2)) +
            self.beta * cupy.tanh(c_orig),
            d,
            rtol=1e-6, atol=1e-6
        )


@testing.parameterize(*testing.product({
    'dtype_combo': ['eee', 'fff', 'ddd', 'FFF', 'DDD', 'dDD', 'DdD'],
    'compute_type_hint': [None, 'down-convert', 'TF32'],
    'shape': [(40, 20, 20)],  # let last two dim be the same for testing cache
    'alpha': [1.0],
    'beta': [0.0, 1.0],
}))
@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestCuTensorContraction:
    _tol = {'e': 1e-3, 'f': 1e-6, 'd': 1e-12}

    def make_random_array(self, shape, dtype):
        return testing.shaped_random(shape, cupy, dtype=dtype, scale=1)

    def make_matrix(self, shape, dtype):
        r_dtype = dtype
        if dtype == numpy.complex64:
            r_dtype = numpy.float32
        elif dtype == numpy.complex128:
            r_dtype = numpy.float64
        a = self.make_random_array(shape, r_dtype)
        if dtype.char in 'FD':
            a = a + 1j * self.make_random_array(shape, r_dtype)
        return a

    @pytest.fixture(autouse=True)
    def setUp(self):
        compute_capability = int(device.get_compute_capability())
        if compute_capability < 70 and 'e' in self.dtype_combo:
            pytest.skip("Not supported")
        dtype_chars = list(self.dtype_combo)
        self.a_dtype = numpy.dtype(dtype_chars[0])
        self.b_dtype = numpy.dtype(dtype_chars[1])
        self.c_dtype = numpy.dtype(dtype_chars[2])
        self.tol = self._tol[dtype_chars[2].lower()]
        self.compute_type = _linalg.COMPUTE_TYPE_DEFAULT
        if self.compute_type_hint == 'down-convert':
            if self.c_dtype.char in 'fF':
                self.compute_type = _linalg.COMPUTE_TYPE_FP16
                self.tol = self._tol['e']
            elif self.c_dtype.char in 'dD':
                self.compute_type = _linalg.COMPUTE_TYPE_FP32
                self.tol = self._tol['f']
        elif self.compute_type_hint == 'TF32':
            if self.c_dtype.char in 'fF':
                self.compute_type = _linalg.COMPUTE_TYPE_TF32
                self.tol = self._tol['e']
        m, n, k = self.shape
        self.a = self.make_matrix((m, k), self.a_dtype)
        self.b = self.make_matrix((k, n), self.b_dtype)
        self.c = self.make_matrix((m, n), self.c_dtype)
        self.c_ref = self.alpha * cupy.matmul(self.a, self.b)
        self.c_ref += self.beta * self.c
        old_compute_type = cupy._core.get_compute_type(self.c_dtype)
        cupy._core.set_compute_type(self.c_dtype, self.compute_type)
        yield
        cupy._core.set_compute_type(self.c_dtype, old_compute_type)

    def test_contraction(self):
        mode_a = cutensor.create_mode('m', 'k')
        mode_b = cutensor.create_mode('k', 'n')
        mode_c = cutensor.create_mode('m', 'n')
        cutensor.contraction(self.alpha,
                             self.a, mode_a,
                             self.b, mode_b,
                             self.beta,
                             self.c, mode_c)
        cupy.testing.assert_allclose(self.c, self.c_ref,
                                     rtol=self.tol, atol=self.tol)

        # test the contraction descriptor cache (issues #7318, #7812)
        del mode_b
        gc.collect()
        mode_b = cutensor.create_mode('n', 'k')  # flipped
        self.c_ref = self.alpha * cupy.matmul(self.a, self.b.T)
        self.c_ref += self.beta * self.c
        cutensor.contraction(self.alpha,
                             self.a, mode_a,
                             self.b, mode_b,
                             self.beta,
                             self.c, mode_c)
        cupy.testing.assert_allclose(self.c, self.c_ref,
                                     rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype_char': ['e', 'f', 'd', 'F', 'D'],
    'shape': [(30, 40, 30, 35)],
    'alpha': [0.5, 1.0],
    'beta': [0.0, 1.0],
    'order': ['C', 'F']
}))
@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestCuTensorIncontiguous:
    _tol = {'e': 1e-3, 'f': 2e-6, 'd': 1e-12}

    @pytest.fixture(autouse=True)
    def setUp(self):
        compute_capability = int(device.get_compute_capability())
        if compute_capability < 70 and self.dtype_char == 'e':
            pytest.skip("Not supported")
        self.dtype = numpy.dtype(self.dtype_char)
        self.tol = self._tol[self.dtype_char.lower()]

    def test_contraction(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_b = cutensor.create_mode('c', 'd', 'b')
        mode_c = cutensor.create_mode('d', 'a')
        a, b, c, d = self.shape
        self.a = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        self.b = testing.shaped_random(
            (c, d, b), cupy, dtype=self.dtype, order=self.order)
        self.c = testing.shaped_random(
            (d, a), cupy, dtype=self.dtype, order=self.order)
        delta = 7
        c_ref = self.c.copy()
        c_ref = cutensor.contraction(self.alpha,
                                     self.a, mode_a,
                                     self.b, mode_b,
                                     self.beta,
                                     c_ref, mode_c)
        for a0 in range(0, a, delta):
            for d0 in range(0, d, delta):
                cutensor.contraction(self.alpha,
                                     self.a[a0:a0+delta], mode_a,
                                     self.b[:, d0:d0+delta], mode_b,
                                     self.beta,
                                     self.c[d0:d0+delta, a0:a0+delta], mode_c)
                cupy.testing.assert_allclose(self.c[d0:d0+delta, a0:a0+delta],
                                             c_ref[d0:d0+delta, a0:a0+delta],
                                             rtol=self.tol, atol=self.tol)

    def test_reduction(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_c = cutensor.create_mode('b')
        a, b, c, _ = self.shape
        self.a = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        self.c = testing.shaped_random(
            (b,), cupy, dtype=self.dtype, order=self.order)

        c_ref = self.c.copy()
        c_ref = cutensor.reduction(self.alpha,
                                   self.a, mode_a,
                                   self.beta,
                                   c_ref, mode_c)
        delta = 7
        for b0 in range(0, b, delta):
            cutensor.reduction(self.alpha,
                               self.a[:, b0:b0+delta, :], mode_a,
                               self.beta,
                               self.c[b0:b0+delta], mode_c)
            cupy.testing.assert_allclose(self.c[b0:b0+delta],
                                         c_ref[b0:b0+delta],
                                         rtol=self.tol, atol=self.tol)

    def test_elementwise_binary(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_c = cutensor.create_mode('c', 'a', 'b')
        a, b, c, _ = self.shape
        self.a = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        self.c = testing.shaped_random(
            (c, a, b), cupy, dtype=self.dtype, order=self.order)

        c_ref = self.c.copy()
        c_ref = cutensor.elementwise_binary(self.alpha,
                                            self.a, mode_a,
                                            self.beta,
                                            c_ref, mode_c)
        delta = 7
        for b0 in range(0, b, delta):
            cutensor.elementwise_binary(self.alpha,
                                        self.a[:, b0:b0+delta], mode_a,
                                        self.beta,
                                        self.c[:, :, b0:b0+delta], mode_c,
                                        out=self.c[:, :, b0:b0+delta])
            cupy.testing.assert_allclose(self.c[:, :, b0:b0+delta],
                                         c_ref[:, :, b0:b0+delta],
                                         rtol=self.tol, atol=self.tol)

    def test_elementwise_trinary(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_b = cutensor.create_mode('b', 'c', 'a')
        mode_c = cutensor.create_mode('c', 'a', 'b')
        a, b, c, _ = self.shape
        self.a = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        self.b = testing.shaped_random(
            (b, c, a), cupy, dtype=self.dtype, order=self.order)
        self.c = testing.shaped_random(
            (c, a, b), cupy, dtype=self.dtype, order=self.order)

        for gamma in [0.0, 1.0]:
            c_ref = self.c.copy()
            c_ref = cutensor.elementwise_trinary(self.alpha, self.a, mode_a,
                                                 self.beta, self.b, mode_b,
                                                 gamma, c_ref, mode_c,
                                                 out=c_ref)
            delta = 7
            for a0 in range(0, a, delta):
                cutensor.elementwise_trinary(self.alpha,
                                             self.a[a0:a0+delta],
                                             mode_a, self.beta,
                                             self.b[:, :, a0:a0+delta],
                                             mode_b, gamma,
                                             self.c[:, a0:a0+delta], mode_c,
                                             out=self.c[:, a0:a0+delta])
                cupy.testing.assert_allclose(self.c[:, a0:a0+delta],
                                             c_ref[:, a0:a0+delta],
                                             rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype_char': ['e', 'f', 'd', 'F', 'D'],
    'shape': [32],
}))
@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestCuTensorMg:
    _tol = {'e': 1e-3, 'f': 2e-6, 'd': 1e-12}

    @pytest.fixture(autouse=True)
    def setUp(self):
        compute_capability = int(device.get_compute_capability())
        if compute_capability < 70 and self.dtype_char == 'e':
            pytest.skip("Not supported")
        self.dtype = numpy.dtype(self.dtype_char)
        self.tol = self._tol[self.dtype_char.lower()]

    def test_contraction(self):
        n = self.shape
        if self.dtype == 'e':
            # 16-bit result host pageable tensors are not supported in the
            # contraction routines.
            self.a = cupyx.empty_pinned((n, n, n, n), dtype=self.dtype)
        else:
            self.a = testing.shaped_random(
                (n, n, n, n), numpy, dtype=self.dtype)
        self.b = testing.shaped_random(
            (n, n, n, n), cupy, dtype=self.dtype)
        self.c = cupyx.empty_pinned((n, n, n, n), dtype=self.dtype)
        c_ref = numpy.einsum('kijl,kadl->iajd', self.a, self.b.get())
        mga = cutensor.ndarray_mg(self.a, block_size=[8, 8, 8, 8])
        cutensor.contractionMg(1, mga, 'kijl', self.b,
                               'kadl', 0, self.c, 'iajd')
        cupy.cuda.Device(0).synchronize()
        cupy.testing.assert_allclose(self.c, c_ref, rtol=self.tol,
                                     atol=self.tol)

    def test_copy(self):
        n = self.shape
        if self.dtype == 'e':
            # 16-bit result host pageable tensors are not supported in the
            # contraction routines.
            self.a = cupyx.empty_pinned((n, n, n, n), dtype=self.dtype)
        else:
            self.a = testing.shaped_random(
                (n, n, n, n), numpy, dtype=self.dtype)
        self.b = testing.shaped_random(
            (n, n, n, n), cupy, dtype=self.dtype)
        cutensor.copyMg(self.b, 'cabd', self.a, 'abcd')
        cupy.cuda.Device(0).synchronize()
        cupy.testing.assert_allclose(self.b.get(), self.a.transpose(
            (2, 0, 1, 3)), rtol=self.tol, atol=self.tol)
