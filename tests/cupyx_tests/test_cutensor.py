from __future__ import annotations

import gc

import numpy
import pytest

import cupy
import cupyx
from cupy._core import _routines_linalg as _linalg
from cupy import testing
from cupy.cuda import device
from cupy_backends.cuda.api import runtime

from cupy.cuda import cutensor as ct

if ct.available:
    from cupyx import cutensor
    from cupyx import _cutensor as _cutensor_module


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
        # hipTensor currently does not support complex kernels.
        if (runtime.is_hip
                and self.dtype in (numpy.complex64, numpy.complex128)):
            pytest.skip('hipTensor does not support complex dtypes')
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
        if not runtime.is_hip:
            compute_capability = int(device.get_compute_capability())
            if compute_capability < 70 and self.dtype == numpy.float16:
                pytest.skip('Not supported.')

        c = self.c.copy("K")
        d = cutensor.contraction(
            self.alpha, self.a, self.mode_a,
            self.b, self.mode_b,
            self.beta, c, self.mode_c
        )

        assert c is d
        testing.assert_allclose(
            self.alpha * self.a_transposed * self.b_transposed +
            self.beta * self.c_transposed,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_reduction(self):
        if not runtime.is_hip and self.dtype == numpy.float16:
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


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
@pytest.mark.parametrize('mode_objects', [False, True])
def test_hip_contraction_without_reduction(mode_objects):
    a = testing.shaped_random((2, 3), cupy, numpy.float32, seed=0)
    b = testing.shaped_random((3, 2), cupy, numpy.float32, seed=1)
    c = testing.shaped_random((2, 3), cupy, numpy.float32, seed=2)
    c_orig = c.copy()
    mode_a = ('m', 'n')
    mode_b = ('n', 'm')
    mode_c = ('m', 'n')
    if mode_objects:
        mode_a = cutensor.create_mode(*mode_a)
        mode_b = cutensor.create_mode(*mode_b)
        mode_c = cutensor.create_mode(*mode_c)

    out = cutensor.contraction(
        1.25, a, mode_a, b, mode_b, 0.5, c, mode_c,
        op_A=ct.OP_NEG, op_C=ct.OP_ABS)

    assert out is c
    expected = 1.25 * (-a) * b.T + 0.5 * cupy.abs(c_orig)
    testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
def test_hip_float16_contraction_beta_and_op_c():
    a = testing.shaped_random((4, 3), cupy, numpy.float16, seed=0)
    b = testing.shaped_random((3, 5), cupy, numpy.float16, seed=1)
    c = testing.shaped_random((4, 5), cupy, numpy.float16, seed=2)
    c_orig = c.copy()

    out = cutensor.contraction(
        1.0, a, ('m', 'k'), b, ('k', 'n'),
        0.5, c, ('m', 'n'), op_C=ct.OP_NEG,
        compute_desc=ct.COMPUTE_DESC_32F)

    assert out is c
    expected = cupy.matmul(a, b) - 0.5 * c_orig
    testing.assert_allclose(out, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
@pytest.mark.parametrize('mode_objects', [False, True])
def test_hip_elementwise_mode_broadcast(mode_objects):
    a = testing.shaped_random((3,), cupy, numpy.float32, seed=0)
    b = testing.shaped_random((2,), cupy, numpy.float32, seed=1)
    c = testing.shaped_random((2, 3), cupy, numpy.float32, seed=2)
    mode_a = ('n',)
    mode_b = ('m',)
    mode_c = ('m', 'n')
    if mode_objects:
        mode_a = cutensor.create_mode(*mode_a)
        mode_b = cutensor.create_mode(*mode_b)
        mode_c = cutensor.create_mode(*mode_c)

    binary_out = cutensor.elementwise_binary(
        1.25, a, mode_a, 0.5, c, mode_c)
    testing.assert_allclose(
        binary_out, 1.25 * a[None, :] + 0.5 * c,
        rtol=1e-6, atol=1e-6)

    trinary_out = cutensor.elementwise_trinary(
        1.25, a, mode_a, 0.75, b, mode_b,
        0.5, c, mode_c)
    testing.assert_allclose(
        trinary_out,
        1.25 * a[None, :] + 0.75 * b[:, None] + 0.5 * c,
        rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
def test_hip_fallback_compute_descriptor():
    a = testing.shaped_random((100, 100), cupy, numpy.float16, seed=0)
    b = testing.shaped_random((100, 100), cupy, numpy.float16, seed=1)
    c = testing.shaped_random((100, 100), cupy, numpy.float16, seed=2)
    c_orig = c.copy()

    binary_out = cutensor.elementwise_binary(
        1.1, a, ('m', 'n'), 0.7, c, ('m', 'n'),
        compute_desc=ct.COMPUTE_DESC_32F)
    expected_binary = (
        numpy.float32(1.1) * a.astype(numpy.float32)
        + numpy.float32(0.7) * c_orig.astype(numpy.float32)
    ).astype(numpy.float16)
    testing.assert_array_equal(binary_out, expected_binary)

    trinary_out = cutensor.elementwise_trinary(
        1.1, a, ('m', 'n'), 0.9, b, ('m', 'n'),
        0.7, c, ('m', 'n'), compute_desc=ct.COMPUTE_DESC_32F)
    expected_trinary = (
        numpy.float32(1.1) * a.astype(numpy.float32)
        + numpy.float32(0.9) * b.astype(numpy.float32)
        + numpy.float32(0.7) * c_orig.astype(numpy.float32)
    ).astype(numpy.float16)
    testing.assert_array_equal(trinary_out, expected_trinary)

    contraction_out = cutensor.contraction(
        1.1, a, ('m', 'n'), b, ('m', 'n'),
        0.7, c, ('m', 'n'), compute_desc=ct.COMPUTE_DESC_32F)
    expected_contraction = (
        numpy.float32(1.1) * a.astype(numpy.float32)
        * b.astype(numpy.float32)
        + numpy.float32(0.7) * c_orig.astype(numpy.float32)
    ).astype(numpy.float16)
    testing.assert_array_equal(contraction_out, expected_contraction)


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
def test_hip_trinary_rejects_mixed_input_dtypes():
    a = cupy.ones((2, 3), dtype=numpy.float32)
    b = cupy.ones((2, 3), dtype=numpy.float64)
    c = cupy.ones((2, 3), dtype=numpy.float32)

    with pytest.raises(ValueError, match='dtype mismatch'):
        cutensor.elementwise_trinary(
            1, a, ('m', 'n'), 1, b, ('m', 'n'),
            1, c, ('m', 'n'))


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
def test_hip_zero_scalar_skips_unary_operator():
    a = -cupy.ones((2, 3), dtype=numpy.float32)
    b = -cupy.ones((2, 3), dtype=numpy.float32)
    c = testing.shaped_random((2, 3), cupy, numpy.float32, seed=0)
    expected = c.copy()

    binary_out = cutensor.elementwise_binary(
        0, a, ('m', 'n'), 1, c, ('m', 'n'),
        op_A=ct.OP_SQRT)
    testing.assert_array_equal(binary_out, expected)

    trinary_out = cutensor.elementwise_trinary(
        0, a, ('m', 'n'), 0, b, ('m', 'n'),
        1, c, ('m', 'n'), op_A=ct.OP_SQRT, op_B=ct.OP_LOG)
    testing.assert_array_equal(trinary_out, expected)

    contraction_out = cutensor.contraction(
        0, a, ('m', 'n'), b, ('m', 'n'),
        1, c, ('m', 'n'), op_A=ct.OP_SQRT, op_B=ct.OP_LOG)
    testing.assert_array_equal(contraction_out, expected)

    positive = cupy.ones((2, 3), dtype=numpy.float32)
    negative = -positive
    binary_out = cutensor.elementwise_binary(
        1, positive, ('m', 'n'), 0, negative, ('m', 'n'),
        op_C=ct.OP_SQRT)
    testing.assert_array_equal(binary_out, positive)

    contraction_out = cutensor.contraction(
        1, positive, ('m', 'n'), positive, ('m', 'n'),
        0, negative, ('m', 'n'), op_C=ct.OP_SQRT)
    testing.assert_array_equal(contraction_out, positive)


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
def test_hip_batched_contraction_fallback():
    a = cupy.arange(24, dtype=numpy.float32).reshape(3, 4, 2)
    b = cupy.arange(24, dtype=numpy.float32).reshape(4, 3, 2)
    c = cupy.zeros((3, 2), dtype=numpy.float32)

    out = cutensor.contraction(
        1, a, ('i', 'j', 'k'), b, ('j', 'i', 'k'),
        0, c, ('i', 'k'))

    expected = cupy.einsum('ijk,jik->ik', a, b)
    testing.assert_array_equal(out, expected)


@pytest.mark.skipif(
    not ct.available or not runtime.is_hip,
    reason='hipTensor is unavailable')
def test_hip_fallback_input_validation():
    a = cupy.ones((2, 3), dtype=numpy.float32)
    c = cupy.ones((2, 3), dtype=numpy.float32)

    with pytest.raises(ValueError, match='mode length'):
        cutensor.elementwise_binary(
            1, a, ('m', 'n'), 1, c, ('m',))
    with pytest.raises(ValueError, match='mode length'):
        cutensor.elementwise_trinary(
            1, a, ('m', 'n'), 1, a, ('m', 'n'),
            1, c, ('m',))
    with pytest.raises(ValueError, match='unsupported unary operator'):
        cutensor.contraction(
            0, a, ('m', 'n'), a, ('m', 'n'),
            1, c, ('m', 'n'), op_A=999)

    a_i = cupy.ones((1,), dtype=numpy.float32)
    b_i = cupy.ones((1,), dtype=numpy.float32)
    c_i = cupy.ones((3,), dtype=numpy.float32)
    with pytest.raises(ValueError, match='extent mismatch'):
        cutensor.elementwise_binary(
            0, a_i, ('i',), 1, c_i, ('i',))
    with pytest.raises(ValueError, match='extent mismatch'):
        cutensor.elementwise_trinary(
            1, a_i, ('i',), 0, b_i, ('i',),
            1, c_i, ('i',))

    a_ij = cupy.ones((1, 2), dtype=numpy.float32)
    b_j = cupy.ones((2,), dtype=numpy.float32)
    with pytest.raises(ValueError, match='extent mismatch'):
        cutensor.contraction(
            0, a_ij, ('i', 'j'), b_j, ('j',),
            1, c_i, ('i',))


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

    def test_hip_unavailable_features(self):
        if not runtime.is_hip:
            pytest.skip('For ROCm/HIP environment')
        assert not cutensor.check_availability('elementwise')
        assert not cutensor.check_availability('contraction')
        assert not cutensor.check_availability('copyMg')
        assert not cutensor.check_availability('contractMg')

    def test_unsupported_status_classification(self):
        assert _cutensor_module._is_unsupported_status(
            ct.STATUS_NOT_SUPPORTED)
        assert _cutensor_module._is_unsupported_status(
            ct.STATUS_ARCH_MISMATCH)
        assert not _cutensor_module._is_unsupported_status(
            ct.STATUS_INVALID_VALUE)


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

    def test_plan_cache(self):
        desc = cutensor.create_tensor_descriptor(self.c)
        mode = cutensor.create_mode(*self.mode_c)
        operator = cutensor.create_elementwise_binary(
            desc, mode, ct.OP_IDENTITY,
            desc, mode, ct.OP_IDENTITY,
            desc, mode, ct.OP_ADD)
        preference = cutensor.create_plan_preference()
        plan = cutensor.create_plan(operator, preference)

        assert cutensor.create_plan(operator, preference) is plan

    def test_repeated_plan_execution(self):
        a = self.a[:, :, 0]
        b = self.b[:, :, 0]
        expected_contraction = cupy.matmul(a, b)
        contraction_out = cupy.empty_like(expected_contraction)
        for _ in range(2):
            cutensor.contraction(
                1, a, ('m', 'k'), b, ('k', 'n'),
                0, contraction_out, ('m', 'n'))
            testing.assert_allclose(
                contraction_out, expected_contraction,
                rtol=1e-6, atol=1e-6)

        reduction_out = cupy.empty((30,), dtype=self.a.dtype)
        expected_reduction = self.a.sum(axis=(0, 1))
        for _ in range(2):
            cutensor.reduction(
                1, self.a, self.mode_a,
                0, reduction_out, ('x',))
            testing.assert_allclose(
                reduction_out, expected_reduction,
                rtol=1e-6, atol=1e-6)


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
        if runtime.is_hip:
            # hipTensor currently does not support complex kernels.
            if any(c in 'FD' for c in self.dtype_combo):
                pytest.skip('hipTensor does not support complex dtypes')
            # hipTensor does not expose TF32 compute; CuPy maps TF32 to a
            # CUDA-specific compute path.
            if self.compute_type_hint == 'TF32':
                pytest.skip('hipTensor does not support TF32 compute')
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
        c = self.c.copy()
        cutensor.contraction(self.alpha,
                             self.a, mode_a,
                             self.b, mode_b,
                             self.beta,
                             c, mode_c)
        cupy.testing.assert_allclose(c, self.c_ref,
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
        if runtime.is_hip and self.dtype_char in ('F', 'D'):
            pytest.skip('hipTensor does not support complex dtypes')
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
        x = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        y = testing.shaped_random(
            (c, d, b), cupy, dtype=self.dtype, order=self.order)
        z = testing.shaped_random(
            (d, a), cupy, dtype=self.dtype, order=self.order)
        delta = 7
        c_ref = z.copy()
        c_ref = cutensor.contraction(self.alpha,
                                     x, mode_a,
                                     y, mode_b,
                                     self.beta,
                                     c_ref, mode_c)
        for a0 in range(0, a, delta):
            for d0 in range(0, d, delta):
                cutensor.contraction(self.alpha,
                                     x[a0:a0+delta], mode_a,
                                     y[:, d0:d0+delta], mode_b,
                                     self.beta,
                                     z[d0:d0+delta, a0:a0+delta], mode_c)
                cupy.testing.assert_allclose(z[d0:d0+delta, a0:a0+delta],
                                             c_ref[d0:d0+delta, a0:a0+delta],
                                             rtol=self.tol, atol=self.tol)

    def test_reduction(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_c = cutensor.create_mode('b')
        a, b, c, _ = self.shape
        x = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        z = testing.shaped_random(
            (b,), cupy, dtype=self.dtype, order=self.order)

        c_ref = z.copy()
        c_ref = cutensor.reduction(self.alpha,
                                   x, mode_a,
                                   self.beta,
                                   c_ref, mode_c)
        delta = 7
        for b0 in range(0, b, delta):
            cutensor.reduction(self.alpha,
                               x[:, b0:b0+delta, :], mode_a,
                               self.beta,
                               z[b0:b0+delta], mode_c)
            cupy.testing.assert_allclose(z[b0:b0+delta],
                                         c_ref[b0:b0+delta],
                                         rtol=self.tol, atol=self.tol)

    def test_elementwise_binary(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_c = cutensor.create_mode('c', 'a', 'b')
        a, b, c, _ = self.shape
        x = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        z = testing.shaped_random(
            (c, a, b), cupy, dtype=self.dtype, order=self.order)

        c_ref = z.copy()
        c_ref = cutensor.elementwise_binary(self.alpha,
                                            x, mode_a,
                                            self.beta,
                                            c_ref, mode_c)
        delta = 7
        for b0 in range(0, b, delta):
            cutensor.elementwise_binary(self.alpha,
                                        x[:, b0:b0+delta], mode_a,
                                        self.beta,
                                        z[:, :, b0:b0+delta], mode_c,
                                        out=z[:, :, b0:b0+delta])
            cupy.testing.assert_allclose(z[:, :, b0:b0+delta],
                                         c_ref[:, :, b0:b0+delta],
                                         rtol=self.tol, atol=self.tol)

    def test_elementwise_trinary(self):
        mode_a = cutensor.create_mode('a', 'b', 'c')
        mode_b = cutensor.create_mode('b', 'c', 'a')
        mode_c = cutensor.create_mode('c', 'a', 'b')
        a, b, c, _ = self.shape
        x = testing.shaped_random(
            (a, b, c), cupy, dtype=self.dtype, order=self.order)
        y = testing.shaped_random(
            (b, c, a), cupy, dtype=self.dtype, order=self.order)
        z = testing.shaped_random(
            (c, a, b), cupy, dtype=self.dtype, order=self.order)

        for gamma in [0.0, 1.0]:
            c_ref = z.copy()
            c_ref = cutensor.elementwise_trinary(self.alpha, x, mode_a,
                                                 self.beta, y, mode_b,
                                                 gamma, c_ref, mode_c,
                                                 out=c_ref)
            delta = 7
            for a0 in range(0, a, delta):
                cutensor.elementwise_trinary(self.alpha,
                                             x[a0:a0+delta],
                                             mode_a, self.beta,
                                             y[:, :, a0:a0+delta],
                                             mode_b, gamma,
                                             z[:, a0:a0+delta], mode_c,
                                             out=z[:, a0:a0+delta])
                cupy.testing.assert_allclose(z[:, a0:a0+delta],
                                             c_ref[:, a0:a0+delta],
                                             rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype_char': ['e', 'f', 'd', 'F', 'D'],
    'shape': [32],
}))
@pytest.mark.skipif(not ct.available, reason='cuTensor is unavailable')
class TestCuTensorMg:
    _tol = {'e': 4e-3, 'f': 2e-6, 'd': 1e-12}

    @pytest.fixture(autouse=True)
    def setUp(self):
        if runtime.is_hip:
            pytest.skip('cuTENSORMg is not available in hipTensor')
        compute_capability = int(device.get_compute_capability())
        if compute_capability < 70 and self.dtype_char == 'e':
            pytest.skip("Not supported")
        self.dtype = numpy.dtype(self.dtype_char)
        self.tol = self._tol[self.dtype_char.lower()]

    def test_contraction(self):
        n = self.shape
        if self.dtype == 'e':
            a = cupyx.empty_pinned((n, n, n, n), dtype=self.dtype)
            a[...] = testing.shaped_random(
                (n, n, n, n), numpy, dtype="float32")
        else:
            a = testing.shaped_random(
                (n, n, n, n), numpy, dtype=self.dtype)
        b = testing.shaped_random(
            (n, n, n, n), cupy, dtype=self.dtype)
        c = cupyx.empty_pinned((n, n, n, n), dtype=self.dtype)
        c_ref = numpy.einsum('kijl,kadl->iajd', a, b.get())
        mga = cutensor.ndarray_mg(a, block_size=[8, 8, 8, 8])
        cutensor.contractionMg(1, mga, 'kijl', b,
                               'kadl', 0, c, 'iajd')
        cupy.cuda.Device(0).synchronize()
        cupy.testing.assert_allclose(c, c_ref, rtol=self.tol,
                                     atol=self.tol)

    def test_copy(self):
        n = self.shape
        if self.dtype == 'e':
            # 16-bit result host pageable tensors are not supported in the
            # contraction routines.
            a = cupyx.empty_pinned((n, n, n, n), dtype=self.dtype)
        else:
            a = testing.shaped_random(
                (n, n, n, n), numpy, dtype=self.dtype)
        b = testing.shaped_random(
            (n, n, n, n), cupy, dtype=self.dtype)
        cutensor.copyMg(b, 'cabd', a, 'abcd')
        cupy.cuda.Device(0).synchronize()
        cupy.testing.assert_allclose(b.get(), a.transpose(
            (2, 0, 1, 3)), rtol=self.tol, atol=self.tol)
