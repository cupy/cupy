import unittest

import numpy

import cupy
from cupy import testing

if cupy.cuda.cutensor_enabled:
    from cupy import cutensor
    from cupy.cuda import cutensor as ct


@testing.parameterize(
    {'dtype': numpy.float16, 'tol': 3e-3},
    {'dtype': numpy.float32, 'tol': 1e-6},
    {'dtype': numpy.float64, 'tol': 1e-12},
    {'dtype': numpy.complex64, 'tol': 1e-6},
    {'dtype': numpy.complex128, 'tol': 1e-12},
)
@unittest.skipUnless(cupy.cuda.cutensor_enabled, 'cuTensor is unavailable')
class TestCuTensor(unittest.TestCase):

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
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_b = cutensor.create_tensor_descriptor(self.b)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.elementwise_trinary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, self.b, desc_b, self.mode_b,
            self.gamma, self.c, desc_c, self.mode_c
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

        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_b = cutensor.create_tensor_descriptor(self.b)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.elementwise_trinary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, self.b, desc_b, self.mode_b,
            self.gamma, self.c, desc_c, self.mode_c, out=out
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
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.elementwise_binary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.gamma, self.c, desc_c, self.mode_c
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
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.elementwise_binary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.gamma, self.c, desc_c, self.mode_c, out=out
        )

        assert d is out
        testing.assert_allclose(
            self.alpha * self.a_transposed +
            self.gamma * self.c_transposed,
            d,
            rtol=self.tol, atol=self.tol
        )

    def test_contraction(self):
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_b = cutensor.create_tensor_descriptor(self.b)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.contraction(
            self.alpha, self.a, desc_a, self.mode_a,
            self.b, desc_b, self.mode_b,
            self.beta, self.c, desc_c, self.mode_c
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
            self.skipTest('Not supported.')

        c = testing.shaped_random((30,), cupy, self.dtype, seed=2)
        c_orig = c.copy()

        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_c = cutensor.create_tensor_descriptor(c)

        d = cutensor.reduction(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, c, desc_c, ('x',)
        )

        assert c is d
        testing.assert_allclose(
            self.alpha * self.a_transposed.sum(axis=(1, 2)) +
            self.beta * c_orig,
            d,
            rtol=self.tol, atol=self.tol
        )


@unittest.skipUnless(cupy.cuda.cutensor_enabled, 'cuTensor is unavailable')
class TestCuTensorDescriptor(unittest.TestCase):

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
        desc_a = cutensor.create_tensor_descriptor(self.a, ct.OP_SQRT)
        desc_b = cutensor.create_tensor_descriptor(self.b, ct.OP_TANH)
        desc_c = cutensor.create_tensor_descriptor(self.c, ct.OP_COS)

        d = cutensor.elementwise_trinary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, self.b, desc_b, self.mode_b,
            self.gamma, self.c, desc_c, self.mode_c,
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
        desc_a = cutensor.create_tensor_descriptor(self.a, ct.OP_SIGMOID)
        desc_c = cutensor.create_tensor_descriptor(self.c, ct.OP_ABS)

        d = cutensor.elementwise_binary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.gamma, self.c, desc_c, self.mode_c,
            op_AC=ct.OP_MUL
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

        desc_a = cutensor.create_tensor_descriptor(self.a, ct.OP_COS)
        desc_c = cutensor.create_tensor_descriptor(c, ct.OP_TANH)

        d = cutensor.reduction(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, c, desc_c, ('x',),
            reduce_op=ct.OP_MAX
        )

        assert c is d
        testing.assert_allclose(
            self.alpha * cupy.cos(self.a_transposed).max(axis=(1, 2)) +
            self.beta * cupy.tanh(c_orig),
            d,
            rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(cupy.cuda.cutensor_enabled, 'cuTensor is unavailable')
class TestCuTensorReduceNumpyArrayCreation(unittest.TestCase):

    def setUp(self):
        self.a = testing.shaped_random(
            (20, 40, 30), cupy, numpy.float32, seed=0)
        self.b = testing.shaped_random(
            (40, 30, 20), cupy, numpy.float32, seed=1)
        self.c = testing.shaped_random(
            (30, 20, 40), cupy, numpy.float32, seed=2)
        self.mode_a = cutensor.create_mode('y', 'z', 'x')
        self.mode_b = cutensor.create_mode('z', 'x', 'y')
        self.mode_c = cutensor.create_mode('x', 'y', 'z')
        self.alpha = numpy.array(1.1, dtype=numpy.float32)
        self.beta = numpy.array(1.2, dtype=numpy.float32)
        self.gamma = numpy.array(1.3, dtype=numpy.float32)
        self.a_transposed = self.a.transpose(2, 0, 1).copy()
        self.b_transposed = self.b.transpose(1, 2, 0).copy()
        self.c_transposed = self.c.copy()

    def test_elementwise_trinary(self):
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_b = cutensor.create_tensor_descriptor(self.b)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.elementwise_trinary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, self.b, desc_b, self.mode_b,
            self.gamma, self.c, desc_c, self.mode_c
        )

        assert d.dtype == numpy.float32

        testing.assert_allclose(
            self.alpha.item() * self.a_transposed +
            self.beta.item() * self.b_transposed +
            self.gamma.item() * self.c_transposed,
            d,
            rtol=1e-6, atol=1e-6
        )

    def test_elementwise_binary(self):
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.elementwise_binary(
            self.alpha, self.a, desc_a, self.mode_a,
            self.gamma, self.c, desc_c, self.mode_c
        )

        assert d.dtype == numpy.float32

        testing.assert_allclose(
            self.alpha.item() * self.a_transposed +
            self.gamma.item() * self.c_transposed,
            d,
            rtol=1e-6, atol=1e-6
        )

    def test_contraction(self):
        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_b = cutensor.create_tensor_descriptor(self.b)
        desc_c = cutensor.create_tensor_descriptor(self.c)

        d = cutensor.contraction(
            self.alpha, self.a, desc_a, self.mode_a,
            self.b, desc_b, self.mode_b,
            self.beta, self.c, desc_c, self.mode_c
        )

        assert self.c is d
        testing.assert_allclose(
            self.alpha.item() * self.a_transposed * self.b_transposed +
            self.beta.item() * self.c_transposed,
            d,
            rtol=1e-6, atol=1e-6
        )

    def test_reduction(self):
        c = testing.shaped_random((30,), cupy, numpy.float32, seed=2)
        c_orig = c.copy()

        desc_a = cutensor.create_tensor_descriptor(self.a)
        desc_c = cutensor.create_tensor_descriptor(c)
        mode_c = cutensor.create_mode('x')

        d = cutensor.reduction(
            self.alpha, self.a, desc_a, self.mode_a,
            self.beta, c, desc_c, mode_c
        )

        assert c is d
        testing.assert_allclose(
            self.alpha.item() * self.a_transposed.sum(axis=(1, 2)) +
            self.beta.item() * c_orig,
            d,
            rtol=1e-6, atol=1e-6
        )
