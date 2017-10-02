import unittest

import numpy
try:
    import scipy.sparse
except ImportError:
    pass

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'transa': [True, False],
}))
@testing.with_requires('scipy')
class TestCsrmm(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        self.op_a = scipy.sparse.random(2, 3, density=0.5, dtype=self.dtype)
        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a
        self.b = numpy.random.uniform(-1, 1, (3, 4)).astype(self.dtype)
        self.c = numpy.random.uniform(-1, 1, (2, 4)).astype(self.dtype)

    def test_csrmm(self):
        a = cupy.sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm(a, b, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * (self.op_a.dot(self.b))
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm_with_c(self):
        a = cupy.sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.csrmm(
            a, b, c=c, alpha=self.alpha, beta=self.beta, transa=self.transa)
        expect = self.alpha * (self.op_a.dot(self.b)) + self.beta * self.c
        self.assertIs(y, c)
        testing.assert_array_almost_equal(y, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'trans': [(False, False), (True, False), (False, True)],
}))
@testing.with_requires('scipy')
class TestCsrmm2(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        self.transa, self.transb = self.trans
        self.op_a = scipy.sparse.random(2, 3, density=0.5, dtype=self.dtype)
        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a
        self.op_b = numpy.random.uniform(-1, 1, (3, 4)).astype(self.dtype)
        if self.transb:
            self.b = self.op_b.T
        else:
            self.b = self.op_b
        self.c = numpy.random.uniform(-1, 1, (2, 4)).astype(self.dtype)

    def test_csrmm2(self):
        a = cupy.sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, alpha=self.alpha, transa=self.transa, transb=self.transb)
        expect = self.alpha * (self.op_a.dot(self.op_b))
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm2_with_c(self):
        a = cupy.sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, c=c, alpha=self.alpha, beta=self.beta,
            transa=self.transa, transb=self.transb)
        expect = self.alpha * (self.op_a.dot(self.op_b)) + self.beta * self.c
        self.assertIs(y, c)
        testing.assert_array_almost_equal(y, expect)
