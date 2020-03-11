import pickle
import unittest

import numpy
try:
    import scipy.sparse
except ImportError:
    pass

import cupy
from cupy import testing
from cupy import cusparse
from cupyx.scipy import sparse


class TestMatDescriptor(unittest.TestCase):

    def test_create(self):
        md = cusparse.MatDescriptor.create()
        assert isinstance(md.descriptor, int)

    def test_pickle(self):
        md = cusparse.MatDescriptor.create()
        md2 = pickle.loads(pickle.dumps(md))
        assert isinstance(md2.descriptor, int)
        assert md.descriptor != md2.descriptor


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
        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm(a, b, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.b)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm_with_c(self):
        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.csrmm(
            a, b, c=c, alpha=self.alpha, beta=self.beta, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.b) + self.beta * self.c
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
        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, alpha=self.alpha, transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm2_with_c(self):
        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, c=c, alpha=self.alpha, beta=self.beta,
            transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b) + self.beta * self.c
        self.assertIs(y, c)
        testing.assert_array_almost_equal(y, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'transa': [False, True],
    'transb': [False, True],
}))
@testing.with_requires('scipy')
class TestCsrgemm(unittest.TestCase):

    def setUp(self):
        self.op_a = scipy.sparse.random(2, 3, density=0.5, dtype=self.dtype)
        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a
        self.op_b = scipy.sparse.random(3, 4, density=0.5, dtype=self.dtype)
        if self.transb:
            self.b = self.op_b.T
        else:
            self.b = self.op_b

    def test_csrgemm(self):
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        y = cupy.cusparse.csrgemm(a, b, transa=self.transa, transb=self.transb)
        expect = self.op_a.dot(self.op_b)
        testing.assert_array_almost_equal(y.toarray(), expect.toarray())


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'transa': [False, True],
}))
@testing.with_requires('scipy')
class TestCsrmv(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        self.op_a = scipy.sparse.random(2, 3, density=0.5, dtype=self.dtype)
        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a
        self.x = numpy.random.uniform(-1, 1, 3).astype(self.dtype)
        self.y = numpy.random.uniform(-1, 1, 2).astype(self.dtype)

    def test_csrmv(self):
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.cusparse.csrmv(
            a, x, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmv_with_y(self):
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.array(self.y, order='f')
        z = cupy.cusparse.csrmv(
            a, x, y=y, alpha=self.alpha, beta=self.beta, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x) + self.beta * self.y
        self.assertIs(y, z)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmvEx_aligned(self):
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')

        self.assertTrue(cupy.cusparse.csrmvExIsAligned(a, x))

    def test_csrmvEx_not_aligned(self):
        a = sparse.csr_matrix(self.a)
        tmp = cupy.array(numpy.hstack([self.x, self.y]), order='f')
        x = tmp[0:len(self.x)]
        y = tmp[len(self.x):]
        self.assertFalse(cupy.cusparse.csrmvExIsAligned(a, x, y))

    def test_csrmvEx(self):
        if self.transa:
            # no support for transa
            return

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.cusparse.csrmvEx(a, x, alpha=self.alpha)
        expect = self.alpha * self.op_a.dot(self.x)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmvEx_with_y(self):
        if self.transa:
            # no support for transa
            return
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.array(self.y, order='f')
        z = cupy.cusparse.csrmvEx(
            a, x, y=y, alpha=self.alpha, beta=self.beta)
        expect = self.alpha * self.op_a.dot(self.x) + self.beta * self.y
        self.assertIs(y, z)
        testing.assert_array_almost_equal(y, expect)


@testing.with_requires('scipy')
class TestCoosort(unittest.TestCase):

    def setUp(self):
        self.a = scipy.sparse.random(
            100, 100, density=0.9, dtype=numpy.float32, format='coo')

    def test_coosort(self):
        a = sparse.coo_matrix(self.a)
        cupy.cusparse.coosort(a)
        # lexsort by row first and col second
        argsort = numpy.lexsort((self.a.col, self.a.row))
        testing.assert_array_equal(self.a.row[argsort], a.row)
        testing.assert_array_equal(self.a.col[argsort], a.col)
        testing.assert_array_almost_equal(self.a.data[argsort], a.data)


@testing.with_requires('scipy')
class TestCsrsort(unittest.TestCase):

    def setUp(self):
        self.a = scipy.sparse.random(
            1, 1000, density=0.9, dtype=numpy.float32, format='csr')
        numpy.random.shuffle(self.a.indices)
        self.a.has_sorted_indices = False

    def test_csrsort(self):
        a = sparse.csr_matrix(self.a)
        cupy.cusparse.csrsort(a)

        self.a.sort_indices()
        testing.assert_array_equal(self.a.indptr, a.indptr)
        testing.assert_array_equal(self.a.indices, a.indices)
        testing.assert_array_almost_equal(self.a.data, a.data)


@testing.with_requires('scipy')
class TestCscsort(unittest.TestCase):

    def setUp(self):
        self.a = scipy.sparse.random(
            1000, 1, density=0.9, dtype=numpy.float32, format='csc')
        numpy.random.shuffle(self.a.indices)
        self.a.has_sorted_indices = False

    def test_csrsort(self):
        a = sparse.csc_matrix(self.a)
        cupy.cusparse.cscsort(a)

        self.a.sort_indices()
        testing.assert_array_equal(self.a.indptr, a.indptr)
        testing.assert_array_equal(self.a.indices, a.indices)
        testing.assert_array_almost_equal(self.a.data, a.data)
