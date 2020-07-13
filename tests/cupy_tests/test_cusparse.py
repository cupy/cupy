import pickle
import unittest

import numpy
import pytest
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
        if not cusparse.check_availability('csrmm'):
            pytest.skip('csrmm is not available')
        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm(a, b, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.b)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm_with_c(self):
        if not cusparse.check_availability('csrmm'):
            pytest.skip('csrmm is not available')
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
        if not cusparse.check_availability('csrmm2'):
            pytest.skip('csrmm2 is not available')
        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, alpha=self.alpha, transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm2_with_c(self):
        if not cusparse.check_availability('csrmm2'):
            pytest.skip('csrmm2 is not available')
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
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(3, 4), (4, 3)]
}))
@testing.with_requires('scipy>=1.2.0')
class TestCsrgeam(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        m, n = self.shape
        self.a = scipy.sparse.random(m, n, density=0.3, dtype=self.dtype)
        self.b = scipy.sparse.random(m, n, density=0.3, dtype=self.dtype)

    def test_csrgeam(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            pytest.skip('csrgeam is not available')
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        c = cupy.cusparse.csrgeam(a, b, alpha=self.alpha, beta=self.beta)
        expect = self.alpha * self.a + self.beta * self.b
        testing.assert_array_almost_equal(c.toarray(), expect.toarray())

    def test_csrgeam2(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            pytest.skip('csrgeam2 is not available')
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        c = cupy.cusparse.csrgeam2(a, b, alpha=self.alpha, beta=self.beta)
        expect = self.alpha * self.a + self.beta * self.b
        testing.assert_array_almost_equal(c.toarray(), expect.toarray())


@testing.with_requires('scipy')
class TestCsrgeamInvalidCases(unittest.TestCase):

    dtype = numpy.float32
    shape = (4, 3)

    def setUp(self):
        m, n = self.shape
        self.a = scipy.sparse.random(m, n, density=0.3, dtype=self.dtype)
        self.b = scipy.sparse.random(m, n, density=0.3, dtype=self.dtype)

    def test_csrgeam_invalid_format(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            pytest.skip('csrgeam is not available')
        a = sparse.csc_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgeam(a, b)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgeam(b, a)

    def test_csrgeam_invalid_shape(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            pytest.skip('csrgeam is not available')
        a = sparse.csr_matrix(self.a.T)
        b = sparse.csr_matrix(self.b)
        with self.assertRaises(ValueError):
            cupy.cusparse.csrgeam(a, b)

    def test_csrgeam_availability(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with self.assertRaises(RuntimeError):
                cupy.cusparse.csrgeam(a, b)

    def test_csrgeam2_invalid_format(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            pytest.skip('csrgeam2 is not available')
        a = sparse.csc_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgeam2(a, b)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgeam2(b, a)

    def test_csrgeam2_invalid_shape(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            pytest.skip('csrgeam2 is not available')
        a = sparse.csr_matrix(self.a.T)
        b = sparse.csr_matrix(self.b)
        with self.assertRaises(ValueError):
            cupy.cusparse.csrgeam2(a, b)

    def test_csrgeam2_availability(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with self.assertRaises(RuntimeError):
                cupy.cusparse.csrgeam2(a, b)


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
        if not cupy.cusparse.check_availability('csrgemm'):
            pytest.skip('csrgemm is not available.')

        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        y = cupy.cusparse.csrgemm(a, b, transa=self.transa, transb=self.transb)
        expect = self.op_a.dot(self.op_b)
        testing.assert_array_almost_equal(y.toarray(), expect.toarray())


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(2, 3, 4), (4, 3, 2)]
}))
@testing.with_requires('scipy>=1.2.0')
class TestCsrgemm2(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        m, n, k = self.shape
        self.a = scipy.sparse.random(m, k, density=0.5, dtype=self.dtype)
        self.b = scipy.sparse.random(k, n, density=0.5, dtype=self.dtype)
        self.d = scipy.sparse.random(m, n, density=0.5, dtype=self.dtype)

    def test_csrgemm2_ab(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            pytest.skip('csrgemm2 is not available.')

        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        c = cupy.cusparse.csrgemm2(a, b, alpha=self.alpha)
        expect = self.alpha * self.a.dot(self.b)
        testing.assert_array_almost_equal(c.toarray(), expect.toarray())

    def test_csrgemm2_abpd(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            pytest.skip('csrgemm2 is not available.')

        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        d = sparse.csr_matrix(self.d)
        c = cupy.cusparse.csrgemm2(a, b, d=d, alpha=self.alpha, beta=self.beta)
        expect = self.alpha * self.a.dot(self.b) + self.beta * self.d
        testing.assert_array_almost_equal(c.toarray(), expect.toarray())


@testing.with_requires('scipy')
class TestCsrgemm2InvalidCases(unittest.TestCase):

    dtype = numpy.float32
    shape = (2, 3, 4)

    def setUp(self):
        m, n, k = self.shape
        self.a = scipy.sparse.random(m, k, density=0.5, dtype=self.dtype)
        self.b = scipy.sparse.random(k, n, density=0.5, dtype=self.dtype)
        self.d = scipy.sparse.random(m, n, density=0.5, dtype=self.dtype)

    def test_csrgemm2_invalid_format(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            pytest.skip('csrgemm2 is not available.')
        a = sparse.csc_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csc_matrix(self.b)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        d = sparse.csc_matrix(self.d)
        with self.assertRaises(TypeError):
            cupy.cusparse.csrgemm2(a, b, d=d)

    def test_csrgemm2_invalid_shape(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            pytest.skip('csrgemm2 is not available.')
        a = sparse.csc_matrix(self.a).T
        b = sparse.csr_matrix(self.b)
        with self.assertRaises(ValueError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csc_matrix(self.b).T
        with self.assertRaises(ValueError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        d = sparse.csc_matrix(self.d).T
        with self.assertRaises(ValueError):
            cupy.cusparse.csrgemm2(a, b, d=d)

    def test_csrgemm2_availability(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with self.assertRaises(RuntimeError):
                cupy.cusparse.csrgemm2(a, b)


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
        if not cusparse.check_availability('csrmv'):
            pytest.skip('csrmv is not available')
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.cusparse.csrmv(
            a, x, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmv_with_y(self):
        if not cusparse.check_availability('csrmv'):
            pytest.skip('csrmv is not available')
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.array(self.y, order='f')
        z = cupy.cusparse.csrmv(
            a, x, y=y, alpha=self.alpha, beta=self.beta, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x) + self.beta * self.y
        self.assertIs(y, z)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmvEx_aligned(self):
        if not cusparse.check_availability('csrmvEx'):
            pytest.skip('csrmvEx is not available')
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')

        self.assertTrue(cupy.cusparse.csrmvExIsAligned(a, x))

    def test_csrmvEx_not_aligned(self):
        if not cusparse.check_availability('csrmvEx'):
            pytest.skip('csrmvEx is not available')
        a = sparse.csr_matrix(self.a)
        tmp = cupy.array(numpy.hstack([self.x, self.y]), order='f')
        x = tmp[0:len(self.x)]
        y = tmp[len(self.x):]
        self.assertFalse(cupy.cusparse.csrmvExIsAligned(a, x, y))

    def test_csrmvEx(self):
        if not cusparse.check_availability('csrmvEx'):
            pytest.skip('csrmvEx is not available')
        if self.transa:
            # no support for transa
            return

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.cusparse.csrmvEx(a, x, alpha=self.alpha)
        expect = self.alpha * self.op_a.dot(self.x)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmvEx_with_y(self):
        if not cusparse.check_availability('csrmvEx'):
            pytest.skip('csrmvEx is not available')
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
        numpy.random.shuffle(self.a.row)
        numpy.random.shuffle(self.a.col)

    def test_coosort(self):
        a = sparse.coo_matrix(self.a)
        cupy.cusparse.coosort(a)
        # lexsort by row first and col second
        argsort = numpy.lexsort((self.a.col, self.a.row))
        testing.assert_array_equal(self.a.row[argsort], a.row)
        testing.assert_array_equal(self.a.col[argsort], a.col)
        testing.assert_array_almost_equal(self.a.data[argsort], a.data)

    def test_coosort_by_column(self):
        a = sparse.coo_matrix(self.a)
        cupy.cusparse.coosort(a, sort_by='c')
        # lexsort by col first and row second
        argsort = numpy.lexsort((self.a.row, self.a.col))
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

    def test_cscsort(self):
        a = sparse.csc_matrix(self.a)
        cupy.cusparse.cscsort(a)

        self.a.sort_indices()
        testing.assert_array_equal(self.a.indptr, a.indptr)
        testing.assert_array_equal(self.a.indices, a.indices)
        testing.assert_array_almost_equal(self.a.data, a.data)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'transa': [False, True],
    'shape': [(3, 2), (4, 3)],
    'format': ['csr', 'csc', 'coo'],
}))
@testing.with_requires('scipy>=1.2.0')
class TestSpmv(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        m, n = self.shape
        self.op_a = scipy.sparse.random(m, n, density=0.5, format=self.format,
                                        dtype=self.dtype)
        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a
        self.x = numpy.random.uniform(-1, 1, n).astype(self.dtype)
        self.y = numpy.random.uniform(-1, 1, m).astype(self.dtype)
        if self.format == 'csr':
            self.sparse_matrix = sparse.csr_matrix
        elif self.format == 'csc':
            self.sparse_matrix = sparse.csc_matrix
        elif self.format == 'coo':
            self.sparse_matrix = sparse.coo_matrix

    def test_spmv(self):
        if not cupy.cusparse.check_availability('spmv'):
            pytest.skip('spmv is not available')
        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        x = cupy.array(self.x)
        y = cupy.cusparse.spmv(a, x, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x)
        testing.assert_array_almost_equal(y, expect)

    def test_spmv_with_y(self):
        if not cupy.cusparse.check_availability('spmv'):
            pytest.skip('spmv is not available')
        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        x = cupy.array(self.x)
        y = cupy.array(self.y)
        z = cupy.cusparse.spmv(a, x, y=y, alpha=self.alpha, beta=self.beta,
                               transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x) + self.beta * self.y
        self.assertIs(y, z)
        testing.assert_array_almost_equal(y, expect)


@testing.with_requires('scipy')
class TestErrorSpmv(unittest.TestCase):

    dtype = numpy.float32

    def setUp(self):
        m, n = 2, 3
        self.a = scipy.sparse.random(m, n, density=0.5,
                                     dtype=self.dtype)
        self.x = numpy.random.uniform(-1, 1, n).astype(self.dtype)
        self.y = numpy.random.uniform(-1, 1, m).astype(self.dtype)

    def test_error_shape(self):
        if not cupy.cusparse.check_availability('spmv'):
            pytest.skip('spmv is not available')

        a = sparse.csr_matrix(self.a.T)
        x = cupy.array(self.x)
        with self.assertRaises(ValueError):
            cupy.cusparse.spmv(a, x)

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x)
        with self.assertRaises(ValueError):
            cupy.cusparse.spmv(a, x, transa=True)

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.y)
        with self.assertRaises(ValueError):
            cupy.cusparse.spmv(a, x)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'transa': [False, True],
    'transb': [False, True],
    'dims': [(2, 3, 4), (3, 4, 2)],
    'format': ['csr', 'csc', 'coo'],
}))
@testing.with_requires('scipy>=1.2.0')
class TestSpmm(unittest.TestCase):

    alpha = 0.5
    beta = 0.25

    def setUp(self):
        m, n, k = self.dims
        self.op_a = scipy.sparse.random(m, k, density=0.5, format=self.format,
                                        dtype=self.dtype)
        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a
        self.op_b = numpy.random.uniform(-1, 1, (k, n)).astype(self.dtype)
        if self.transb:
            self.b = self.op_b.T
        else:
            self.b = self.op_b
        self.c = numpy.random.uniform(-1, 1, (m, n)).astype(self.dtype)
        if self.format == 'csr':
            self.sparse_matrix = sparse.csr_matrix
        elif self.format == 'csc':
            self.sparse_matrix = sparse.csc_matrix
        elif self.format == 'coo':
            self.sparse_matrix = sparse.coo_matrix

    def test_spmm(self):
        if not cupy.cusparse.check_availability('spmm'):
            pytest.skip('spmm is not available')
        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        b = cupy.array(self.b, order='f')
        c = cupy.cusparse.spmm(
            a, b, alpha=self.alpha, transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b)
        testing.assert_array_almost_equal(c, expect)

    def test_spmm_with_c(self):
        if not cupy.cusparse.check_availability('spmm'):
            pytest.skip('spmm is not available')
        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.spmm(
            a, b, c=c, alpha=self.alpha, beta=self.beta,
            transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b) + self.beta * self.c
        self.assertIs(y, c)
        testing.assert_array_almost_equal(y, expect)


@testing.with_requires('scipy')
class TestErrorSpmm(unittest.TestCase):

    dtype = numpy.float32

    def setUp(self):
        m, n, k = 2, 3, 4
        self.a = scipy.sparse.random(m, k, density=0.5,
                                     dtype=self.dtype)
        self.b = numpy.random.uniform(-1, 1, (k, n)).astype(self.dtype)
        self.c = numpy.random.uniform(-1, 1, (m, n)).astype(self.dtype)

    def test_error_shape(self):
        if not cupy.cusparse.check_availability('spmm'):
            pytest.skip('spmm is not available')

        a = sparse.csr_matrix(self.a.T)
        b = cupy.array(self.b, order='f')
        with self.assertRaises(ValueError):
            cupy.cusparse.spmm(a, b)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        with self.assertRaises(AssertionError):
            cupy.cusparse.spmm(a, b.T)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b)
        with self.assertRaises(AssertionError):
            cupy.cusparse.spmm(a, b)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.c, order='f')
        with self.assertRaises(ValueError):
            cupy.cusparse.spmm(a, b)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.b, order='f')
        with self.assertRaises(ValueError):
            cupy.cusparse.spmm(a, b, c=c)
