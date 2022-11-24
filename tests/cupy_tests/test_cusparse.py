import functools
import pickle

import numpy
import pytest
try:
    import scipy.sparse
except ImportError:
    pass

import cupy
from cupy import testing
from cupy import cusparse
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse


class TestMatDescriptor:

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
class TestCsrmm:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip:
            if self.transa:
                pytest.xfail('may be buggy')

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm(a, b, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.b)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm_with_c(self):
        if not cusparse.check_availability('csrmm'):
            pytest.skip('csrmm is not available')
        if runtime.is_hip:
            if self.transa:
                pytest.xfail('may be buggy')

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.csrmm(
            a, b, c=c, alpha=self.alpha, beta=self.beta, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.b) + self.beta * self.c
        assert y is c
        testing.assert_array_almost_equal(y, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'trans': [(False, False), (True, False), (False, True)],
}))
@testing.with_requires('scipy')
class TestCsrmm2:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip:
            if self.transa:
                pytest.xfail('may be buggy')

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, alpha=self.alpha, transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmm2_with_c(self):
        if not cusparse.check_availability('csrmm2'):
            pytest.skip('csrmm2 is not available')
        if runtime.is_hip:
            if self.transa:
                pytest.xfail('may be buggy')

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.csrmm2(
            a, b, c=c, alpha=self.alpha, beta=self.beta,
            transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b) + self.beta * self.c
        assert y is c
        testing.assert_array_almost_equal(y, expect)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(3, 4), (4, 3)]
}))
@testing.with_requires('scipy>=1.2.0')
class TestCsrgeam:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
class TestCsrgeamInvalidCases:

    dtype = numpy.float32
    shape = (4, 3)

    @pytest.fixture(autouse=True)
    def setUp(self):
        m, n = self.shape
        self.a = scipy.sparse.random(m, n, density=0.3, dtype=self.dtype)
        self.b = scipy.sparse.random(m, n, density=0.3, dtype=self.dtype)

    def test_csrgeam_invalid_format(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            pytest.skip('csrgeam is not available')
        a = sparse.csc_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        with pytest.raises(TypeError):
            cupy.cusparse.csrgeam(a, b)
        with pytest.raises(TypeError):
            cupy.cusparse.csrgeam(b, a)

    def test_csrgeam_invalid_shape(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            pytest.skip('csrgeam is not available')
        a = sparse.csr_matrix(self.a.T)
        b = sparse.csr_matrix(self.b)
        with pytest.raises(ValueError):
            cupy.cusparse.csrgeam(a, b)

    def test_csrgeam_availability(self):
        if not cupy.cusparse.check_availability('csrgeam'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with pytest.raises(RuntimeError):
                cupy.cusparse.csrgeam(a, b)

    def test_csrgeam2_invalid_format(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            pytest.skip('csrgeam2 is not available')
        a = sparse.csc_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        with pytest.raises(TypeError):
            cupy.cusparse.csrgeam2(a, b)
        with pytest.raises(TypeError):
            cupy.cusparse.csrgeam2(b, a)

    def test_csrgeam2_invalid_shape(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            pytest.skip('csrgeam2 is not available')
        a = sparse.csr_matrix(self.a.T)
        b = sparse.csr_matrix(self.b)
        with pytest.raises(ValueError):
            cupy.cusparse.csrgeam2(a, b)

    def test_csrgeam2_availability(self):
        if not cupy.cusparse.check_availability('csrgeam2'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with pytest.raises(RuntimeError):
                cupy.cusparse.csrgeam2(a, b)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'transa': [False, True],
    'transb': [False, True],
}))
@testing.with_requires('scipy')
class TestCsrgemm:

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip:
            if self.transa or self.transb:
                pytest.xfail('may be buggy')

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
class TestCsrgemm2:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip and driver.get_build_version() < 402:
            pytest.xfail('csrgemm2 is buggy')

        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        d = sparse.csr_matrix(self.d)
        c = cupy.cusparse.csrgemm2(a, b, d=d, alpha=self.alpha, beta=self.beta)
        expect = self.alpha * self.a.dot(self.b) + self.beta * self.d
        testing.assert_array_almost_equal(c.toarray(), expect.toarray())


@testing.with_requires('scipy')
class TestCsrgemm2InvalidCases:

    dtype = numpy.float32
    shape = (2, 3, 4)

    @pytest.fixture(autouse=True)
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
        with pytest.raises(TypeError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csc_matrix(self.b)
        with pytest.raises(TypeError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        d = sparse.csc_matrix(self.d)
        with pytest.raises(TypeError):
            cupy.cusparse.csrgemm2(a, b, d=d)

    def test_csrgemm2_invalid_shape(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            pytest.skip('csrgemm2 is not available.')
        a = sparse.csc_matrix(self.a).T
        b = sparse.csr_matrix(self.b)
        with pytest.raises(ValueError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csc_matrix(self.b).T
        with pytest.raises(ValueError):
            cupy.cusparse.csrgemm2(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        d = sparse.csc_matrix(self.d).T
        with pytest.raises(ValueError):
            cupy.cusparse.csrgemm2(a, b, d=d)

    def test_csrgemm2_availability(self):
        if not cupy.cusparse.check_availability('csrgemm2'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with pytest.raises(RuntimeError):
                cupy.cusparse.csrgemm2(a, b)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(2, 3, 4), (4, 3, 2)]
}))
@testing.with_requires('scipy>=1.2.0')
class TestSpgemm:

    alpha = 0.5

    @pytest.fixture(autouse=True)
    def setUp(self):
        m, n, k = self.shape
        self.a = scipy.sparse.random(m, k, density=0.5, dtype=self.dtype)
        self.b = scipy.sparse.random(k, n, density=0.5, dtype=self.dtype)

    def test_spgemm_ab(self):
        if not cupy.cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available.')

        a = sparse.csr_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        c = cupy.cusparse.spgemm(a, b, alpha=self.alpha)
        expect = self.alpha * self.a.dot(self.b)
        testing.assert_array_almost_equal(c.toarray(), expect.toarray())


@testing.with_requires('scipy')
class TestSpgemmInvalidCases:

    dtype = numpy.float32
    shape = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setUp(self):
        m, n, k = self.shape
        self.a = scipy.sparse.random(m, k, density=0.5, dtype=self.dtype)
        self.b = scipy.sparse.random(k, n, density=0.5, dtype=self.dtype)

    def test_spgemm_invalid_format(self):
        if not cupy.cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available.')
        a = sparse.csc_matrix(self.a)
        b = sparse.csr_matrix(self.b)
        with pytest.raises(TypeError):
            cupy.cusparse.spgemm(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csc_matrix(self.b)
        with pytest.raises(TypeError):
            cupy.cusparse.spgemm(a, b)

    def test_spgemm_invalid_shape(self):
        if not cupy.cusparse.check_availability('spgemm'):
            pytest.skip('spgemm is not available.')
        a = sparse.csc_matrix(self.a).T
        b = sparse.csr_matrix(self.b)
        with pytest.raises(ValueError):
            cupy.cusparse.spgemm(a, b)
        a = sparse.csr_matrix(self.a)
        b = sparse.csc_matrix(self.b).T
        with pytest.raises(ValueError):
            cupy.cusparse.spgemm(a, b)

    def test_spgemm_availability(self):
        if not cupy.cusparse.check_availability('spgemm'):
            a = sparse.csr_matrix(self.a)
            b = sparse.csr_matrix(self.b)
            with pytest.raises(RuntimeError):
                cupy.cusparse.spgemm(a, b)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'transa': [False, True],
}))
@testing.with_requires('scipy')
class TestCsrmv:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip:
            if self.transa:
                pytest.xfail('may be buggy')

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.cusparse.csrmv(
            a, x, alpha=self.alpha, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x)
        testing.assert_array_almost_equal(y, expect)

    def test_csrmv_with_y(self):
        if not cusparse.check_availability('csrmv'):
            pytest.skip('csrmv is not available')
        if runtime.is_hip:
            if self.transa:
                pytest.xfail('may be buggy')

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')
        y = cupy.array(self.y, order='f')
        z = cupy.cusparse.csrmv(
            a, x, y=y, alpha=self.alpha, beta=self.beta, transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x) + self.beta * self.y
        assert y is z
        testing.assert_array_almost_equal(y, expect)

    def test_csrmvEx_aligned(self):
        if not cusparse.check_availability('csrmvEx'):
            pytest.skip('csrmvEx is not available')
        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x, order='f')

        assert cupy.cusparse.csrmvExIsAligned(a, x)

    def test_csrmvEx_not_aligned(self):
        if not cusparse.check_availability('csrmvEx'):
            pytest.skip('csrmvEx is not available')
        a = sparse.csr_matrix(self.a)
        tmp = cupy.array(numpy.hstack([self.x, self.y]), order='f')
        x = tmp[0:len(self.x)]
        y = tmp[len(self.x):]
        assert not cupy.cusparse.csrmvExIsAligned(a, x, y)

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
        assert y is z
        testing.assert_array_almost_equal(y, expect)


@testing.with_requires('scipy')
class TestCoosort:

    @pytest.fixture(autouse=True)
    def setUp(self):
        if not cusparse.check_availability('coosort'):
            pytest.skip('coosort is not available')

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
class TestCsrsort:

    @pytest.fixture(autouse=True)
    def setUp(self):
        if not cusparse.check_availability('csrsort'):
            pytest.skip('csrsort is not available')

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
class TestCscsort:

    @pytest.fixture(autouse=True)
    def setUp(self):
        if not cusparse.check_availability('cscsort'):
            pytest.skip('cscsort is not available')

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
class TestSpmv:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip:
            if ((self.format == 'csr' and self.transa is True)
                    or (self.format == 'csc' and self.transa is False)
                    or (self.format == 'coo' and self.transa is True)):
                pytest.xfail('may be buggy')

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
        if runtime.is_hip:
            if ((self.format == 'csr' and self.transa is True)
                    or (self.format == 'csc' and self.transa is False)
                    or (self.format == 'coo' and self.transa is True)):
                pytest.xfail('may be buggy')

        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        x = cupy.array(self.x)
        y = cupy.array(self.y)
        z = cupy.cusparse.spmv(a, x, y=y, alpha=self.alpha, beta=self.beta,
                               transa=self.transa)
        expect = self.alpha * self.op_a.dot(self.x) + self.beta * self.y
        assert y is z
        testing.assert_array_almost_equal(y, expect)


@testing.with_requires('scipy')
class TestErrorSpmv:

    dtype = numpy.float32

    @pytest.fixture(autouse=True)
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
        with pytest.raises(ValueError):
            cupy.cusparse.spmv(a, x)

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.x)
        with pytest.raises(ValueError):
            cupy.cusparse.spmv(a, x, transa=True)

        a = sparse.csr_matrix(self.a)
        x = cupy.array(self.y)
        with pytest.raises(ValueError):
            cupy.cusparse.spmv(a, x)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'transa': [False, True],
    'transb': [False, True],
    'dims': [(2, 3, 4), (3, 4, 2)],
    'format': ['csr', 'csc', 'coo'],
}))
@testing.with_requires('scipy>=1.2.0')
class TestSpmm:

    alpha = 0.5
    beta = 0.25

    @pytest.fixture(autouse=True)
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
        if runtime.is_hip:
            if ((self.format == 'csr' and self.transa is True)
                    or (self.format == 'csc' and self.transa is False)
                    or (self.format == 'coo' and self.transa is True)):
                pytest.xfail('may be buggy')

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
        if runtime.is_hip:
            if ((self.format == 'csr' and self.transa is True)
                    or (self.format == 'csc' and self.transa is False)
                    or (self.format == 'coo' and self.transa is True)):
                pytest.xfail('may be buggy')

        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.c, order='f')
        y = cupy.cusparse.spmm(
            a, b, c=c, alpha=self.alpha, beta=self.beta,
            transa=self.transa, transb=self.transb)
        expect = self.alpha * self.op_a.dot(self.op_b) + self.beta * self.c
        assert y is c
        testing.assert_array_almost_equal(y, expect)


@testing.with_requires('scipy')
class TestErrorSpmm:

    dtype = numpy.float32

    @pytest.fixture(autouse=True)
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
        with pytest.raises(ValueError):
            cupy.cusparse.spmm(a, b)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        with pytest.raises(AssertionError):
            cupy.cusparse.spmm(a, b.T)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b)
        with pytest.raises(AssertionError):
            cupy.cusparse.spmm(a, b)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.c, order='f')
        with pytest.raises(ValueError):
            cupy.cusparse.spmm(a, b)

        a = sparse.csr_matrix(self.a)
        b = cupy.array(self.b, order='f')
        c = cupy.array(self.b, order='f')
        with pytest.raises(ValueError):
            cupy.cusparse.spmm(a, b, c=c)


@testing.parameterize(*testing.product({
    'lower': [True, False],
    'unit_diag': [True, False],
    'transa': ['N', 'T', 'H'],
    'blocking': [True, False],
    'level_info': [True, False],
    'format': ['csr', 'csc'],
    'nrhs': [None, 1, 4],
    'order': ['C', 'F']
}))
@testing.with_requires('scipy')
class TestCsrsm2:

    n = 6
    alpha = 1.0
    density = 0.75
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _setup(self, dtype):
        dtype = numpy.dtype(dtype)
        self.tol = self._tol[dtype.char.lower()]

        a_shape = (self.n, self.n)
        a = testing.shaped_random(a_shape, numpy, dtype=dtype, scale=1)
        a_mask = testing.shaped_random(a_shape, numpy, dtype='f', scale=1)
        a[a_mask > self.density] = 0
        a_diag = numpy.diag(numpy.ones((self.n,), dtype=dtype))
        if self.unit_diag:
            a[a_diag > 0] = 0
        a = a + a_diag
        cp_a = cupy.array(a)
        if self.unit_diag:
            cp_a[a_diag > 0] = 0.1  # any number except 0
        if self.format == 'csr':
            self.a = sparse.csr_matrix(cp_a)
        elif self.format == 'csc':
            self.a = sparse.csc_matrix(cp_a)

        b_shape = (self.n,) if self.nrhs is None else (self.n, self.nrhs)
        b = numpy.arange(1, numpy.prod(b_shape) + 1,
                         dtype=dtype).reshape(b_shape)
        b = b.copy(order=self.order)
        self.b = cupy.array(b, order=self.order)

        if self.lower:
            a = numpy.tril(a)
        else:
            a = numpy.triu(a)
        if self.transa == 'T':
            a = a.T
        elif self.transa == 'H':
            a = a.conj().T
        self.ref_x = numpy.linalg.solve(a, self.alpha * b)

    @testing.for_dtypes('fdFD')
    def test_csrsm2(self, dtype):
        if not cusparse.check_availability('csrsm2'):
            pytest.skip('csrsm2 is not available')
        if runtime.is_hip:
            if (self.transa == 'H'
                or (driver.get_build_version() < 400
                    and ((self.format == 'csc' and self.transa == 'N')
                         or (self.format == 'csr' and self.transa == 'T')))):
                pytest.xfail('may be buggy')

        if (self.format == 'csc' and numpy.dtype(dtype).char in 'FD' and
                self.transa == 'H'):
            pytest.skip('unsupported combination')
        self._setup(dtype)
        x = self.b.copy(order=self.order)
        cusparse.csrsm2(self.a, x, alpha=self.alpha,
                        lower=self.lower, unit_diag=self.unit_diag,
                        transa=self.transa, blocking=self.blocking,
                        level_info=self.level_info)
        testing.assert_allclose(x, self.ref_x, atol=self.tol, rtol=self.tol)


@testing.parameterize(*testing.product({
    'n': [7, 10],
    'level_info': [True, False],
}))
@testing.with_requires('scipy')
class TestCsrilu02:

    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype):
        if not cusparse.check_availability('csrilu02'):
            pytest.skip('csrilu02 is not available')
        a = testing.shaped_random((self.n, self.n), cupy, dtype=dtype,
                                  scale=0.9) + 0.1
        a = a + cupy.diag(cupy.ones((self.n,), dtype=dtype.char.lower()))
        return a

    @testing.for_dtypes('fdFD')
    def test_csrilu02(self, dtype):
        dtype = numpy.dtype(dtype)
        a_ref = self._make_matrix(dtype)
        a = sparse.csr_matrix(a_ref)
        cusparse.csrilu02(a, level_info=self.level_info)
        a = a.todense()
        al = cupy.tril(a, k=-1)
        al = al + cupy.diag(cupy.ones((self.n,), dtype=dtype.char.lower()))
        au = cupy.triu(a)
        a = al @ au
        tol = self._tol[dtype.char.lower()]
        cupy.testing.assert_allclose(a, a_ref, atol=tol, rtol=tol)

    def test_invalid_cases(self):
        dtype = numpy.dtype('d')
        a_ref = self._make_matrix(dtype)

        # invalid format
        a = sparse.csc_matrix(a_ref)
        with pytest.raises(TypeError):
            cusparse.csrilu02(a, level_info=self.level_info)

        # invalid shape
        a = cupy.ones((self.n, self.n + 1), dtype=dtype)
        a = sparse.csr_matrix(a)
        with pytest.raises(ValueError):
            cusparse.csrilu02(a, level_info=self.level_info)

        # matrix with zero diagonal element
        a = a_ref
        a[-1, -1] = 0
        a = sparse.csr_matrix(a)
        with pytest.raises(ValueError):
            cusparse.csrilu02(a, level_info=self.level_info)

        # singular matrix
        a = a_ref
        a[1:] = a[0]
        a = sparse.csr_matrix(a)
        with pytest.raises(ValueError):
            cusparse.csrilu02(a, level_info=self.level_info)


def skip_HIP_0_size_matrix():
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            try:
                impl(self, *args, **kw)
            except ValueError as e:
                if runtime.is_hip:
                    assert 'hipSPARSE' in str(e)
                    pytest.xfail('may be buggy')
                raise
        return test_func
    return decorator


@testing.parameterize(*testing.product({
    'shape': [(3, 4), (4, 4), (4, 3)],
    'density': [0.0, 0.5, 1.0],
    'format': ['csr', 'csc', 'coo']
}))
@testing.with_requires('scipy')
class TestSparseMatrixConversion:

    @skip_HIP_0_size_matrix()
    @testing.for_dtypes('fdFD')
    def test_denseToSparse(self, dtype):
        if not cusparse.check_availability('denseToSparse'):
            pytest.skip('denseToSparse is not available')
        x = cupy.random.uniform(0, 1, self.shape).astype(dtype)
        x[x < self.density] = 0
        y = cusparse.denseToSparse(x, format=self.format)
        assert y.format == self.format
        testing.assert_array_equal(x, y.todense())

    @skip_HIP_0_size_matrix()
    @testing.for_dtypes('fdFD')
    def test_sparseToDense(self, dtype):
        if not cusparse.check_availability('sparseToDense'):
            pytest.skip('sparseToDense is not available')
        m, n = self.shape
        x = scipy.sparse.random(m, n, density=self.density, format=self.format,
                                dtype=dtype)
        if self.format == 'csr':
            x = sparse.csr_matrix(x)
        elif self.format == 'csc':
            x = sparse.csc_matrix(x)
        elif self.format == 'coo':
            x = sparse.coo_matrix(x)
        y = cusparse.sparseToDense(x)
        testing.assert_array_equal(x.todense(), y)


@pytest.mark.parametrize('dims', [(3, 4), (4, 3), (3, None)])
@pytest.mark.parametrize(
    'dtype', [cupy.float32, cupy.float64, cupy.complex64, cupy.complex128])
@pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
@pytest.mark.parametrize(
    'transa,lower,unit_diag,b_order',
    [('N', True, False, 'f'),   # base
     ('T', True, False, 'f'),   # transa == 'T'
     ('H', True, False, 'f'),   # transa == 'H'
     ('N', False, False, 'f'),  # lower == False
     ('T', False, True, 'f'),   # transa == 'T', lower == False
     ('H', False, True, 'f'),   # transa == 'H', lower == False
     ('N', True, True, 'f'),    # unit_diag == True
     ('N', True, False, 'c'),   # b_order == 'c'
     ])
@testing.with_requires('scipy')
class TestSpsm:

    alpha = 0.5

    def _make_matrix(self, dtype, m, lower, unit_diag, format):
        # Make a sparse m x m triangular non-singular matrix
        a = scipy.sparse.random(
            m, m, density=0.5, format=format, dtype=dtype)

        if unit_diag:
            diag = numpy.diag(numpy.ones(m).astype(dtype))
        else:
            diag = numpy.diag(numpy.random.uniform(0.1, 1, m).astype(dtype))
        a = a - numpy.diag(a.diagonal()) + diag

        if lower:
            a = scipy.sparse.tril(a)
        else:
            a = scipy.sparse.triu(a)

        if not a.has_canonical_format:
            a.sum_duplicates()

        return a

    @pytest.fixture(autouse=True)
    def setUp(self, dims, dtype, lower, unit_diag, transa, format):
        m, n = dims

        self.op_a = self._make_matrix(dtype, m, lower, unit_diag, format)
        self.a = self.op_a

        if n is None:
            b_shape = m,
        else:
            b_shape = m, n
        self.op_b = numpy.random.uniform(-1, 1, b_shape).astype(dtype)
        self.b = self.op_b

        if format == 'csr':
            self.sparse_matrix = sparse.csr_matrix
        elif format == 'csc':
            self.sparse_matrix = sparse.csc_matrix
        elif format == 'coo':
            self.sparse_matrix = sparse.coo_matrix
        else:
            assert False

    def test_spsm(self, lower, unit_diag, transa, b_order, dtype, format):
        if not cusparse.check_availability('spsm'):
            pytest.skip('spsm is not available')
        if not runtime.is_hip and _cusparse.get_build_version() < 11701:
            # eariler than CUDA 11.6
            if b_order == 'c':
                pytest.skip("Older CUDA has a bug")
        if runtime.is_hip:
            if format == 'coo' or b_order == 'c':
                pytest.skip('may be buggy or not supported')
        a = self.sparse_matrix(self.a)
        b = cupy.array(self.b, order=b_order)
        c = cusparse.spsm(
            a, b, alpha=self.alpha, lower=lower, unit_diag=unit_diag,
            transa=transa)

        if transa == 'N':
            op_a = self.op_a
        elif transa == 'T':
            op_a = self.op_a.T
        else:
            op_a = self.op_a.conj().T
        lhs = op_a.dot(c.get())

        rhs = self.alpha * self.op_b

        if dtype in (cupy.float32, cupy.complex64):
            tol = 1e-5
        else:
            tol = 1e-12
        testing.assert_allclose(lhs, rhs, rtol=tol, atol=tol)
