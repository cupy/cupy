import unittest

import numpy

import cupy
from cupy import testing
import cupyx

try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False


@testing.gpu
class TestGetArrayModule(unittest.TestCase):

    def test_get_array_module_1(self):
        n1 = numpy.array([2], numpy.float32)
        c1 = cupy.array([2], numpy.float32)
        csr1 = cupyx.scipy.sparse.csr_matrix((5, 3), dtype=numpy.float32)

        assert numpy is cupy.get_array_module()
        assert numpy is cupy.get_array_module(n1)
        assert cupy is cupy.get_array_module(c1)
        assert cupy is cupy.get_array_module(csr1)

        assert numpy is cupy.get_array_module(n1, n1)
        assert cupy is cupy.get_array_module(c1, c1)
        assert cupy is cupy.get_array_module(csr1, csr1)

        assert cupy is cupy.get_array_module(n1, csr1)
        assert cupy is cupy.get_array_module(csr1, n1)
        assert cupy is cupy.get_array_module(c1, n1)
        assert cupy is cupy.get_array_module(n1, c1)
        assert cupy is cupy.get_array_module(c1, csr1)
        assert cupy is cupy.get_array_module(csr1, c1)

        if scipy_available:
            csrn1 = scipy.sparse.csr_matrix((5, 3), dtype=numpy.float32)

            assert numpy is cupy.get_array_module(csrn1)
            assert cupy is cupy.get_array_module(csrn1, csr1)
            assert cupy is cupy.get_array_module(csr1, csrn1)
            assert cupy is cupy.get_array_module(c1, csrn1)
            assert cupy is cupy.get_array_module(csrn1, c1)
            assert numpy is cupy.get_array_module(n1, csrn1)
            assert numpy is cupy.get_array_module(csrn1, n1)
