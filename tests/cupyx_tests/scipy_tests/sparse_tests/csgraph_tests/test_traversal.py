import unittest

import numpy
try:
    import scipy.sparse  # NOQA
    import scipy.sparse.csgraph  # NOQA
    scipy_available = True
except ImportError:
    scipy_available = False
import cupyx.scipy.sparse.csgraph  # NOQA
try:
    from cupy_backends.cuda.libs import cugraph  # NOQA
    cugraph_available = True
except ImportError:
    cugraph_available = False
from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64'],
    'm': [5, 10, 20],
    'nnz_per_row': [1, 2, 3],
    'directed': [True, False],
    'connection': ['weak', 'strong'],
    'return_labels': [True, False],
}))
@unittest.skipUnless(scipy_available and cugraph_available,
                     'requires scipy and cugraph')
class TestConnectedComponents(unittest.TestCase):

    def _make_matrix(self, dtype, xp):
        shape = (self.m, self.m)
        density = self.nnz_per_row / self.m
        a = testing.shaped_random(shape, xp, dtype=dtype, scale=1)
        a = a / density
        a[a > 1] = 0
        return a

    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_connected_components(self, xp, sp):
        a = self._make_matrix(self.dtype, xp)
        a = sp.csr_matrix(a)
        if self.return_labels:
            n, labels = sp.csgraph.connected_components(
                a, directed=self.directed, connection=self.connection,
                return_labels=self.return_labels)
            if xp == numpy and self.directed and self.connection == 'strong':
                # Note: Label IDs are adjusted as SciPy returns un-ordered
                # labels in strong connection case.
                table = xp.zeros((n,), dtype=numpy.int32) - 1
                j = 0
                for i in range(labels.size):
                    if table[labels[i]] < 0:
                        table[labels[i]] = j
                        j = j + 1
                    labels[i] = table[labels[i]]
            return n, labels
        else:
            return sp.csgraph.connected_components(
                a, directed=self.directed, connection=self.connection,
                return_labels=self.return_labels)
