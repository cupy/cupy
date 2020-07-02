import unittest

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.2, 0.4, 0.5, 0.9, 1.0],
    'dtype': ['float32', 'float64'],
    'n_rows': [100, 1000],
    'n_cols': [10, 100]
}))
@testing.with_requires('scipy')
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None):
        a = cupy.sparse.random(self.n_rows, self.n_cols,
                               format=self.format,
                               density=self.density,
                               dtype=self.dtype)
        if min is not None:
            expected = a.get()[maj, min]
            actual = a[maj, min]
        else:
            expected = a.get()[maj]
            actual = a[maj]

        if cupy.sparse.isspmatrix(actual):
            actual = actual.todense()
            expected = expected.todense()

        cupy.testing.assert_array_equal(actual.ravel(),
                                        cupy.array(expected).ravel())

    def test_major_slice(self):
        self._run(slice(5, 9))
        self._run(slice(9, 5))

    def test_major_all(self):
        self._run(slice(None))

    def test_major_scalar(self):
        self._run(10)

    def test_major_fancy(self):
        self._run([1, 5, 4])
        self._run([10, 2])
        self._run([2])

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 5), slice(1, 5))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 5), slice(None))
        self._run(slice(5, 1), slice(None))

    def test_major_slice_minor_scalar(self):
        self._run(slice(1, 5), 5)
        self._run(slice(5, 1), 5)
        self._run(slice(5, 1, -1), 5)

    def test_major_slice_minor_fancy(self):
        self._run(slice(1, 10, 2), [1, 5, 4])

    def test_major_scalar_minor_slice(self):
        self._run(5, slice(1, 5))

    def test_major_scalar_minor_all(self):
        self._run(5, slice(None))

    def test_major_scalar_minor_scalar(self):
        self._run(5, 5)

    def test_major_scalar_minor_fancy(self):
        self._run(5, [1, 5, 4])

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 10))

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        self._run(slice(None), [1, 5, 4])

    def test_major_fancy_minor_fancy(self):
        self._run([1, 5, 4], [1, 5, 4])
        self._run([2, 0, 10], [9, 2, 1])
        self._run([2, 0], [2, 1])

    def test_major_fancy_minor_all(self):
        self._run([1, 5, 4], slice(None))

    def test_major_fancy_minor_scalar(self):
        self._run([1, 5, 4], 5)

    def test_major_fancy_minor_slice(self):
        self._run([1, 5, 4], slice(1, 5))
        self._run([1, 5, 4], slice(5, 1, -1))

    def test_major_bool_fancy(self):
        rand_bool = cupy.random.random(self.n_rows).astype(cupy.bool)
        self._run(rand_bool.get())

    def test_major_slice_with_step(self):

        # positive step
        self._run(slice(1, 10, 2))
        self._run(slice(2, 10, 5))
        self._run(slice(0, 10, 10))

        # negative step
        self._run(slice(10, 1, 2))
        self._run(slice(10, 2, 5))
        self._run(slice(10, 0, 10))

    def test_major_slice_with_step_minor_slice_with_step(self):

        # positive step
        self._run(slice(1, 10, 2), slice(1, 10, 2))
        self._run(slice(2, 10, 5), slice(2, 10, 5))
        self._run(slice(0, 10, 10), slice(0, 10, 10))

        # negative step
        self._run(slice(10, 1, 2), slice(10, 1, 2))
        self._run(slice(10, 2, 5), slice(10, 2, 5))
        self._run(slice(10, 0, 10), slice(10, 0, 10))

    def test_major_slice_with_step_minor_all(self):

        # positive step
        self._run(slice(1, 10, 2), slice(None))
        self._run(slice(2, 10, 5), slice(None))
        self._run(slice(0, 10, 10), slice(None))

        # negative step
        self._run(slice(10, 1, 2), slice(None))
        self._run(slice(10, 2, 5), slice(None))
        self._run(slice(10, 0, 10), slice(None))

    def test_major_all_minor_slice_step(self):

        # positive step incr
        self._run(slice(None), slice(1, 10, 2))
        self._run(slice(None), slice(2, 10, 5))
        self._run(slice(None), slice(0, 10, 10))

        # positive step decr
        self._run(slice(None), slice(10, 1, 2))
        self._run(slice(None), slice(10, 2, 5))
        self._run(slice(None), slice(10, 0, 10))

        # positive step incr
        self._run(slice(None), slice(10, 1, 2))
        self._run(slice(None), slice(10, 2, 5))
        self._run(slice(None), slice(10, 0, 10))

        # negative step decr
        self._run(slice(None), slice(10, 1, -2))
        self._run(slice(None), slice(10, 2, -5))
        self._run(slice(None), slice(10, 0, -10))

    def test_major_reorder(self):
        self._run(slice(None, None, -1))

    def test_major_reorder_minor_reorder(self):
        self._run(slice(None, None, -1), slice(None, None, -1))
