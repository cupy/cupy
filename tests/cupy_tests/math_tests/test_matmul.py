import operator
import unittest

import numpy

from cupy import testing


@testing.parameterize(
    *testing.product({
        'arrays': [
            (numpy.random.randn(*s1), numpy.random.randn(*s2)) for s1, s2 in [
                ((3, 2), (2, 4)),
                ((2,), (2, 4)),
                ((3, 2), (2,)),
                ((2,), (2,)),
                ((5, 3, 2), (5, 2, 4)),
                ((5, 3, 2), (2, 4)),
                ((3, 2), (5, 2, 4)),
                ((5, 3, 2), (1, 2, 4)),
                ((1, 3, 2), (5, 2, 4)),
                ((5, 3, 2), (2,)),
                ((2,), (5, 2, 4)),
                ((6, 5, 3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (6, 1, 2, 4)),
                ((6, 5, 3, 2), (1, 5, 2, 4)),
                ((6, 5, 3, 2), (1, 1, 2, 4)),
                ((6, 1, 3, 2), (6, 5, 2, 4)),
                ((1, 5, 3, 2), (6, 5, 2, 4)),
                ((1, 1, 3, 2), (6, 5, 2, 4)),
                ((3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2, 4)),
                ((2,), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2,)),
            ]],
    }))
@testing.gpu
class TestMatmul(unittest.TestCase):

    # _multiprocess_can_split_ = True

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_almost_equal()
    def test_matmul(self, xp, dtype1, dtype2):
        x1, x2 = self.arrays
        x1, x2 = xp.array(x1), xp.array(x2)
        return operator.matmul(x1, x2)
