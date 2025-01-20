import unittest

from cupy import testing
import cupyx.scipy.linalg  # NOQA

try:
    import scipy.linalg  # NOQA
except ImportError:
    pass


class TestSpecialMatricesBase(unittest.TestCase):
    def _get_arg(self, xp, arg):
        if isinstance(arg, tuple):
            # Allocate array with the given shape
            return testing.shaped_random(arg, xp)

        # Otherwise just pass the arg back
        return arg


@testing.parameterize(*(
    testing.product({
        # 1 argument: 1D array
        'function': ['circulant', 'toeplitz', 'hankel', 'companion'],
        'args': [((1,),), ((2,),), ((4,),), ((10,),), ((25,),)],
    }) + testing.product({
        # 2 arguments: both 1D arrays
        # For leslie, second array should be 1 less long than first
        'function': ['toeplitz', 'hankel', 'leslie'],
        'args': [((1,), (1,)), ((2,), (1,)), ((4,), (5,)),
                 ((10,), (9,)), ((25,), (24,))],
    }) + testing.product({
        # 1 argument: int
        'function': ['hadamard', 'helmert', 'hilbert', 'dft'],
        'args': [(1,), (2,), (4,), (10,), (25,)],
    }) + testing.product({
        # 2 arguments: int, dtype
        'function': ['hadamard'],
        'args': [(4, 'int32'), (8, 'float64'), (6, 'float64')],
    }) + testing.product({
        # 2 arguments: int, bool
        'function': ['helmert'],
        'args': [(4, True), (5, False)],
    }) + testing.product({
        # 2 arguments: int, str
        'function': ['dft'],
        'args': [(4, 'sqrtn'), (5, 'n')],
    }) + testing.product({
        # 2 arguments: both 2D arrays
        'function': ['kron'],
        'args': [((5, 5), (4, 5),), ((5, 4), (4, 4),), ((1, 2), (4, 5))],
    }) + testing.product({
        # 0 or more arguments: all 1 or 2D arrays
        'function': ['block_diag'],
        'args': [(), ((5,),), ((4, 5),), ((5,), (4, 4),), ((1, 2), (4, 5)),
                 ((1,), (4, 5), (3, 6), (7, 2), (8, 9))],
    })
))
@testing.with_requires('scipy')
class TestSpecialMatrices(TestSpecialMatricesBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_special_matrix(self, xp, scp):
        function = getattr(scp.linalg, self.function)
        return function(*[self._get_arg(xp, arg) for arg in self.args])


@testing.parameterize(*(
    testing.product({
        # 1 argument: 1D array
        'function': ['fiedler', 'fiedler_companion'],
        'args': [((0,),), ((1,),), ((2,),), ((4,),), ((10,),), ((25,),)],
    })
))
@testing.with_requires('scipy>=1.3.0')
class TestSpecialMatrices_1_3_0(TestSpecialMatricesBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_special_matrix(self, xp, scp):
        function = getattr(scp.linalg, self.function)
        return function(*[self._get_arg(xp, arg) for arg in self.args])

    def _get_arg(self, xp, arg):
        if isinstance(arg, tuple):
            # Allocate array with the given shape
            return testing.shaped_random(arg, xp)

        # Otherwise just pass the arg back
        return arg


@testing.parameterize(*(
    testing.product({
        # 2-3 arguments: 1D array, 1 int, 1 optional str
        'function': ['convolution_matrix'],
        'args': [((1,), 5), ((2,), 3), ((4,), 10), ((10,), 15), ((25,), 25),
                 ((4,), 6, 'full'), ((10,), 8, 'same'), ((25,), 25, 'valid')],
    })
))
@testing.with_requires('scipy>=1.5.0')
class TestSpecialMatrices_1_5_0(TestSpecialMatricesBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_special_matrix(self, xp, scp):
        function = getattr(scp.linalg, self.function)
        return function(*[self._get_arg(xp, arg) for arg in self.args])
