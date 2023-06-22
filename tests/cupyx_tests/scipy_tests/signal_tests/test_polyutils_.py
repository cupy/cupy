import numpy as np
import pytest

import cupy
from cupy import testing
import cupyx.scipy.signal

try:
    import scipy.signal  # NOQA
except ImportError:
    pass


@testing.with_requires("scipy")
class TestUniqueRoots:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_real_no_repeat(self, xp, scp):
        p = xp.asarray([-1.0, -0.5, 0.3, 1.2, 10.0])
        unique, multiplicity = scp.signal.unique_roots(p)
        return unique, multiplicity

    @pytest.mark.parametrize('rtype', ['min', 'max', 'avg'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_real_repeat(self, xp, scp, rtype):
        p = xp.asarray([-1.0, -0.95, -0.89, -0.8, 0.5, 1.0, 1.05])

        unique, multiplicity = scp.signal.unique_roots(
            p, tol=1e-1, rtype=rtype)
        return unique, multiplicity

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_no_repeat(self, xp, scp):
        p = xp.asarray([-1.0, 1.0j, 0.5 + 0.5j, -1.0 - 1.0j, 3.0 + 2.0j])
        unique, multiplicity = scp.signal.unique_roots(p)
        return unique, multiplicity

    @pytest.mark.parametrize('rtype', ['min', 'max', 'avg'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_repeat(self, xp, scp, rtype):
        p = xp.asarray([-1.0, -1.0 + 0.05j, -0.95 + 0.15j, -0.90 + 0.15j, 0.0,
                        0.5 + 0.5j, 0.45 + 0.55j])

        unique, multiplicity = scp.signal.unique_roots(
            p, tol=1e-1, rtype='min')
        return unique, multiplicity

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_gh_4915(self, xp, scp):
        p = np.roots(np.convolve(np.ones(5), np.ones(5)))
        # true_roots = [-(-1)**(1/5), (-1)**(4/5), -(-1)**(3/5), (-1)**(2/5)]

        unique, multiplicity = scp.signal.unique_roots(xp.asarray(p))
        unique = xp.sort(unique)
        return unique, multiplicity

    @pytest.mark.parametrize('p', [[1.0, 1.0j, 1.0], [1, 1 + 2e-9, 1e-9 + 1j]])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_roots_extra(self, xp, scp, p):
        p = xp.asarray(p)
        unique, multiplicity = scp.signal.unique_roots(p)
        return unique, multiplicity

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_single_unique_root(self, xp, scp):
        np.random.seed(1234)
        p = np.random.rand(100) + 1j * np.random.rand(100)
        p = xp.asarray(p)

        unique, multiplicity = scp.signal.unique_roots(p, 2)
        return unique, multiplicity


rtypes = ('avg', 'mean', 'min', 'minimum', 'max', 'maximum')


@testing.with_requires("scipy")
class TestPartialFractionExpansion:

    def test_compute_factors(self):

        from cupyx.scipy.signal._polyutils import _compute_factors

        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1])
        assert len(factors) == 3
        testing.assert_allclose(factors[0], [1, -7, 16, -12], atol=1e-15)
        testing.assert_allclose(factors[1], [1, -6, 12, -10, 3], atol=1e-15)
        testing.assert_allclose(
            factors[2], [1, -7, 19, -25, 16, -4], atol=1e-15)
        testing.assert_allclose(
            poly, [1, -10, 40, -82, 91, -52, 12], atol=1e-15)

        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1],
                                         include_powers=True)
        assert len(factors) == 6
        testing.assert_allclose(
            factors[0], [1,  -9,  31, -51,  40, -12], atol=1e-15)
        testing.assert_allclose(
            factors[1], [1,  -8,  23, -28,  12], atol=1e-15)
        testing.assert_allclose(factors[2], [1,  -7,  16, -12], atol=1e-15)
        testing.assert_allclose(
            factors[3], [1,  -8,  24, -34,  23,  -6], atol=1e-15)
        testing.assert_allclose(
            factors[4], [1,  -6,  12, -10,   3], atol=1e-15)
        testing.assert_allclose(
            factors[5], [1,  -7,  19, -25,  16,  -4], atol=1e-15)

        testing.assert_allclose(
            poly, [1, -10,  40, -82,  91, -52,  12], atol=1e-15)

    def test_group_poles(self):
        _group_poles = cupyx.scipy.signal.unique_roots

        unique, multiplicity = _group_poles(
            cupy.asarray([1.0, 1.001, 1.003, 2.0, 2.003, 3.0]), 0.1, 'min')
        testing.assert_allclose(unique, [1.0, 2.0, 3.0], atol=1e-15)
        testing.assert_allclose(multiplicity, [3, 2, 1], atol=1e-15)

    @pytest.mark.parametrize('ba',
                             [([5, 3, -2, 7], [-4, 0, 8, 3]),
                              ([-4, 8], [1, 6, 8]),
                                 ([4, 1], [1, -1, -2]),
                                 ([4, 3], [2, -3.4, 1.98, -0.406]),
                                 ([2, 1], [1, 5, 8, 4]),
                                 ([3, -1.1, 0.88, -2.396, 1.348],
                                  [1, -0.7, -0.14, 0.048]),
                                 ([1], [1, 2, -3]),
                                 ([1, 0, -5], [1, 0, 0, 0, -1]),
                                 ([3, 8, 6], [1, 3, 3, 1]),
                                 ([3, -1], [1, -3, 2]),
                                 ([2, 3, -1], [1, -3, 2]),
                                 ([7, 2, 3, -1], [1, -3, 2]),
                                 ([2, 3, -1], [1, -3, 4, -2]),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_residue_general(self, xp, scp, ba):
        # Test are taken from issue #4464, note that poles in scipy are
        # in increasing by absolute value order, opposite to MATLAB.
        b, a = map(xp.asarray, ba)
        r, p, k = scp.signal.residue(b, a)
        return r, p, k

    @pytest.mark.parametrize('ba',
                             [([5, 3, -2, 7], [-4, 0, 8, 3]),
                              ([0, 5, 3, -2, 7], [-4, 0, 8, 3]),
                                 ([5, 3, -2, 7], [0, -4, 0, 8, 3]),
                                 ([0, 0, 5, 3, -2, 7], [0, 0, 0, -4, 0, 8, 3]),
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_residue_leading_zeros(self, xp, scp, ba):
        # Leading zeros in numerator or denominator must not affect the answer.
        ba = map(xp.asarray, ba)
        r0, p0, k0 = scp.signal.residue(*ba)
        return r0, p0, k0

    @pytest.mark.parametrize('ba',
                             [([0, 0], [1, 6, 8]),
                              (0, 1)
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_residue_degenerate(self, xp, scp, ba):
        # Several tests for zero numerator and denominator.
        ba = map(xp.asarray, ba)
        r, p, k = scp.signal.residue(*ba)
        return r, p, k

    @pytest.mark.parametrize('ba',
                             [([1, 6, 6, 2], [1, -(2 + 1j), (1 + 2j), -1j]),
                              ([1, 2, 1], [1, -1, 0.3561]),
                                 ([1, -1], [1, -5, 6]),
                                 ([2, 3, 4], [1, 3, 3, 1]),
                                 ([1, -10, -4, 4], [2, -2, -4]),
                                 ([18], [18, 3, -4, -1]),
                                 ([2, 3], np.polymul([1, -1/2], [1, 1/4])),
                                 ([1, -2, 1], [1, -1]),
                                 (1, [1, -1j]),
                                 (1, [1, -1, 0.25]),
                                 ([1, 6, 2], [1, -2, 1]),
                                 ([6, 2], [1, -2, 1]),
                                 ([1, 6, 6, 2], [1, -2, 1]),
                                 ([1, 0, 1], [1, 0, 0, 0, 0, -1])
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_residuez_general(self, xp, scp, ba):
        ba = map(xp.asarray, ba)
        r, p, k = scp.signal.residuez(*ba)
        return r, p, k

    @pytest.mark.parametrize('ba',
                             [([5, 3, -2, 7], [-4, 0, 8, 3]),
                              ([5, 3, -2, 7, 0], [-4, 0, 8, 3]),
                                 ([5, 3, -2, 7], [-4, 0, 8, 3, 0]),
                                 ([5, 3, -2, 7, 0, 0], [-4, 0, 8, 3, 0, 0, 0])
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_residuez_trailing_zeros(self, xp, scp, ba):
        # Trailing zeros in numerator or denominator must not affect the
        # answer.
        ba = map(xp.asarray, ba)
        r, p, k = scp.signal.residuez(*ba)
        return r, p, k

    @pytest.mark.parametrize('ba',
                             [([0, 0], [1, 6, 8]),
                              (0, 1)
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_residuez_degenerate(self, xp, scp, ba):
        ba = map(xp.asarray, ba)
        r, p, k = scp.signal.residuez(*ba)
        return r, p, k

    @pytest.mark.parametrize('rtype', rtypes)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_inverse_unique_roots_different_rtypes(self, xp, scp, rtype):
        # This test was inspired by github issue 2496.
        r = xp.asarray([3 / 10, -1 / 6, -2 / 15])
        p = xp.asarray([0, -2, -5])
        k = xp.asarray([])

        # With the default tolerance, the rtype does not matter
        # for this example.
        b, a = scp.signal.invres(r, p, k, rtype=rtype)
        return b, a

    @pytest.mark.parametrize('rtype', rtypes)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_inverse_unique_roots_different_rtypes_2(self, xp, scp, rtype):
        # This test was inspired by github issue 2496.
        r = xp.asarray([3 / 10, -1 / 6, -2 / 15])
        p = xp.asarray([0, -2, -5])
        k = xp.asarray([])

        b, a = scp.signal.invresz(r, p, k, rtype=rtype)
        return b, a

    @pytest.mark.parametrize('rtype', rtypes)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_inverse_repeated_roots_different_rtypes(self, xp, scp, rtype):
        r = xp.asarray([3 / 20, -7 / 36, -1 / 6, 2 / 45])
        p = xp.asarray([0, -2, -2, -5])
        k = xp.asarray([])
        b, a = scp.signal.invres(r, p, k, rtype=rtype)
        return b, a

    @pytest.mark.parametrize('rtype', rtypes)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_inverse_repeated_roots_different_rtypes_2(self, xp, scp, rtype):
        r = xp.asarray([3 / 20, -7 / 36, -1 / 6, 2 / 45])
        p = xp.asarray([0, -2, -2, -5])
        k = xp.asarray([])
        b, a = scp.signal.invresz(r, p, k, rtype=rtype)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_invresz_one_coefficient_bug(self, xp, scp):
        # Regression test for issue in gh-4646.
        r = xp.asarray([1])
        p = xp.asarray([2])
        k = xp.asarray([0])
        b, a = scp.signal.invresz(r, p, k)
        return b, a

    @pytest.mark.parametrize('rpk',
                             [([1], [1], []),
                              ([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], []),
                                 ([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3]),
                                 ([-1, 2, 1j, 3 - 1j, 4, -2],
                                  [-1, 2 - 1j, 2 - 1j, 3, 3, 3], []),
                                 ([-1, 1j], [1, 1], [1, 2])
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_invres(self, xp, scp, rpk):
        r, p, k = map(xp.asarray, rpk)
        b, a = scp.signal.invres(r, p, k)
        return b, a

    @pytest.mark.parametrize('rpk',
                             [([1], [1], []),
                              ([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], []),
                                 ([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3]),
                                 ([-1, 2, 1j, 3 - 1j, 4, -2],
                                  [-1, 2 - 1j, 2 - 1j, 3, 3, 3], []),
                                 ([-1, 1j], [1, 1], [1, 2])
                              ])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_invresz(self, xp, scp, rpk):
        r, p, k = map(xp.asarray, rpk)
        b, a = scp.signal.invresz(r, p, k)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_inverse_scalar_arguments(self, xp, scp):
        b, a = scp.signal.invres(1, 1, 1)
        return b, a

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-15)
    def test_inverse_scalar_arguments_2(self, xp, scp):
        b, a = scp.signal.invresz(1, 1, 1)
        return b, a
