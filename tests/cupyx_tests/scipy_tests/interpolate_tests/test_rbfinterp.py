import pickle
import pytest
import warnings

import numpy as _np
from numpy.linalg import LinAlgError
import cupy as cp
import cupyx
from cupy import testing
import cupyx.scipy.interpolate  # NOQA


try:
    from scipy import interpolate  # NOQA
except ImportError:
    pass

try:
    from scipy.stats.qmc import Halton
except ImportError:
    # qmc.Halton is only available in SciPy >= 1.70
    pass


from cupyx.scipy.interpolate._rbfinterp import (
    _AVAILABLE, _SCALE_INVARIANT, _NAME_TO_MIN_DEGREE, NAME_TO_FUNC,
    _monomial_powers, polynomial_matrix, kernel_matrix)


def _kernel_matrix(x, kernel):
    """Return RBFs, with centers at `x`, evaluated at `x`."""
    out = cp.empty((x.shape[0], x.shape[0]), dtype=float)
    kernel_func = NAME_TO_FUNC[kernel]
    kernel_matrix(x, kernel_func, out)
    return out


def _polynomial_matrix(x, powers):
    """Return monomials, with exponents from `powers`, evaluated at `x`."""
    out = cp.empty((x.shape[0], powers.shape[0]), dtype=float)
    polynomial_matrix(x, powers, out)
    return out


def _vandermonde(x, degree):
    # Returns a matrix of monomials that span polynomials with the specified
    # degree evaluated at x.
    powers = _monomial_powers(x.shape[1], degree)
    return _polynomial_matrix(x, powers)


def _1d_test_function(x, xp):
    # Test function used in Wahba's "Spline Models for Observational Data".
    # domain ~= (0, 3), range ~= (-1.0, 0.2)
    x = x[:, 0]
    y = 4.26*(xp.exp(-x) - 4*xp.exp(-2*x) + 3*xp.exp(-3*x))
    return y


def _2d_test_function(x, xp):
    # Franke's test function.
    # domain ~= (0, 1) X (0, 1), range ~= (0.0, 1.2)
    x1, x2 = x[:, 0], x[:, 1]
    term1 = 0.75 * xp.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
    term2 = 0.75 * xp.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
    term3 = 0.5 * xp.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
    term4 = -0.2 * xp.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    y = term1 + term2 + term3 + term4
    return y


def _is_conditionally_positive_definite(kernel, m, xp, scp):
    # Tests whether the kernel is conditionally positive definite of order m.
    # See chapter 7 of Fasshauer's "Meshfree Approximation Methods with
    # MATLAB".
    nx = 10
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        # Generate sample points with a Halton sequence to avoid samples that
        # are too close to eachother, which can make the matrix singular.
        seq = Halton(ndim, scramble=False, seed=_np.random.RandomState())
        for _ in range(ntests):
            x = xp.asarray(2*seq.random(nx)) - 1
            A = _kernel_matrix(x, kernel)
            P = _vandermonde(x, m - 1)
            Q, R = cp.linalg.qr(P, mode='complete')
            # Q2 forms a basis spanning the space where P.T.dot(x) = 0. Project
            # A onto this space, and then see if it is positive definite using
            # the Cholesky decomposition. If not, then the kernel is not c.p.d.
            # of order m.
            Q2 = Q[:, P.shape[1]:]
            B = Q2.T.dot(A).dot(Q2)
            try:
                cp.linalg.cholesky(B)
            except cp.linalg.LinAlgError:
                return False

    return True


# Sorting the parametrize arguments is necessary to avoid a parallelization
# issue described here: https://github.com/pytest-dev/pytest-xdist/issues/432.
@testing.with_requires("scipy>=1.7.0")
@pytest.mark.skip(reason='conditionally posdef: skip for now')
@testing.numpy_cupy_allclose(scipy_name='scp')
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_conditionally_positive_definite(xp, scp, kernel):
    # Test if each kernel in _AVAILABLE is conditionally positive definite of
    # order m, where m comes from _NAME_TO_MIN_DEGREE. This is a necessary
    # condition for the smoothed RBF interpolant to be well-posed in general.
    m = _NAME_TO_MIN_DEGREE.get(kernel, -1) + 1
    assert _is_conditionally_positive_definite(kernel, m, xp, scp)


@testing.with_requires("scipy>=1.7.0")
class _TestRBFInterpolator:
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_1d(self, xp, scp, kernel):
        # Verify that the functions in _SCALE_INVARIANT are insensitive to the
        # shape parameter (when smoothing == 0) in 1d.
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(3*seq.random(50))
        y = _1d_test_function(x, xp)
        xitp = xp.asarray(3*seq.random(50))
        yitp1 = self.build(scp, x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(scp, x, y, epsilon=2.0, kernel=kernel)(xitp)
        return yitp1, yitp2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_2d(self, xp, scp, kernel):
        # Verify that the functions in _SCALE_INVARIANT are insensitive to the
        # shape parameter (when smoothing == 0) in 2d.
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        x = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp)
        xitp = xp.asarray(seq.random(100))
        yitp1 = self.build(scp, x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(scp, x, y, epsilon=2.0, kernel=kernel)(xitp)
        return yitp1, yitp2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_extreme_domains(self, xp, scp, kernel):
        # Make sure the interpolant remains numerically stable for very
        # large/small domains.
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())
        scale = 1e50
        shift = 1e55

        x = xp.asarray(seq.random(100))
        y = _2d_test_function(x, xp)
        xitp = xp.asarray(seq.random(100))

        if kernel in _SCALE_INVARIANT:
            yitp1 = self.build(scp, x, y, kernel=kernel)(xitp)
            yitp2 = self.build(scp,
                               x*scale + shift, y,
                               kernel=kernel
                               )(xitp*scale + shift)
        else:
            yitp1 = self.build(scp, x, y, epsilon=5.0, kernel=kernel)(xitp)
            yitp2 = self.build(scp,
                               x*scale + shift, y,
                               epsilon=5.0/scale,
                               kernel=kernel
                               )(xitp*scale + shift)

        return yitp1, yitp2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_polynomial_reproduction(self, xp, scp):
        # If the observed data comes from a polynomial, then the interpolant
        # should be able to reproduce the polynomial exactly, provided that
        # `degree` is sufficiently high.
        rng = _np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3

        x = xp.asarray(seq.random(50))
        xitp = xp.asarray(seq.random(50))

        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
            Pitp = _vandermonde(cp.asarray(xitp), degree).get()
        else:
            P = _vandermonde(x, degree)
            Pitp = _vandermonde(xitp, degree)

        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        poly_coeffs = xp.asarray(poly_coeffs)

        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        yitp2 = self.build(scp, x, y, degree=degree)(xitp)

        return yitp1, yitp2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_vector_data(self, xp, scp):
        # Make sure interpolating a vector field is the same as interpolating
        # each component separately.
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())

        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))

        y = xp.array([_2d_test_function(x, xp),
                      _2d_test_function(x[:, ::-1], xp)]).T

        yitp1 = self.build(scp, x, y)(xitp)
        yitp2 = self.build(scp, x, y[:, 0])(xitp)
        yitp3 = self.build(scp, x, y[:, 1])(xitp)

        return yitp1, yitp2, yitp3

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_data(self, xp, scp):
        # Interpolating complex input should be the same as interpolating the
        # real and complex components.
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())

        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))

        y = _2d_test_function(x, xp) + 1j*_2d_test_function(x[:, ::-1], xp)

        yitp1 = self.build(scp, x, y)(xitp)
        yitp2 = self.build(scp, x, y.real)(xitp)
        yitp3 = self.build(scp, x, y.imag)(xitp)

        return yitp1, yitp2, yitp3

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_1d(self, xp, scp, kernel):
        # Make sure that each kernel, with its default `degree` and an
        # appropriate `epsilon`, does a good job at interpolation in 1d.
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())

        x = xp.asarray(3*seq.random(50))
        xitp = xp.asarray(3*seq.random(50))

        y = _1d_test_function(x, xp)
        ytrue = _1d_test_function(xitp, xp)
        yitp = self.build(scp, x, y, epsilon=5.0, kernel=kernel)(xitp)

        mse = xp.mean((yitp - ytrue)**2)
        assert mse < 1.0e-4
        return yitp

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_2d(self, xp, scp, kernel):
        # Make sure that each kernel, with its default `degree` and an
        # appropriate `epsilon`, does a good job at interpolation in 2d.
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())

        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))

        y = _2d_test_function(x, xp)
        ytrue = _2d_test_function(xitp, xp)
        yitp = self.build(scp, x, y, epsilon=5.0, kernel=kernel)(xitp)

        mse = xp.mean((yitp - ytrue)**2)
        assert mse < 2.0e-4
        return yitp

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-8)
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_smoothing_misfit(self, xp, scp, kernel):
        # Make sure we can find a smoothing parameter for each kernel that
        # removes a sufficient amount of noise.
        rng = _np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)

        noise = 0.2
        rmse_tol = 0.1
        smoothing_range = 10**xp.linspace(-4, 1, 20)

        x = xp.asarray(3*seq.random(100))
        y = _1d_test_function(
            x, xp) + xp.asarray(rng.normal(0.0, noise, (100,)))
        ytrue = _1d_test_function(x, xp)
        rmse_within_tol = False
        for smoothing in smoothing_range:
            ysmooth = self.build(scp,
                                 x, y,
                                 epsilon=1.0,
                                 smoothing=smoothing,
                                 kernel=kernel)(x)
            rmse = xp.sqrt(xp.mean((ysmooth - ytrue)**2))
            if rmse < rmse_tol:
                rmse_within_tol = True
                break

        assert rmse_within_tol
        return ysmooth

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_array_smoothing(self, xp, scp):
        # Test using an array for `smoothing` to give less weight to a known
        # outlier.
        rng = _np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        degree = 2

        x = xp.asarray(seq.random(50))
        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
        else:
            P = _vandermonde(x, degree)
        poly_coeffs = xp.asarray(rng.normal(0.0, 1.0, P.shape[1]))
        y = P.dot(poly_coeffs)
        y_with_outlier = xp.copy(y)
        y_with_outlier[10] += 1.0
        smoothing = xp.zeros((50,))
        smoothing[10] = 1000.0
        yitp = self.build(scp, x, y_with_outlier, smoothing=smoothing)(x)
        # Should be able to reproduce the uncorrupted data almost exactly.
        return yitp, y

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_inconsistent_x_dimensions_error(self, xp, scp):
        # ValueError should be raised if the observation points and evaluation
        # points have a different number of dimensions.
        y = Halton(2, scramble=False, seed=_np.random.RandomState()).random(10)
        y = xp.asarray(y)
        d = _2d_test_function(y, xp)
        x = Halton(1, scramble=False, seed=_np.random.RandomState()).random(10)
        x = xp.asarray(x)
        self.build(scp, y, d)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_inconsistent_d_length_error(self, xp, scp):
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(1)
        self.build(scp, y, d)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_y_not_2d_error(self, xp, scp):
        y = xp.linspace(0, 1, 5)
        d = xp.zeros(5)
        self.build(scp, y, d)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_inconsistent_smoothing_length_error(self, xp, scp):
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        smoothing = xp.ones(1)
        self.build(scp, y, d, smoothing=smoothing)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_invalid_kernel_name_error(self, xp, scp):
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        self.build(scp, y, d, kernel='test')

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_epsilon_not_specified_error(self, xp, scp, kernel):
        if kernel in _SCALE_INVARIANT:
            return True

        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        self.build(scp, y, d, kernel=kernel)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_x_not_2d_error(self, xp, scp):
        y = xp.linspace(0, 1, 5)[:, None]
        x = xp.linspace(0, 1, 5)
        d = xp.zeros(5)
        self.build(scp, y, d)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_not_enough_observations_error(self, xp, scp):
        y = xp.linspace(0, 1, 1)[:, None]
        d = xp.zeros(1)
        self.build(scp, y, d, kernel='thin_plate_spline')

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=UserWarning)
    @pytest.mark.parametrize('kernel',
                             [kl for kl in _NAME_TO_MIN_DEGREE])
    def test_degree_warning(self, xp, scp, kernel):
        y = xp.linspace(0, 1, 5)[:, None]
        d = xp.zeros(5)
        deg = _NAME_TO_MIN_DEGREE[kernel]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.build(scp, y, d, epsilon=1.0, kernel=kernel, degree=deg-1)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=LinAlgError)
    def test_rank_error(self, xp, scp):
        # An error should be raised when `kernel` is "thin_plate_spline" and
        # observations are 2-D and collinear.
        y = xp.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        d = xp.array([0.0, 0.0, 0.0])
        with cupyx.errstate(linalg='raise'):
            self.build(scp, y, d, kernel='thin_plate_spline')(y)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_single_point(self, xp, scp, dim):
        # Make sure interpolation still works with only one point (in 1, 2, and
        # 3 dimensions).
        y = xp.zeros((1, dim))
        d = xp.ones((1,))
        f = self.build(scp, y, d, kernel='linear')(y)
        return d, f

    def test_pickleable(self):
        # Make sure we can pickle and unpickle the interpolant without any
        # changes in the behavior.
        seq = Halton(1, scramble=False,
                     seed=_np.random.RandomState(2305982309))

        x = cp.asarray(3*seq.random(50))
        xitp = cp.asarray(3*seq.random(50))
        y = _1d_test_function(x, cp)

        interp = cupyx.scipy.interpolate.RBFInterpolator(x, y)

        yitp1 = interp(xitp)
        yitp2 = pickle.loads(pickle.dumps(interp))(xitp)

        testing.assert_array_equal(yitp1, yitp2)


@testing.with_requires("scipy>=1.7.0")
class TestRBFInterpolatorNeighborsNone(_TestRBFInterpolator):
    def build(self, scp, *args, **kwargs):
        return scp.interpolate.RBFInterpolator(*args, **kwargs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_smoothing_limit_1d(self, xp, scp):
        # For large smoothing parameters, the interpolant should approach a
        # least squares fit of a polynomial with the specified degree.
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())

        degree = 3
        smoothing = 1e8

        x = xp.asarray(3*seq.random(50))
        xitp = xp.asarray(3*seq.random(50))

        y = _1d_test_function(x, xp)

        yitp1 = self.build(scp,
                           x, y,
                           degree=degree,
                           smoothing=smoothing
                           )(xitp)

        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
            Pitp = _vandermonde(cp.asarray(xitp), degree).get()
        else:
            P = _vandermonde(x, degree)
            Pitp = _vandermonde(xitp, degree)

        yitp2 = Pitp.dot(xp.linalg.lstsq(P, y, rcond=None)[0])

        return yitp1, yitp2

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_smoothing_limit_2d(self, xp, scp):
        # For large smoothing parameters, the interpolant should approach a
        # least squares fit of a polynomial with the specified degree.
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())

        degree = 3
        smoothing = 1e8

        x = xp.asarray(seq.random(100))
        xitp = xp.asarray(seq.random(100))

        y = _2d_test_function(x, xp)

        yitp1 = self.build(scp,
                           x, y,
                           degree=degree,
                           smoothing=smoothing
                           )(xitp)

        if xp is _np:
            P = _vandermonde(cp.asarray(x), degree).get()
            Pitp = _vandermonde(cp.asarray(xitp), degree).get()
        else:
            P = _vandermonde(x, degree)
            Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(xp.linalg.lstsq(P, y, rcond=None)[0])

        return yitp1, yitp2

    @pytest.mark.slow
    def test_chunking(self):
        # If the observed data comes from a polynomial, then the interpolant
        # should be able to reproduce the polynomial exactly, provided that
        # `degree` is sufficiently high.
        rng = _np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3

        largeN = 1000 + 33
        # this is large to check that chunking of the RBFInterpolator is tested
        x = cp.asarray(seq.random(50))
        xitp = cp.asarray(seq.random(largeN))

        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)

        poly_coeffs = cp.asarray(rng.normal(0.0, 1.0, P.shape[1]))

        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        interp = cupyx.scipy.interpolate.RBFInterpolator(x, y, degree=degree)
        ce_real = interp._chunk_evaluator

        def _chunk_evaluator(*args, **kwargs):
            kwargs.update(memory_budget=100)
            return ce_real(*args, **kwargs)

        # monkeypatch.setattr(interp, '_chunk_evaluator', _chunk_evaluator)
        interp._chunk_evaluator = _chunk_evaluator
        yitp2 = interp(xitp)
        testing.assert_allclose(yitp1, yitp2, atol=1e-8)


"""
# Disable `all neighbors not None` tests : they need KDTree

class TestRBFInterpolatorNeighbors20(_TestRBFInterpolator):
    # RBFInterpolator using 20 nearest neighbors.
    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs, neighbors=20)

    def test_equivalent_to_rbf_interpolator(self):
        seq = Halton(2, scramble=False, seed=_np.random.RandomState())

        x = cp.asarray(seq.random(100))
        xitp = cp.asarray(seq.random(100))

        y = _2d_test_function(x)

        yitp1 = self.build(x, y)(xitp)

        yitp2 = []
        tree = cKDTree(x)
        for xi in xitp:
            _, nbr = tree.query(xi, 20)
            yitp2.append(RBFInterpolator(x[nbr], y[nbr])(xi[None])[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)


class TestRBFInterpolatorNeighborsInf(TestRBFInterpolatorNeighborsNone):
    # RBFInterpolator using neighbors=np.inf. This should give exactly the same
    # results as neighbors=None, but it will be slower.
    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs, neighbors=cp.inf)

    def test_equivalent_to_rbf_interpolator(self):
        seq = Halton(1, scramble=False, seed=_np.random.RandomState())

        x = cp.asarray(3*seq.random(50))
        xitp = cp.asarray(3*seq.random(50))

        y = _1d_test_function(x)
        yitp1 = self.build(x, y)(xitp)
        yitp2 = RBFInterpolator(x, y)(xitp)

        assert_allclose(yitp1, yitp2, atol=1e-8)
"""
