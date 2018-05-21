import atexit
import binascii
import functools
import operator
import os
import time
import warnings

import numpy
import six

import cupy
from cupy import core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device


_beta_kernel = None
_binomial_kernel = None
_chisquare_kernel = None
_gumbel_kernel = None
_laplace_kernel = None
_poisson_kernel = None
_standard_gamma_kernel = None
_standard_t_kernel = None


loggam_difinition = '''
/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static __device__ double loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}
'''


rk_state_difinition = '''
#define RK_STATE_LEN 624

typedef struct rk_state_
{
    unsigned long key[RK_STATE_LEN];
    int pos;
    int has_gauss; /* !=0: gauss contains a gaussian deviate */
    double gauss;

    /* The rk_state structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */

    int has_binomial; /* !=0: following parameters initialized for
                              binomial */
    double psave;
    long nsave;
    double r;
    double q;
    double fm;
    long m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;

}
rk_state;
'''


rk_seed_definition = '''
__device__ void
rk_seed(unsigned long seed, rk_state *state)
{
    int pos;
    seed &= 0xffffffffUL;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < RK_STATE_LEN; pos++) {
        state->key[pos] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
    }
    state->pos = RK_STATE_LEN;
    state->gauss = 0;
    state->has_gauss = 0;
    state->has_binomial = 0;
}
'''


rk_random_definition = '''
/* Magic Mersenne Twister constants */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

/*
 * Slightly optimised reference implementation of the Mersenne Twister
 * Note that regardless of the precision of long, only 32 bit random
 * integers are produced
 */
__device__ unsigned long
rk_random(rk_state *state)
{
    unsigned long y;

    if (state->pos == RK_STATE_LEN) {
        int i;

        for (i = 0; i < N - M; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i] = state->key[i+M] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        for (; i < N - 1; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i]
                = state->key[i+(M-N)] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
        state->key[N - 1]
            = state->key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

        state->pos = 0;
    }
    y = state->key[state->pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
'''


rk_double_definition = '''
__device__ double
rk_double(rk_state *state)
{
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    long a = rk_random(state) >> 5, b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
'''


rk_gauss_definition = '''
__device__ double
rk_gauss(rk_state *state)
{
    if (state->has_gauss) {
        const double tmp = state->gauss;
        state->gauss = 0;
        state->has_gauss = 0;
        return tmp;
    }
    else {
        double f, x1, x2, r2;

        do {
            x1 = 2.0*rk_double(state) - 1.0;
            x2 = 2.0*rk_double(state) - 1.0;
            r2 = x1*x1 + x2*x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrt(-2.0*log(r2)/r2);
        /* Keep for next call */
        state->gauss = f*x1;
        state->has_gauss = 1;
        return f*x2;
    }
}
'''


rk_standard_exponential_definition = '''
__device__ double rk_standard_exponential(rk_state *state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - rk_double(state));
}
'''


rk_standard_gamma_definition = '''
__device__ double rk_standard_gamma(rk_state *state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return rk_standard_exponential(state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = rk_double(state);
            V = rk_standard_exponential(state);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1./shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;)
        {
            do
            {
                X = rk_gauss(state);
                V = 1.0 + c*X;
            } while (V <= 0.0);

            V = V*V*V;
            U = rk_double(state);
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}
'''


rk_beta_definition = '''
__device__ double rk_beta(rk_state *state, double a, double b)
{
    double Ga, Gb;

    if ((a <= 1.0) && (b <= 1.0))
    {
        double U, V, X, Y;
        /* Use Johnk's algorithm */

        while (1)
        {
            U = rk_double(state);
            V = rk_double(state);
            X = pow(U, 1.0/a);
            Y = pow(V, 1.0/b);

            if ((X + Y) <= 1.0)
            {
                if (X +Y > 0)
                {
                    return X / (X + Y);
                }
                else
                {
                    double logX = log(U) / a;
                    double logY = log(V) / b;
                    double logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;

                    return exp(logX - log(exp(logX) + exp(logY)));
                }
            }
        }
    }
    else
    {
        Ga = rk_standard_gamma(state, a);
        Gb = rk_standard_gamma(state, b);
        return Ga/(Ga + Gb);
    }
}
'''

rk_chisquare_definition = '''
__device__ double rk_chisquare(rk_state *state, double df)
{
    return 2.0*rk_standard_gamma(state, df/2.0);
}
'''

rk_standard_t_definition = '''
__device__ double rk_standard_t(rk_state *state, double df)
{
    return sqrt(df/2)*rk_gauss(state)/sqrt(rk_standard_gamma(state, df/2));
}
'''


rk_poisson_mult_definition = '''
__device__ long rk_poisson_mult(rk_state *state, double lam)
{
    long X;
    double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1)
    {
        U = rk_double(state);
        prod *= U;
        if (prod > enlam)
        {
            X += 1;
        }
        else
        {
            return X;
        }
    }
}
'''


rk_poisson_ptrs_definition = '''
/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
__device__ long rk_poisson_ptrs(rk_state *state, double lam)
{
    long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
        U = rk_double(state) - 0.5;
        V = rk_double(state);
        us = 0.5 - fabs(U);
        k = (long)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1)))
        {
            return k;
        }


    }

}
'''


rk_poisson_definition = '''
__device__ long rk_poisson(rk_state *state, double lam)
{
    if (lam >= 10)
    {
        return rk_poisson_ptrs(state, lam);
    }
    else if (lam == 0)
    {
        return 0;
    }
    else
    {
        return rk_poisson_mult(state, lam);
    }
}
'''


def _get_beta_kernel():
    global _beta_kernel
    if _beta_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_beta_definition]
        _beta_kernel = core.ElementwiseKernel(
            'T a, T b, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_beta(&internal_state, a, b);
            ''',
            'beta_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _beta_kernel


# TODO(YoshikawaMasashi): implementation of BTPE same as numpy
def _get_binomial_kernel():
    global _binomial_kernel
    if _binomial_kernel is None:
        _binomial_kernel = core.ElementwiseKernel(
            'T x, T n, T p', 'T y',
            '''
            y = 0.;
            T px = exp(n * log(1-p));
            while(x > px){
                y += 1.;
                x -= px;
                px = ((n-y+1) * p * px)/(y*(1-p));
            }
            ''',
            'binomial_kernel'
        )
    return _binomial_kernel


def _get_chisquare_kernel():
    global _chisquare_kernel
    if _chisquare_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_chisquare_definition]
        _chisquare_kernel = core.ElementwiseKernel(
            'T df, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_chisquare(&internal_state, df);
            ''',
            'beta_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _chisquare_kernel


def _get_gumbel_kernel():
    global _gumbel_kernel
    if _gumbel_kernel is None:
        _gumbel_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = loc - log(-log(1 - x)) * scale',
            'gumbel_kernel'
        )
    return _gumbel_kernel


def _get_laplace_kernel():
    global _laplace_kernel
    if _laplace_kernel is None:
        _laplace_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = (x < 0.5)? loc + scale * log(x + x):'
            ' loc - scale * log(2.0 - x - x)',
            'laplace_kernel'
        )
    return _laplace_kernel


def _get_poisson_kernel():
    global _poisson_kernel
    if _poisson_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, loggam_difinition,
             rk_poisson_mult_definition, rk_poisson_ptrs_definition,
             rk_poisson_definition]
        _poisson_kernel = core.ElementwiseKernel(
            'T lam, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_poisson(&internal_state, lam);
            ''',
            'poisson_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _poisson_kernel


def _get_standard_gamma_kernel():
    global _standard_gamma_kernel
    if _standard_gamma_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition]
        _standard_gamma_kernel = core.ElementwiseKernel(
            'T shape, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_standard_gamma(&internal_state, shape);
            ''',
            'standard_gamma_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _standard_gamma_kernel


def _get_standard_t_kernel():
    global _standard_t_kernel
    if _standard_t_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_standard_t_definition]
        _standard_t_kernel = core.ElementwiseKernel(
            'T df, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_standard_t(&internal_state, df);
            ''',
            'standard_t_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _standard_t_kernel


class RandomState(object):

    """Portable container of a pseudo-random number generator.

    An instance of this class holds the state of a random number generator. The
    state is available only on the device which has been current at the
    initialization of the instance.

    Functions of :mod:`cupy.random` use global instances of this class.
    Different instances are used for different devices. The global state for
    the current device can be obtained by the
    :func:`cupy.random.get_random_state` function.

    Args:
        seed (None or int): Seed of the random number generator. See the
            :meth:`~cupy.random.RandomState.seed` method for detail.
        method (int): Method of the random number generator. Following values
            are available::

               cupy.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT
               cupy.cuda.curand.CURAND_RNG_XORWOW
               cupy.cuda.curand.CURAND_RNG_MRG32K3A
               cupy.cuda.curand.CURAND_RNG_MTGP32
               cupy.cuda.curand.CURAND_RNG_MT19937
               cupy.cuda.curand.CURAND_RNG_PHILOX4_32_10

    """

    def __init__(self, seed=None, method=curand.CURAND_RNG_PSEUDO_DEFAULT):
        self._generator = curand.createGenerator(method)
        self.seed(seed)

    def __del__(self):
        # When createGenerator raises an error, _generator is not initialized
        if hasattr(self, '_generator'):
            curand.destroyGenerator(self._generator)

    def _generate_normal(self, func, size, dtype, *args):
        # curand functions below don't support odd size.
        # * curand.generateNormal
        # * curand.generateNormalDouble
        # * curand.generateLogNormal
        # * curand.generateLogNormalDouble
        size = core.get_size(size)
        element_size = six.moves.reduce(operator.mul, size, 1)
        if element_size % 2 == 0:
            out = cupy.empty(size, dtype=dtype)
            func(self._generator, out.data.ptr, out.size, *args)
            return out
        else:
            out = cupy.empty((element_size + 1,), dtype=dtype)
            func(self._generator, out.data.ptr, out.size, *args)
            return out[:element_size].reshape(size)

    # NumPy compatible functions

    def lognormal(self, mean=0.0, sigma=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a log normal distribution.

        .. seealso::
            :func:`cupy.random.lognormal` for full documentation,
            :meth:`numpy.random.RandomState.lognormal`

        """
        dtype = _check_and_get_dtype(dtype)
        if dtype.char == 'f':
            func = curand.generateLogNormal
        else:
            func = curand.generateLogNormalDouble
        return self._generate_normal(func, size, dtype, mean, sigma)

    def multivariate_normal(self, mean, cov, size=None, check_valid='warn',
                            tol=1e-8, dtype=float):
        """Returns an array of multivariate nomally distributed samples.

        .. seealso::
            :func:`cupy.random.multivariate_normal` for full documentation,
            :meth:`numpy.random.RandomState.multivariate_normal`

        """
        mean = cupy.array(mean, dtype=dtype)
        cov = cupy.array(cov, dtype=dtype)
        if size is None:
            shape = []
        elif isinstance(size, (int, cupy.integer)):
            shape = [size]
        else:
            shape = size

        if len(mean.shape) != 1:
            raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])

        x = self.standard_normal(size=final_shape, dtype=dtype)
        x = x.reshape(-1, mean.shape[0])

        (u, s, v) = cupy.linalg.svd(cov)

        if check_valid != 'ignore':
            if check_valid != 'warn' and check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")

            a = cupy.dot(v.T * s, v)
            b = cov
            psd = cupy.all(cupy.abs(a-b) <= tol*(1+cupy.abs(b)))
            if not psd:
                if check_valid == 'warn':
                    warnings.warn(
                        "covariance is not symmetric positive-semidefinite.",
                        RuntimeWarning)
                else:
                    raise ValueError(
                        "covariance is not symmetric positive-semidefinite.")

        x = cupy.dot(x, cupy.sqrt(s)[:, None] * v)
        x += mean
        x.shape = tuple(final_shape)
        return x

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of normally distributed samples.

        .. seealso::
            :func:`cupy.random.normal` for full documentation,
            :meth:`numpy.random.RandomState.normal`

        """
        dtype = _check_and_get_dtype(dtype)
        if dtype.char == 'f':
            func = curand.generateNormal
        else:
            func = curand.generateNormalDouble
        return self._generate_normal(func, size, dtype, loc, scale)

    def rand(self, *size, **kwarg):
        """Returns uniform random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`cupy.random.rand` for full documentation,
            :meth:`numpy.random.RandomState.rand`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('rand() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.random_sample(size=size, dtype=dtype)

    def randn(self, *size, **kwarg):
        """Returns an array of standard normal random values.

        .. seealso::
            :func:`cupy.random.randn` for full documentation,
            :meth:`numpy.random.RandomState.randn`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('randn() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.normal(size=size, dtype=dtype)

    _1m_kernel = core.ElementwiseKernel(
        '', 'T x', 'x = 1 - x', 'cupy_random_1_minus_x')

    def random_sample(self, size=None, dtype=float):
        """Returns an array of random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`cupy.random.random_sample` for full documentation,
            :meth:`numpy.random.RandomState.random_sample`

        """
        dtype = _check_and_get_dtype(dtype)
        out = cupy.empty(size, dtype=dtype)
        if dtype.char == 'f':
            func = curand.generateUniform
        else:
            func = curand.generateUniformDouble
        func(self._generator, out.data.ptr, out.size)
        RandomState._1m_kernel(out)
        return out

    def interval(self, mx, size):
        """Generate multiple integers independently sampled uniformly from ``[0, mx]``.

        Args:
            mx (int): Upper bound of the interval
            size (None or int or tuple): Shape of the array or the scalar
                returned.
        Returns:
            int or cupy.ndarray: If ``None``, an :class:`cupy.ndarray` with
            shape ``()`` is returned.
            If ``int``, 1-D array of length size is returned.
            If ``tuple``, multi-dimensional array with shape
            ``size`` is returned.
            Currently, only 32 bit integers can be sampled.
            If 0 :math:`\\leq` ``mx`` :math:`\\leq` 0x7fffffff,
            a ``numpy.int32`` array is returned.
            If 0x80000000 :math:`\\leq` ``mx`` :math:`\\leq` 0xffffffff,
            a ``numpy.uint32`` array is returned.
        """
        if size is None:
            return self.interval(mx, 1).reshape(())
        elif isinstance(size, int):
            size = (size, )

        if mx == 0:
            return cupy.zeros(size, dtype=numpy.int32)

        if mx < 0:
            raise ValueError(
                'mx must be non-negative (actual: {})'.format(mx))
        elif mx <= 0x7fffffff:
            dtype = numpy.int32
        elif mx <= 0xffffffff:
            dtype = numpy.uint32
        else:
            raise ValueError(
                'mx must be within uint32 range (actual: {})'.format(mx))

        mask = (1 << mx.bit_length()) - 1
        mask = cupy.array(mask, dtype=dtype)

        n = functools.reduce(operator.mul, size, 1)

        sample = cupy.empty((n,), dtype=dtype)
        n_rem = n  # The number of remaining elements to sample
        ret = None
        while n_rem > 0:
            curand.generate(
                self._generator, sample.data.ptr, sample.size)
            # Drop the samples that exceed the upper limit
            sample &= mask
            success = sample <= mx

            if ret is None:
                # If the sampling has finished in the first iteration,
                # just return the sample.
                if success.all():
                    n_rem = 0
                    ret = sample
                    break

                # Allocate the return array.
                ret = cupy.empty((n,), dtype=dtype)

            n_succ = min(n_rem, int(success.sum()))
            ret[n - n_rem:n - n_rem + n_succ] = sample[success][:n_succ]
            n_rem -= n_succ

        assert n_rem == 0
        return ret.reshape(size)

    def seed(self, seed=None):
        """Resets the state of the random number generator with a seed.

        .. seealso::
            :func:`cupy.random.seed` for full documentation,
            :meth:`numpy.random.RandomState.seed`

        """
        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.asarray(seed).astype(numpy.uint64, casting='safe')

        curand.setPseudoRandomGeneratorSeed(self._generator, seed)
        curand.setGeneratorOffset(self._generator, 0)

        self.rk_seed = numpy.uint32(seed)

    def standard_normal(self, size=None, dtype=float):
        """Returns samples drawn from the standard normal distribution.

        .. seealso::
            :func:`cupy.random.standard_normal` for full documentation,
            :meth:`numpy.random.RandomState.standard_normal`

        """
        return self.normal(size=size, dtype=dtype)

    def tomaxint(self, size=None):
        """Draws integers between 0 and max integer inclusive.

        Args:
            size (int or tuple of ints): Output shape.

        Returns:
            cupy.ndarray: Drawn samples.

        .. seealso::
            :meth:`numpy.random.RandomState.tomaxint`

        """
        if size is None:
            size = ()
        sample = cupy.empty(size, dtype=cupy.int_)
        # cupy.random only uses int32 random generator
        size_in_int = sample.dtype.itemsize // 4
        curand.generate(
            self._generator, sample.data.ptr, sample.size * size_in_int)

        # Disable sign bit
        sample &= cupy.iinfo(cupy.int_).max
        return sample

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float):
        """Returns an array of uniformly-distributed samples over an interval.

        .. seealso::
            :func:`cupy.random.uniform` for full documentation,
            :meth:`numpy.random.RandomState.uniform`

        """
        dtype = numpy.dtype(dtype)
        rand = self.random_sample(size=size, dtype=dtype)
        return dtype.type(low) + rand * dtype.type(high - low)

    def choice(self, a, size=None, replace=True, p=None):
        """Returns an array of random values from a given 1-D array.

        .. seealso::
            :func:`cupy.random.choice` for full document,
            :func:`numpy.random.choice`

        """
        if a is None:
            raise ValueError('a must be 1-dimensional or an integer')
        if isinstance(a, cupy.ndarray) and a.ndim == 0:
            raise NotImplementedError
        if isinstance(a, six.integer_types):
            a_size = a
            if a_size <= 0:
                raise ValueError('a must be greater than 0')
        else:
            a = cupy.array(a, copy=False)
            if a.ndim != 1:
                raise ValueError('a must be 1-dimensional or an integer')
            else:
                a_size = len(a)
                if a_size == 0:
                    raise ValueError('a must be non-empty')

        if p is not None:
            p = cupy.array(p)
            if p.ndim != 1:
                raise ValueError('p must be 1-dimensional')
            if len(p) != a_size:
                raise ValueError('a and p must have same size')
            if not (p >= 0).all():
                raise ValueError('probabilities are not non-negative')
            p_sum = cupy.sum(p).get()
            if not numpy.allclose(p_sum, 1):
                raise ValueError('probabilities do not sum to 1')

        if size is None:
            raise NotImplementedError
        shape = size
        size = numpy.prod(shape)

        if not replace and p is None:
            if a_size < size:
                raise ValueError(
                    'Cannot take a larger sample than population when '
                    '\'replace=False\'')
            if isinstance(a, six.integer_types):
                indices = cupy.arange(a, dtype='l')
            else:
                indices = a.copy()
            self.shuffle(indices)
            return indices[:size].reshape(shape)

        if not replace:
            raise NotImplementedError

        if p is not None:
            p = cupy.broadcast_to(p, (size, a_size))
            index = cupy.argmax(cupy.log(p) +
                                self.gumbel(size=(size, a_size)),
                                axis=1)
            if not isinstance(shape, six.integer_types):
                index = cupy.reshape(index, shape)
        else:
            index = self.randint(0, a_size, size=shape)
            # Align the dtype with NumPy
            index = index.astype(cupy.int64, copy=False)

        if isinstance(a, six.integer_types):
            return index

        if index.ndim == 0:
            return cupy.array(a[index], dtype=a.dtype)

        return a[index]

    def shuffle(self, a):
        """Returns a shuffled array.

        .. seealso::
            :func:`cupy.random.shuffle` for full document,
            :func:`numpy.random.shuffle`

        """
        if not isinstance(a, cupy.ndarray):
            raise TypeError('The array must be cupy.ndarray')

        if a.ndim == 0:
            raise TypeError('An array whose ndim is 0 is not supported')

        a[:] = a[self.permutation(len(a))]

    def permutation(self, num):
        """Returns a permuted range."""
        if not isinstance(num, six.integer_types):
            raise TypeError('The data type of argument "num" must be integer')

        sample = cupy.empty((num), dtype=numpy.int32)
        curand.generate(self._generator, sample.data.ptr, num)
        if 128 < num <= 32 * 1024 * 1024:
            array = cupy.arange(num, dtype=numpy.int32)
            # apply sort of cache blocking
            block_size = 1 * 1024 * 1024
            # The block size above is a value determined from the L2 cache size
            # of GP100 (L2 cache size / size of int = 4MB / 4B = 1M). It may be
            # better to change the value base on the L2 cache size of the GPU
            # you use.
            # When num > block_size, cupy kernel: _cupy_permutation is to be
            # launched multiple times. However, it is observed that performance
            # will be degraded if the launch count is too many. Therefore,
            # the block size is adjusted so that launch count will not exceed
            # twelve Note that this twelve is the value determined from
            # measurement on GP100.
            while num // block_size > 12:
                block_size *= 2
            for j_start in range(0, num, block_size):
                j_end = j_start + block_size
                _cupy_permutation()(array, sample, j_start, j_end, size=num)
        else:
            # When num > 32M, argsort is used, because it is faster than
            # custom kernel. See https://github.com/cupy/cupy/pull/603.
            array = cupy.argsort(sample)
        return array

    def binomial(self, n, p, size=None, dtype=float):
        """Returns an array of samples drawn from a Binomial distribution.

        .. seealso::
            :func:`cupy.random.binomial` for full documentation,
            :meth:`numpy.random.RandomState.binomial`
        """
        x = self.uniform(size=size, dtype=dtype)
        y = cupy.zeros_like(x, dtype=dtype)
        _get_binomial_kernel()(x, n, p, y)
        return y

    def beta(self, a, b, size=None, dtype=float):
        """Returns an array of samples drawn from a Beta distribution.

        .. seealso::
            :func:`cupy.random.beta` for full documentation,
            :meth:`numpy.random.RandomState.beta`
        """
        y = cupy.zeros(shape=size, dtype=dtype)
        _get_beta_kernel()(a, b, self.rk_seed, y)
        if size is None:
            self.rk_seed += 1
        else:
            self.rk_seed += numpy.prod(size)
        return y

    def chisquare(self, df, size=None, dtype=float):
        """Returns an array of samples drawn from a Chisquare distribution.

        .. seealso::
            :func:`cupy.random.chisquare` for full documentation,
            :meth:`numpy.random.RandomState.chisquare`
        """
        y = cupy.zeros(shape=size, dtype=dtype)
        _get_chisquare_kernel()(df, self.rk_seed, y)
        if size is None:
            self.rk_seed += 1
        else:
            self.rk_seed += numpy.prod(size)
        return y

    def dirichlet(self, alpha, size=None, dtype=float):
        """Returns an array of samples drawn from a Dirichlet distribution.

        .. seealso::
            :func:`cupy.random.dirichlet` for full documentation,
            :meth:`numpy.random.RandomState.dirichlet`
        """
        y = cupy.zeros(shape=size, dtype=dtype)
        _get_standard_gamma_kernel()(alpha, self.rk_seed, y)
        y /= cupy.expand_dims(y.sum(axis=-1), axis=-1)
        if size is None:
            self.rk_seed += 1
        else:
            self.rk_seed += numpy.prod(size)
        return y

    def gamma(self, shape, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a Gamma distribution.

        .. seealso::
            :func:`cupy.random.gamma` for full documentation,
            :meth:`numpy.random.RandomState.gamma`
        """
        y = cupy.zeros(shape=size, dtype=dtype)
        _get_standard_gamma_kernel()(shape, self.rk_seed, y)
        y *= scale
        if size is None:
            self.rk_seed += 1
        else:
            self.rk_seed += numpy.prod(size)
        return y

    def gumbel(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a Gumbel distribution.

        .. seealso::
            :func:`cupy.random.gumbel` for full documentation,
            :meth:`numpy.random.RandomState.gumbel`
        """
        x = self.uniform(size=size, dtype=dtype)
        # We use `1 - x` as input of `log` method to prevent overflow.
        # It obeys numpy implementation.
        _get_gumbel_kernel()(x, loc, scale, x)
        return x

    def laplace(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a Laplace distribution.

        .. seealso::
            :func:`cupy.random.laplace` for full documentation,
            :meth:`numpy.random.RandomState.laplace`
        """
        x = self.uniform(size=size, dtype=dtype)
        _get_laplace_kernel()(x, loc, scale, x)
        return x

    def poisson(self, lam=1.0, size=None, dtype=int):
        """Returns an array of samples drawn from a Poisson distribution.

        .. seealso::
            :func:`cupy.random.poisson` for full documentation,
            :meth:`numpy.random.RandomState.poisson`
        """
        y = cupy.zeros(shape=size, dtype=dtype)
        _get_poisson_kernel()(lam, self.rk_seed, y)
        if size is None:
            self.rk_seed += 1
        else:
            self.rk_seed += numpy.prod(size)
        return y

    def standard_t(self, df, size=None, dtype=float):
        """Returns an array of samples drawn from a Standard Student’s t distribution.

        .. seealso::
            :func:`cupy.random.standard_t` for full documentation,
            :meth:`numpy.random.RandomState.standard_t`
        """
        y = cupy.zeros(shape=size, dtype=dtype)
        _get_standard_t_kernel()(df, self.rk_seed, y)
        if size is None:
            self.rk_seed += 1
        else:
            self.rk_seed += numpy.prod(size)
        return y

    def randint(self, low, high=None, size=None, dtype='l'):
        """Returns a scalar or an array of integer values over ``[low, high)``.

        .. seealso::
            :func:`cupy.random.randint` for full documentation,
            :meth:`numpy.random.RandomState.randint`
        """
        if high is None:
            lo = 0
            hi = low
        else:
            lo = low
            hi = high

        if lo >= hi:
            raise ValueError('low >= high')
        if lo < cupy.iinfo(dtype).min:
            raise ValueError(
                'low is out of bounds for {}'.format(cupy.dtype(dtype).name))
        if hi > cupy.iinfo(dtype).max + 1:
            raise ValueError(
                'high is out of bounds for {}'.format(cupy.dtype(dtype).name))

        diff = hi - lo - 1
        if diff > cupy.iinfo(cupy.int32).max - cupy.iinfo(cupy.int32).min + 1:
            raise NotImplementedError(
                'Sampling from a range whose extent is larger than int32 '
                'range is currently not supported')
        x = self.interval(diff, size).astype(dtype, copy=False)
        cupy.add(x, lo, out=x)
        return x


def _cupy_permutation():
    return core.ElementwiseKernel(
        'raw int32 array, raw int32 sample, int32 j_start, int32 _j_end',
        '',
        '''
            const int invalid = -1;
            const int num = _ind.size();
            int j = (sample[i] & 0x7fffffff) % num;
            int j_end = _j_end;
            if (j_end > num) j_end = num;
            if (j == i || j < j_start || j >= j_end) continue;

            // If a thread fails to do data swaping once, it changes j
            // value using j_offset below and try data swaping again.
            // This process is repeated until data swapping is succeeded.
            // The j_offset is determined from the initial j
            // (random number assigned to each thread) and the initial
            // offset between j and i (ID of each thread).
            // If a given number sequence in sample is really random,
            // this j-update would not be necessary. This is work-around
            // mainly to avoid potential eternal conflict when sample has
            // rather synthetic number sequence.
            int j_offset = ((2*j - i + num) % (num - 1)) + 1;

            // A thread gives up to do data swapping if loop count exceed
            // a threathod determined below. This is kind of safety
            // mechanism to escape the eternal race condition, though I
            // believe it never happens.
            int loops = 256;

            bool do_next = true;
            while (do_next && loops > 0) {
                // try to swap the contents of array[i] and array[j]
                if (i != j) {
                    int val_j = atomicExch(&array[j], invalid);
                    if (val_j != invalid) {
                        int val_i = atomicExch(&array[i], invalid);
                        if (val_i != invalid) {
                            array[i] = val_j;
                            array[j] = val_i;
                            do_next = false;
                            // done
                        }
                        else {
                            // restore array[j]
                            array[j] = val_j;
                        }
                    }
                }
                j = (j + j_offset) % num;
                loops--;
            }
        ''',
        'cupy_permutation',
    )


def seed(seed=None):
    """Resets the state of the random number generator with a seed.

    This function resets the state of the global random number generator for
    the current device. Be careful that generators for other devices are not
    affected.

    Args:
        seed (None or int): Seed for the random number generator. If ``None``,
            it uses :func:`os.urandom` if available or :func:`time.clock`
            otherwise. Note that this function does not support seeding by an
            integer array.

    """
    get_random_state().seed(seed)


# CuPy specific functions

_random_states = {}


@atexit.register
def reset_states():
    global _random_states
    _random_states = {}


def get_random_state():
    """Gets the state of the random number generator for the current device.

    If the state for the current device is not created yet, this function
    creates a new one, initializes it, and stores it as the state for the
    current device.

    Returns:
        RandomState: The state of the random number generator for the
        device.

    """
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        seed = os.getenv('CUPY_SEED')
        if seed is None:
            seed = os.getenv('CHAINER_SEED')
        if seed is not None:
            seed = numpy.uint64(int(seed))
        rs = RandomState(seed)
        rs = _random_states.setdefault(dev.id, rs)
    return rs


def set_random_state(rs):
    """Sets the state of the random number generator for the current device.

    Args:
        state(RandomState): Random state to set for the current device.
    """
    if not isinstance(rs, RandomState):
        raise TypeError(
            'Random state must be an instance of RandomState. '
            'Actual: {}'.format(type(rs)))
    _random_states[device.get_device_id()] = rs


def _check_and_get_dtype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype.char not in ('f', 'd'):
        raise TypeError('cupy.random only supports float32 and float64')
    return dtype
