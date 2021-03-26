import atexit
import binascii
import functools
import hashlib
import operator
import os
import time

import numpy
import warnings
from numpy.linalg import LinAlgError

import cupy
from cupy import _core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device
from cupy.random import _kernels
from cupy import _util

import cupyx


_UINT32_MAX = 0xffffffff
_UINT64_MAX = 0xffffffffffffffff


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
               cupy.cuda.curand.CURAND_RNG_PSEUDO_XORWOW
               cupy.cuda.curand.CURAND_RNG_PSEUDO_MRG32K3A
               cupy.cuda.curand.CURAND_RNG_PSEUDO_MTGP32
               cupy.cuda.curand.CURAND_RNG_PSEUDO_MT19937
               cupy.cuda.curand.CURAND_RNG_PSEUDO_PHILOX4_32_10

    """

    def __init__(self, seed=None, method=curand.CURAND_RNG_PSEUDO_DEFAULT):
        self._generator = curand.createGenerator(method)
        self.method = method
        self.seed(seed)

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        # When createGenerator raises an error, _generator is not initialized
        if is_shutting_down():
            return
        if hasattr(self, '_generator'):
            curand.destroyGenerator(self._generator)

    def _update_seed(self, size):
        self._rk_seed = (self._rk_seed + size) % _UINT64_MAX

    def _generate_normal(self, func, size, dtype, *args):
        # curand functions below don't support odd size.
        # * curand.generateNormal
        # * curand.generateNormalDouble
        # * curand.generateLogNormal
        # * curand.generateLogNormalDouble
        size = _core.get_size(size)
        element_size = _core.internal.prod(size)
        if element_size % 2 == 0:
            out = cupy.empty(size, dtype=dtype)
            func(self._generator, out.data.ptr, out.size, *args)
            return out
        else:
            out = cupy.empty((element_size + 1,), dtype=dtype)
            func(self._generator, out.data.ptr, out.size, *args)
            return out[:element_size].reshape(size)

    # NumPy compatible functions

    def beta(self, a, b, size=None, dtype=float):
        """Returns an array of samples drawn from the beta distribution.

        .. seealso::
            - :func:`cupy.random.beta` for full documentation
            - :meth:`numpy.random.RandomState.beta`
        """
        a, b = cupy.asarray(a), cupy.asarray(b)
        if size is None:
            size = cupy.broadcast(a, b).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.beta_kernel(a, b, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def binomial(self, n, p, size=None, dtype=int):
        """Returns an array of samples drawn from the binomial distribution.

        .. seealso::
            - :func:`cupy.random.binomial` for full documentation
            - :meth:`numpy.random.RandomState.binomial`
        """
        n, p = cupy.asarray(n), cupy.asarray(p)
        if size is None:
            size = cupy.broadcast(n, p).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.binomial_kernel(n, p, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def chisquare(self, df, size=None, dtype=float):
        """Returns an array of samples drawn from the chi-square distribution.

        .. seealso::
            - :func:`cupy.random.chisquare` for full documentation
            - :meth:`numpy.random.RandomState.chisquare`
        """
        df = cupy.asarray(df)
        if size is None:
            size = df.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.chisquare_kernel(df, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def dirichlet(self, alpha, size=None, dtype=float):
        """Returns an array of samples drawn from the dirichlet distribution.

        .. seealso::
            - :func:`cupy.random.dirichlet` for full documentation
            - :meth:`numpy.random.RandomState.dirichlet`
        """
        alpha = cupy.asarray(alpha)
        if size is None:
            size = alpha.shape
        elif isinstance(size, (int, cupy.integer)):
            size = (size,) + alpha.shape
        else:
            size += alpha.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.standard_gamma_kernel(alpha, self._rk_seed, y)
        y /= y.sum(axis=-1, keepdims=True)
        self._update_seed(y.size)
        return y

    def exponential(self, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a exponential distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.exponential` for full documentation
            - :meth:`numpy.random.RandomState.exponential`
        """
        scale = cupy.asarray(scale, dtype)
        if (scale < 0).any():  # synchronize!
            raise ValueError('scale < 0')
        if size is None:
            size = scale.shape
        x = self.standard_exponential(size, dtype)
        x *= scale
        return x

    def f(self, dfnum, dfden, size=None, dtype=float):
        """Returns an array of samples drawn from the f distribution.

        .. seealso::
            - :func:`cupy.random.f` for full documentation
            - :meth:`numpy.random.RandomState.f`
        """
        dfnum, dfden = cupy.asarray(dfnum), cupy.asarray(dfden)
        if size is None:
            size = cupy.broadcast(dfnum, dfden).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.f_kernel(dfnum, dfden, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def gamma(self, shape, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a gamma distribution.

        .. seealso::
            - :func:`cupy.random.gamma` for full documentation
            - :meth:`numpy.random.RandomState.gamma`
        """
        shape, scale = cupy.asarray(shape), cupy.asarray(scale)
        if size is None:
            size = cupy.broadcast(shape, scale).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.standard_gamma_kernel(shape, self._rk_seed, y)
        y *= scale
        self._update_seed(y.size)
        return y

    def geometric(self, p, size=None, dtype=int):
        """Returns an array of samples drawn from the geometric distribution.

        .. seealso::
            - :func:`cupy.random.geometric` for full documentation
            - :meth:`numpy.random.RandomState.geometric`
        """
        p = cupy.asarray(p)
        if size is None:
            size = p.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.geometric_kernel(p, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def hypergeometric(self, ngood, nbad, nsample, size=None, dtype=int):
        """Returns an array of samples drawn from the hypergeometric distribution.

        .. seealso::
            - :func:`cupy.random.hypergeometric` for full documentation
            - :meth:`numpy.random.RandomState.hypergeometric`
        """
        ngood, nbad, nsample = \
            cupy.asarray(ngood), cupy.asarray(nbad), cupy.asarray(nsample)
        if size is None:
            size = cupy.broadcast(ngood, nbad, nsample).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.hypergeometric_kernel(ngood, nbad, nsample, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    _laplace_kernel = _core.ElementwiseKernel(
        'T x, T loc, T scale', 'T y',
        'y = loc + scale * ((x <= 0.5) ? log(x + x): -log(x + x - 1.0))',
        'laplace_kernel')

    def laplace(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from the laplace distribution.

        .. seealso::
            - :func:`cupy.random.laplace` for full documentation
            - :meth:`numpy.random.RandomState.laplace`
        """
        loc = cupy.asarray(loc, dtype)
        scale = cupy.asarray(scale, dtype)
        if size is None:
            size = cupy.broadcast(loc, scale).shape
        x = self._random_sample_raw(size, dtype)
        RandomState._laplace_kernel(x, loc, scale, x)
        return x

    def logistic(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from the logistic distribution.

        .. seealso::
            - :func:`cupy.random.logistic` for full documentation
            - :meth:`numpy.random.RandomState.logistic`
        """
        loc, scale = cupy.asarray(loc), cupy.asarray(scale)
        if size is None:
            size = cupy.broadcast(loc, scale).shape
        x = cupy.empty(shape=size, dtype=dtype)
        _kernels.open_uniform_kernel(self._rk_seed, x)
        self._update_seed(x.size)
        x = (1.0 - x) / x
        cupy.log(x, out=x)
        cupy.multiply(x, scale, out=x)
        cupy.add(x, loc, out=x)
        return x

    def lognormal(self, mean=0.0, sigma=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a log normal distribution.

        .. seealso::
            - :func:`cupy.random.lognormal` for full documentation
            - :meth:`numpy.random.RandomState.lognormal`

        """
        if any(isinstance(arg, cupy.ndarray) for arg in (mean, sigma)):
            x = self.normal(mean, sigma, size, dtype)
            cupy.exp(x, out=x)
            return x
        if size is None:
            size = ()
        dtype = _check_and_get_dtype(dtype)
        if dtype.char == 'f':
            func = curand.generateLogNormal
        else:
            func = curand.generateLogNormalDouble
        return self._generate_normal(func, size, dtype, mean, sigma)

    def logseries(self, p, size=None, dtype=int):
        """Returns an array of samples drawn from a log series distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.logseries` for full documentation
            - :meth:`numpy.random.RandomState.logseries`

        """
        p = cupy.asarray(p)
        if cupy.any(p <= 0):  # synchronize!
            raise ValueError('p <= 0.0')
        if cupy.any(p >= 1):  # synchronize!
            raise ValueError('p >= 1.0')
        if size is None:
            size = p.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.logseries_kernel(p, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def multivariate_normal(self, mean, cov, size=None, check_valid='ignore',
                            tol=1e-08, method='cholesky', dtype=float):
        """Returns an array of samples drawn from the multivariate normal
        distribution.

        .. warning::
            This function calls one or more cuSOLVER routine(s) which may yield
            invalid results if input conditions are not met.
            To detect these invalid results, you can set the `linalg`
            configuration to a value that is not `ignore` in
            :func:`cupyx.errstate` or :func:`cupyx.seterr`.

        .. seealso::
            - :func:`cupy.random.multivariate_normal` for full documentation
            - :meth:`numpy.random.RandomState.multivariate_normal`
        """
        _util.experimental('cupy.random.RandomState.multivariate_normal')
        mean = cupy.asarray(mean, dtype=dtype)
        cov = cupy.asarray(cov, dtype=dtype)
        if size is None:
            shape = []
        elif isinstance(size, (int, cupy.integer)):
            shape = [size]
        else:
            shape = size

        if len(mean.shape) != 1:
            raise ValueError('mean must be 1 dimensional')
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError('cov must be 2 dimensional and square')
        if mean.shape[0] != cov.shape[0]:
            raise ValueError('mean and cov must have same length')

        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])

        if method not in {'eigh', 'svd', 'cholesky'}:
            raise ValueError(
                "method must be one of {'eigh', 'svd', 'cholesky'}")

        if check_valid != 'ignore':
            if check_valid != 'warn' and check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")

        if check_valid == 'warn':
            with cupyx.errstate(linalg='raise'):
                try:
                    decomp = cupy.linalg.cholesky(cov)
                except LinAlgError:
                    with cupyx.errstate(linalg='ignore'):
                        if method != 'cholesky':
                            if method == 'eigh':
                                (s, u) = cupy.linalg.eigh(cov)
                                psd = not cupy.any(s < -tol)
                            if method == 'svd':
                                (u, s, vh) = cupy.linalg.svd(cov)
                                psd = cupy.allclose(cupy.dot(vh.T * s, vh),
                                                    cov, rtol=tol, atol=tol)
                            decomp = u * cupy.sqrt(cupy.abs(s))
                            if not psd:
                                warnings.warn("covariance is not positive-" +
                                              "semidefinite, output may be " +
                                              "invalid.", RuntimeWarning)

                        else:
                            warnings.warn("covariance is not positive-" +
                                          "semidefinite, output *is* " +
                                          "invalid.", RuntimeWarning)
                            decomp = cupy.linalg.cholesky(cov)

        else:
            with cupyx.errstate(linalg=check_valid):
                try:
                    if method == 'cholesky':
                        decomp = cupy.linalg.cholesky(cov)
                    elif method == 'eigh':
                        (s, u) = cupy.linalg.eigh(cov)
                        decomp = u * cupy.sqrt(cupy.abs(s))
                    elif method == 'svd':
                        (u, s, vh) = cupy.linalg.svd(cov)
                        decomp = u * cupy.sqrt(cupy.abs(s))

                except LinAlgError:
                    raise LinAlgError("Matrix is not positive definite; if " +
                                      "matrix is positive-semidefinite, set" +
                                      "'check_valid' to 'warn'")

        x = self.standard_normal(final_shape,
                                 dtype=dtype).reshape(-1, mean.shape[0])
        x = cupy.dot(decomp, x.T)
        x = x.T
        x += mean
        x.shape = tuple(final_shape)
        return x

    def negative_binomial(self, n, p, size=None, dtype=int):
        """Returns an array of samples drawn from the negative binomial distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.negative_binomial` for full documentation
            - :meth:`numpy.random.RandomState.negative_binomial`
        """
        n = cupy.asarray(n)
        p = cupy.asarray(p)
        if cupy.any(n <= 0):  # synchronize!
            raise ValueError('n <= 0')
        if cupy.any(p < 0):  # synchronize!
            raise ValueError('p < 0')
        if cupy.any(p > 1):  # synchronize!
            raise ValueError('p > 1')
        y = self.gamma(n, (1-p)/p, size)
        return self.poisson(y, dtype=dtype)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of normally distributed samples.

        .. seealso::
            - :func:`cupy.random.normal` for full documentation
            - :meth:`numpy.random.RandomState.normal`

        """
        dtype = _check_and_get_dtype(dtype)
        if size is None:
            size = cupy.broadcast(loc, scale).shape
        if dtype.char == 'f':
            func = curand.generateNormal
        else:
            func = curand.generateNormalDouble
        if isinstance(scale, cupy.ndarray):
            x = self._generate_normal(func, size, dtype, 0.0, 1.0)
            cupy.multiply(x, scale, out=x)
            cupy.add(x, loc, out=x)
        elif isinstance(loc, cupy.ndarray):
            x = self._generate_normal(func, size, dtype, 0.0, scale)
            cupy.add(x, loc, out=x)
        else:
            x = self._generate_normal(func, size, dtype, loc, scale)
        return x

    def pareto(self, a, size=None, dtype=float):
        """Returns an array of samples drawn from the pareto II distribution.

        .. seealso::
            - :func:`cupy.random.pareto` for full documentation
            - :meth:`numpy.random.RandomState.pareto`
        """
        a = cupy.asarray(a)
        if size is None:
            size = a.shape
        x = self._random_sample_raw(size, dtype)
        cupy.log(x, out=x)
        cupy.exp(-x/a, out=x)
        return x - 1

    def noncentral_chisquare(self, df, nonc, size=None, dtype=float):
        """Returns an array of samples drawn from the noncentral chi-square
        distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.noncentral_chisquare` for full documentation
            - :meth:`numpy.random.RandomState.noncentral_chisquare`
        """
        df, nonc = cupy.asarray(df), cupy.asarray(nonc)
        if cupy.any(df <= 0):  # synchronize!
            raise ValueError('df <= 0')
        if cupy.any(nonc < 0):  # synchronize!
            raise ValueError('nonc < 0')
        if size is None:
            size = cupy.broadcast(df, nonc).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.noncentral_chisquare_kernel(df, nonc, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def noncentral_f(self, dfnum, dfden, nonc, size=None, dtype=float):
        """Returns an array of samples drawn from the noncentral F distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.noncentral_f` for full documentation
            - :meth:`numpy.random.RandomState.noncentral_f`
        """
        dfnum, dfden, nonc = \
            cupy.asarray(dfnum), cupy.asarray(dfden), cupy.asarray(nonc)
        if cupy.any(dfnum <= 0):  # synchronize!
            raise ValueError('dfnum <= 0')
        if cupy.any(dfden <= 0):  # synchronize!
            raise ValueError('dfden <= 0')
        if cupy.any(nonc < 0):  # synchronize!
            raise ValueError('nonc < 0')
        if size is None:
            size = cupy.broadcast(dfnum, dfden, nonc).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.noncentral_f_kernel(dfnum, dfden, nonc, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def poisson(self, lam=1.0, size=None, dtype=int):
        """Returns an array of samples drawn from the poisson distribution.

        .. seealso::
            - :func:`cupy.random.poisson` for full documentation
            - :meth:`numpy.random.RandomState.poisson`
        """
        lam = cupy.asarray(lam)
        if size is None:
            size = lam.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.poisson_kernel(lam, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def power(self, a, size=None, dtype=float):
        """Returns an array of samples drawn from the power distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.power` for full documentation
            - :meth:`numpy.random.RandomState.power`
        """
        a = cupy.asarray(a)
        if cupy.any(a < 0):  # synchronize!
            raise ValueError('a < 0')
        if size is None:
            size = a.shape
        x = self.standard_exponential(size=size, dtype=dtype)
        cupy.exp(-x, out=x)
        cupy.add(1, -x, out=x)
        cupy.power(x, 1./a, out=x)
        return x

    def rand(self, *size, **kwarg):
        """Returns uniform random values over the interval ``[0, 1)``.

        .. seealso::
            - :func:`cupy.random.rand` for full documentation
            - :meth:`numpy.random.RandomState.rand`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('rand() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.random_sample(size=size, dtype=dtype)

    def randn(self, *size, **kwarg):
        """Returns an array of standard normal random values.

        .. seealso::
            - :func:`cupy.random.randn` for full documentation
            - :meth:`numpy.random.RandomState.randn`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('randn() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.normal(size=size, dtype=dtype)

    _mod1_kernel = _core.ElementwiseKernel(
        '', 'T x', 'x = (x == (T)1) ? 0 : x', 'cupy_random_x_mod_1')

    def _random_sample_raw(self, size, dtype):
        dtype = _check_and_get_dtype(dtype)
        out = cupy.empty(size, dtype=dtype)
        if dtype.char == 'f':
            func = curand.generateUniform
        else:
            func = curand.generateUniformDouble
        func(self._generator, out.data.ptr, out.size)
        return out

    def random_sample(self, size=None, dtype=float):
        """Returns an array of random values over the interval ``[0, 1)``.

        .. seealso::
            - :func:`cupy.random.random_sample` for full documentation
            - :meth:`numpy.random.RandomState.random_sample`

        """
        if size is None:
            size = ()
        out = self._random_sample_raw(size, dtype)
        RandomState._mod1_kernel(out)
        return out

    def rayleigh(self, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a rayleigh distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.rayleigh` for full documentation
            - :meth:`numpy.random.RandomState.rayleigh`
        """
        scale = cupy.asarray(scale)
        if size is None:
            size = scale.shape
        if cupy.any(scale < 0):  # synchronize!
            raise ValueError('scale < 0')
        x = self._random_sample_raw(size, dtype)
        x = cupy.log(x, out=x)
        x = cupy.multiply(x, -2., out=x)
        x = cupy.sqrt(x, out=x)
        x = cupy.multiply(x, scale, out=x)
        return x

    def _interval(self, mx, size):
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
            Currently, only 32 bit or 64 bit integers can be sampled.
        """  # NOQA
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = size,

        if mx == 0:
            return cupy.zeros(size, dtype=numpy.uint32)

        if mx < 0:
            raise ValueError(
                'mx must be non-negative (actual: {})'.format(mx))
        elif mx <= _UINT32_MAX:
            dtype = numpy.uint32
            upper_limit = _UINT32_MAX - (1 << 32) % (mx + 1)
        elif mx <= _UINT64_MAX:
            dtype = numpy.uint64
            upper_limit = _UINT64_MAX - (1 << 64) % (mx + 1)
        else:
            raise ValueError(
                'mx must be within uint64 range (actual: {})'.format(mx))

        n_sample = functools.reduce(operator.mul, size, 1)
        if n_sample == 0:
            return cupy.empty(size, dtype=dtype)
        sample = self._curand_generate(n_sample, dtype)

        mx1 = mx + 1
        if mx1 != (1 << (mx1.bit_length() - 1)):
            # Get index of samples that exceed the upper limit
            ng_indices = self._get_indices(sample, upper_limit, False)
            n_ng = ng_indices.size

            while n_ng > 0:
                n_supplement = max(n_ng * 2, 1024)
                supplement = self._curand_generate(n_supplement, dtype)

                # Get index of supplements that are within the upper limit
                ok_indices = self._get_indices(supplement, upper_limit, True)
                n_ok = ok_indices.size

                # Replace the values that exceed the upper limit
                if n_ok >= n_ng:
                    sample[ng_indices] = supplement[ok_indices[:n_ng]]
                    n_ng = 0
                else:
                    sample[ng_indices[:n_ok]] = supplement[ok_indices]
                    ng_indices = ng_indices[n_ok:]
                    n_ng -= n_ok
            sample %= mx1
        else:
            mask = (1 << mx.bit_length()) - 1
            sample &= mask

        return sample.reshape(size)

    def _curand_generate(self, num, dtype):
        sample = cupy.empty((num,), dtype=dtype)
        # Call 32-bit RNG to fill 32-bit or 64-bit `sample`
        size32 = sample.view(dtype=numpy.uint32).size
        curand.generate(self._generator, sample.data.ptr, size32)
        return sample

    def _get_indices(self, sample, upper_limit, cond):
        dtype = numpy.uint32 if sample.size < 2**32 else numpy.uint64
        flags = (sample <= upper_limit) if cond else (sample > upper_limit)
        csum = cupy.cumsum(flags, dtype=dtype)
        del flags
        indices = cupy.empty((int(csum[-1]),), dtype=dtype)
        self._kernel_get_indices(csum, indices, size=csum.size)
        return indices

    _kernel_get_indices = _core.ElementwiseKernel(
        'raw U csum', 'raw U indices',
        '''
        int j = 0;
        if (i > 0) { j = csum[i-1]; }
        if (csum[i] > j) { indices[j] = i; }
        ''',
        'cupy_get_indices')

    def seed(self, seed=None):
        """Resets the state of the random number generator with a seed.

        .. seealso::
            - :func:`cupy.random.seed` for full documentation
            - :meth:`numpy.random.RandomState.seed`

        """
        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = int(seed_str, 16)
            except NotImplementedError:
                seed = (time.time() * 1000000) % _UINT64_MAX
        else:
            if isinstance(seed, numpy.ndarray):
                seed = int(hashlib.md5(seed).hexdigest()[:16], 16)
            else:
                seed = int(
                    numpy.asarray(seed).astype(numpy.uint64, casting='safe'))

        curand.setPseudoRandomGeneratorSeed(self._generator, seed)
        if (self.method not in (curand.CURAND_RNG_PSEUDO_MT19937,
                                curand.CURAND_RNG_PSEUDO_MTGP32)):
            curand.setGeneratorOffset(self._generator, 0)

        self._rk_seed = seed

    def standard_cauchy(self, size=None, dtype=float):
        """Returns an array of samples drawn from the standard cauchy distribution.

        .. seealso::
            - :func:`cupy.random.standard_cauchy` for full documentation
            - :meth:`numpy.random.RandomState.standard_cauchy`
        """
        x = self.uniform(size=size, dtype=dtype)
        return cupy.tan(cupy.pi * (x - 0.5))

    def standard_exponential(self, size=None, dtype=float):
        """Returns an array of samples drawn from the standard exp distribution.

         .. seealso::
            - :func:`cupy.random.standard_exponential` for full documentation
            - :meth:`numpy.random.RandomState.standard_exponential`
        """
        if size is None:
            size = ()
        x = self._random_sample_raw(size, dtype)
        return -cupy.log(x, out=x)

    def standard_gamma(self, shape, size=None, dtype=float):
        """Returns an array of samples drawn from a standard gamma distribution.

        .. seealso::
            - :func:`cupy.random.standard_gamma` for full documentation
            - :meth:`numpy.random.RandomState.standard_gamma`
        """
        shape = cupy.asarray(shape)
        if size is None:
            size = shape.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.standard_gamma_kernel(shape, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def standard_normal(self, size=None, dtype=float):
        """Returns samples drawn from the standard normal distribution.

        .. seealso::
            - :func:`cupy.random.standard_normal` for full documentation
            - :meth:`numpy.random.RandomState.standard_normal`

        """
        return self.normal(size=size, dtype=dtype)

    def standard_t(self, df, size=None, dtype=float):
        """Returns an array of samples drawn from the standard t distribution.

        .. seealso::
            - :func:`cupy.random.standard_t` for full documentation
            - :meth:`numpy.random.RandomState.standard_t`
        """
        df = cupy.asarray(df)
        if size is None:
            size = df.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.standard_t_kernel(df, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def tomaxint(self, size=None):
        """Draws integers between 0 and max integer inclusive.

        Return a sample of uniformly distributed random integers in the
        interval [0, ``np.iinfo(np.int_).max``]. The `np.int_` type translates
        to the C long integer type and its precision is platform dependent.

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

    _triangular_kernel = _core.ElementwiseKernel(
        'L left, M mode, R right', 'T x',
        """
        T base, leftbase, ratio, leftprod, rightprod;

        base = right - left;
        leftbase = mode - left;
        ratio = leftbase / base;
        leftprod = leftbase*base;
        rightprod = (right - mode)*base;

        if (x <= ratio)
        {
            x = left + sqrt(x*leftprod);
        } else
        {
            x = right - sqrt((1.0 - x) * rightprod);
        }
        """,
        'triangular_kernel'
    )

    def triangular(self, left, mode, right, size=None, dtype=float):
        """Returns an array of samples drawn from the triangular distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.triangular` for full documentation
            - :meth:`numpy.random.RandomState.triangular`
        """
        left, mode, right = \
            cupy.asarray(left), cupy.asarray(mode), cupy.asarray(right)
        if cupy.any(left > mode):  # synchronize!
            raise ValueError('left > mode')
        if cupy.any(mode > right):  # synchronize!
            raise ValueError('mode > right')
        if cupy.any(left == right):  # synchronize!
            raise ValueError('left == right')
        if size is None:
            size = cupy.broadcast(left, mode, right).shape
        x = self.random_sample(size=size, dtype=dtype)
        return RandomState._triangular_kernel(left, mode, right, x)

    _scale_kernel = _core.ElementwiseKernel(
        'T low, T high', 'T x',
        'x = T(low) + x * T(high - low)',
        'cupy_scale')

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float):
        """Returns an array of uniformly-distributed samples over an interval.

        .. seealso::
            - :func:`cupy.random.uniform` for full documentation
            - :meth:`numpy.random.RandomState.uniform`

        """
        dtype = numpy.dtype(dtype)
        rand = self.random_sample(size=size, dtype=dtype)
        if not numpy.isscalar(low):
            low = cupy.asarray(low, dtype)
        if not numpy.isscalar(high):
            high = cupy.asarray(high, dtype)
        return RandomState._scale_kernel(low, high, rand)

    def vonmises(self, mu, kappa, size=None, dtype=float):
        """Returns an array of samples drawn from the von Mises distribution.

        .. seealso::
            - :func:`cupy.random.vonmises` for full documentation
            - :meth:`numpy.random.RandomState.vonmises`
        """
        mu, kappa = cupy.asarray(mu), cupy.asarray(kappa)
        if size is None:
            size = cupy.broadcast(mu, kappa).shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.vonmises_kernel(mu, kappa, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    _wald_kernel = _core.ElementwiseKernel(
        'T mean, T scale, T U', 'T X',
        """
            T mu_2l;
            T Y;
            mu_2l = mean / (2*scale);
            Y = mean*X*X;
            X = mean + mu_2l*(Y - sqrt(4*scale*Y + Y*Y));
            if (U > mean/(mean+X))
            {
                X = mean*mean/X;
            }
        """,
        'wald_scale')

    def wald(self, mean, scale, size=None, dtype=float):
        """Returns an array of samples drawn from the Wald distribution.

         .. seealso::
            - :func:`cupy.random.wald` for full documentation
            - :meth:`numpy.random.RandomState.wald`
        """
        mean, scale = \
            cupy.asarray(mean, dtype=dtype), cupy.asarray(scale, dtype=dtype)
        if size is None:
            size = cupy.broadcast(mean, scale).shape
        x = self.normal(size=size, dtype=dtype)
        u = self.random_sample(size=size, dtype=dtype)
        return RandomState._wald_kernel(mean, scale, u, x)

    def weibull(self, a, size=None, dtype=float):
        """Returns an array of samples drawn from the weibull distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.weibull` for full documentation
            - :meth:`numpy.random.RandomState.weibull`
        """
        a = cupy.asarray(a)
        if cupy.any(a < 0):  # synchronize!
            raise ValueError('a < 0')
        x = self.standard_exponential(size, dtype)
        cupy.power(x, 1./a, out=x)
        return x

    def zipf(self, a, size=None, dtype=int):
        """Returns an array of samples drawn from the Zipf distribution.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            - :func:`cupy.random.zipf` for full documentation
            - :meth:`numpy.random.RandomState.zipf`
        """
        a = cupy.asarray(a)
        if cupy.any(a <= 1.0):  # synchronize!
            raise ValueError('\'a\' must be a valid float > 1.0')
        if size is None:
            size = a.shape
        y = cupy.empty(shape=size, dtype=dtype)
        _kernels.zipf_kernel(a, self._rk_seed, y)
        self._update_seed(y.size)
        return y

    def choice(self, a, size=None, replace=True, p=None):
        """Returns an array of random values from a given 1-D array.

        .. seealso::
            - :func:`cupy.random.choice` for full documentation
            - :meth:`numpy.random.choice`

        """
        if a is None:
            raise ValueError('a must be 1-dimensional or an integer')
        if isinstance(a, cupy.ndarray) and a.ndim == 0:
            raise NotImplementedError
        if isinstance(a, int):
            a_size = a
            if a_size < 0:
                raise ValueError('a must be greater than or equal to 0')
        else:
            a = cupy.array(a, copy=False)
            if a.ndim != 1:
                raise ValueError('a must be 1-dimensional or an integer')
            a_size = len(a)

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

        if a_size == 0 and size > 0:
            raise ValueError('a cannot be empty unless no samples are taken')

        if not replace and p is None:
            if a_size < size:
                raise ValueError(
                    'Cannot take a larger sample than population when '
                    '\'replace=False\'')
            if isinstance(a, int):
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
            if not isinstance(shape, int):
                index = cupy.reshape(index, shape)
        else:
            if a_size == 0:  # TODO: (#4511) Fix `randint` instead
                a_size = 1
            index = self.randint(0, a_size, size=shape)
            # Align the dtype with NumPy
            index = index.astype(cupy.int64, copy=False)

        if isinstance(a, int):
            return index

        if index.ndim == 0:
            return cupy.array(a[index], dtype=a.dtype)

        return a[index]

    def shuffle(self, a):
        """Returns a shuffled array.

        .. seealso::
            - :func:`cupy.random.shuffle` for full documentation
            - :meth:`numpy.random.shuffle`

        """
        if not isinstance(a, cupy.ndarray):
            raise TypeError('The array must be cupy.ndarray')

        if a.ndim == 0:
            raise TypeError('An array whose ndim is 0 is not supported')

        a[:] = a[self._permutation(len(a))]

    def permutation(self, a):
        """Returns a permuted range or a permutation of an array."""
        if isinstance(a, int):
            return self._permutation(a)
        else:
            return a[self._permutation(len(a))]

    def _permutation(self, num):
        """Returns a permuted range."""
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
                _cupy_permutation(sample, j_start, j_end, array, size=num)
        else:
            # When num > 32M, argsort is used, because it is faster than
            # custom kernel. See https://github.com/cupy/cupy/pull/603.
            array = cupy.argsort(sample)
        return array

    _gumbel_kernel = _core.ElementwiseKernel(
        'T x, T loc, T scale', 'T y',
        'y = T(loc) - log(-log(x)) * T(scale)',
        'gumbel_kernel')

    def gumbel(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a Gumbel distribution.

        .. seealso::
            - :func:`cupy.random.gumbel` for full documentation
            - :meth:`numpy.random.RandomState.gumbel`
        """
        if not numpy.isscalar(loc):
            loc = cupy.asarray(loc, dtype)
        if not numpy.isscalar(scale):
            scale = cupy.asarray(scale, dtype)
        if size is None:
            size = cupy.broadcast(loc, scale).shape
        x = self._random_sample_raw(size=size, dtype=dtype)
        RandomState._gumbel_kernel(x, loc, scale, x)
        return x

    def randint(self, low, high=None, size=None, dtype=int):
        """Returns a scalar or an array of integer values over ``[low, high)``.

        .. seealso::
            - :func:`cupy.random.randint` for full documentation
            - :meth:`numpy.random.RandomState.randint`
        """
        if high is None:
            lo = 0
            hi1 = int(low) - 1
        else:
            lo = int(low)
            hi1 = int(high) - 1

        if lo > hi1:
            raise ValueError('low >= high')
        if lo < cupy.iinfo(dtype).min:
            raise ValueError(
                'low is out of bounds for {}'.format(cupy.dtype(dtype).name))
        if hi1 > cupy.iinfo(dtype).max:
            raise ValueError(
                'high is out of bounds for {}'.format(cupy.dtype(dtype).name))

        diff = hi1 - lo
        x = self._interval(diff, size).astype(dtype, copy=False)
        cupy.add(x, lo, out=x)
        return x


_cupy_permutation = _core.ElementwiseKernel(
    'raw int32 sample, int32 j_start, int32 _j_end',
    'raw int32 array',
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
            it uses :func:`os.urandom` if available or :func:`time.time`
            otherwise. Note that this function does not support seeding by
            an integer array.

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
