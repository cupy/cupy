# distutils: language = c++
import numpy

from libc.stdint cimport intptr_t, uint64_t, int32_t, int64_t

import cupy
from cupy import _core
from cupy.cuda cimport stream
from cupy._core.core cimport _ndarray_base
from cupy_backends.cuda.api import runtime

_UINT32_MAX = 0xffffffff
_UINT64_MAX = 0xffffffffffffffff

cdef extern from 'cupy_distributions.cuh' nogil:
    cppclass rk_binomial_state:
        pass
    void init_curand_generator(
        int generator, intptr_t state_ptr, uint64_t seed,
        ssize_t size, intptr_t stream)
    void random_uniform(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream)
    void random_uniform_float(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream)
    void raw(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream)
    void interval_32(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, int32_t mx, int32_t mask)
    void interval_64(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, int64_t mx, int64_t mask)
    void beta(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, intptr_t a, intptr_t b)
    void exponential(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream)
    void geometric(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, intptr_t arg1)
    void hypergeometric(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream,
        intptr_t arg1, intptr_t arg2, intptr_t arg3)
    void logseries(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, intptr_t arg1)
    void standard_normal(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream)
    void standard_normal_float(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream)
    # if the types are the same, but the names are different
    # cython will fail when trying to create a PyObj wrapper
    # to use these functions from python
    # arg1 is shape
    void standard_gamma(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, intptr_t arg1)
    # arg1 is lam
    void poisson(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream, intptr_t arg1)
    void binomial(
        int generator, intptr_t state, ssize_t state_size, intptr_t out,
        ssize_t size, intptr_t stream,
        intptr_t arg1, intptr_t arg2, intptr_t arg3)


cdef _ndarray_base _array_data(_ndarray_base x):
    return cupy.array((x.data.ptr, x.ndim) + x.shape + x.strides)


class Generator:
    """Container for the BitGenerators.

    ``Generator`` exposes a number of methods for generating random
    numbers drawn from a variety of probability distributions. In addition to
    the distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    array filled with generated values is returned. If `size` is a tuple,
    then an array with that shape is filled and returned.
    The function :func:`numpy.random.default_rng` will instantiate
    a `Generator` with numpy's default `BitGenerator`.
    **No Compatibility Guarantee**
    ``Generator`` does not provide a version compatibility guarantee. In
    particular, as better algorithms evolve the bit stream may change.

    Args:
        bit_generator : (cupy.random.BitGenerator): BitGenerator to use
            as the core generator.

    """
    def __init__(self, bit_generator):
        if runtime.is_hip and int(str(runtime.runtimeGetVersion())[:3]) < 403:
            raise RuntimeError('Generator API not supported in ROCm<4.3,'
                               ' please use the legacy one or update ROCm.')
        self.bit_generator = bit_generator
        self._binomial_state = None

    def _check_output_array(self, dtype, size, out, check_only_c_cont=False):
        # Checks borrowed from NumPy
        # https://github.com/numpy/numpy/blob/cb557b79fa0ce467c881830f8e8e042c484ccfaa/numpy/random/_common.pyx#L235-L251
        dtype = numpy.dtype(dtype)
        if out.dtype.char != dtype.char:
            raise TypeError(
                f'Supplied output array has the wrong type. '
                f'Expected {dtype.name}, got {out.dtype.name}')
        if not out.flags.c_contiguous:
            if check_only_c_cont or not out.flags.f_contiguous:
                raise ValueError(
                    'Supplied output array is not contiguous,'
                    ' writable or aligned.')
        if size is not None:
            try:
                tup_size = tuple(size)
            except TypeError:
                tup_size = tuple([size])
            if tup_size != out.shape:
                raise ValueError(
                    'size must match out.shape when used together')

    def random(self, size=None, dtype=numpy.float64, out=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random` by `(b-a)` and add `a`::

          (b - a) * random() + a

        Args:
            size (None or int or tuple of ints): The shape of returned value.
            dtype: Data type specifier.
            out (cupy.ndarray, optional): If specified, values will be written
                to this array

        Returns:
            cupy.ndarray: Samples uniformly drawn from the [0, 1) interval

        .. seealso::
            - :meth:`numpy.random.Generator.random`
        """
        cdef _ndarray_base y

        if out is not None:
            self._check_output_array(dtype, size, out)

        y = _core.ndarray(size if size is not None else (), dtype)
        if y.dtype.char == 'd':
            _launch_dist(self.bit_generator, random_uniform, y, ())
        else:
            _launch_dist(self.bit_generator, random_uniform_float, y, ())
        if out is not None:
            _core.elementwise_copy(y, out)
            y = out
        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=numpy.float64):
        """
        Draw samples from a uniform distribution.
        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float or array_like of floats, optional
            Lower boundary of the output interval.  All values generated will
            be greater than or equal to low.  The default value is 0.
        high : float or array_like of floats
            Upper boundary of the output interval.  All values generated will
            be less than high.  The high limit may be included in the returned
            array of floats due to floating-point rounding in the equation
            ``low + (high-low) * random()``.  high - low must be
            non-negative.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``low`` and ``high`` are both
            scalars.  Otherwise, ``cupy.broadcast(low, high).size`` samples are
            drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized uniform distribution.
        See Also
        --------
        - :meth:`numpy.random.Generator.uniform`
        - :meth:`integers`: Discrete uniform distribution, yielding integers.
        - :meth:`random`: Floats uniformly distributed over ``[0, 1)``.
        """

        cdef _ndarray_base y

        low = cupy.asarray(low)
        high = cupy.asarray(high)

        if size is None:
            size = cupy.broadcast(low, high).shape

        y = _core.ndarray(size, dtype)
        low = cupy.broadcast_to(low, y.shape)
        high = cupy.broadcast_to(high, y.shape)

        if y.dtype.char == 'd':
            _launch_dist(self.bit_generator, random_uniform, y, ())
        else:
            _launch_dist(self.bit_generator, random_uniform_float, y, ())

        y = low + (high - low) * y

        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)

    def integers(
            self, low, high=None, size=None,
            dtype=numpy.int64, endpoint=False):
        """Returns a scalar or an array of integer values over an interval.

        Each element of returned values are independently sampled from
        uniform distribution over the ``[low, high)`` or ``[low, high]``
        intervals.

        Args:
            low (int): If ``high`` is not ``None``,
                it is the lower bound of the interval.
                Otherwise, it is the **upper** bound of the interval
                and lower bound of the interval is set to ``0``.
            high (int): Upper bound of the interval.
            size (None or int or tuple of ints): The shape of returned value.
            dtype: Data type specifier.
            endpoint (bool): If ``True``, sample from ``[low, high]``.
                Defaults to ``False``

        Returns:
            int or cupy.ndarray of ints: If size is ``None``,
            it is single integer sampled.
            If size is integer, it is the 1D-array of length ``size`` element.
            Otherwise, it is the array whose shape specified by ``size``.

        .. seealso::
            - :meth:`numpy.random.Generator.integers`
        """
        cdef _ndarray_base y
        if high is None:
            lo = 0
            hi1 = int(low)
        else:
            lo = int(low)
            hi1 = int(high)

        if not endpoint:
            hi1 -= 1

        if lo > hi1:
            raise ValueError('low >= high')
        if lo < cupy.iinfo(dtype).min:
            raise ValueError(
                'low is out of bounds for {}'.format(cupy.dtype(dtype).name))
        if hi1 > cupy.iinfo(dtype).max:
            raise ValueError(
                'high is out of bounds for {}'.format(cupy.dtype(dtype).name))

        diff = hi1 - lo

        cdef uint64_t mask = (1 << diff.bit_length()) - 1
        # TODO adjust dtype
        if diff <= _UINT32_MAX:
            pdtype = numpy.uint32
        elif diff <= _UINT64_MAX:
            pdtype = numpy.uint64
        else:
            raise ValueError(
                f'high - low must be within uint64 range (actual: {diff})')

        y = _core.ndarray(size if size is not None else (), pdtype)
        if pdtype is numpy.uint32:
            _launch_dist(self.bit_generator, interval_32, y, (diff, mask))
        else:
            _launch_dist(self.bit_generator, interval_64, y, (diff, mask))
        return cupy.add(lo, y, dtype=dtype)

    def beta(self, a, b, size=None, dtype=numpy.float64):
        """Beta distribution.

        Returns an array of samples drawn from the beta distribution. Its
        probability density function is defined as

        .. math::
           f(x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)}.

        Args:
            a (float): Parameter of the beta distribution :math:`\\alpha`.
            b (float): Parameter of the beta distribution :math:`\\beta`.
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.
            dtype: Data type specifier. Only :class:`numpy.float32` and
                :class:`numpy.float64` types are allowed.

        Returns:
            cupy.ndarray: Samples drawn from the beta distribution.

        .. seealso::
            :meth:`numpy.random.Generator.beta`
        """
        cdef _ndarray_base y

        if not isinstance(a, _ndarray_base):
            if type(a) in (float, int):
                a = cupy.asarray(a, numpy.float64)
            else:
                raise TypeError('a is required to be a cupy.ndarray'
                                ' or a scalar')
        if not isinstance(b, _ndarray_base):
            if type(b) in (float, int):
                b = cupy.asarray(b, numpy.float64)
            else:
                raise TypeError('b is required to be a cupy.ndarray'
                                ' or a scalar')

        if size is not None and not isinstance(size, tuple):
            size = (size, )
        elif size is None:
            size = cupy.broadcast(a, b).shape

        y = _core.ndarray(size, numpy.float64)

        a = cupy.broadcast_to(a, y.shape)
        b = cupy.broadcast_to(b, y.shape)

        _launch_dist(self.bit_generator, beta, y, (a, b))
        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)

    def chisquare(self, df, size=None):
        """Chi-square distribution.

        Returns an array of samples drawn from the chi-square distribution. Its
        probability density function is defined as

        .. math::
           f(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}x^{k/2-1}e^{-x/2}.

        Args:
            df (float or array_like of floats): Degree of freedom :math:`k`.
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.

        Returns:
            cupy.ndarray: Samples drawn from the chi-square distribution.

        .. seealso::
            :meth:`numpy.random.Generator.chisquare`
        """

        cdef _ndarray_base y

        if not isinstance(df, _ndarray_base):
            if type(df) in (float, int):
                df = cupy.asarray(df, numpy.float64)
            else:
                raise TypeError('df is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            df = df.astype('d', copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size,)
        if size is None:
            size = df.shape

        y = _core.ndarray(size, numpy.float64)

        df = cupy.broadcast_to(df, y.shape)
        y = self.standard_gamma(df / 2)
        y *= 2
        return y

    def dirichlet(self, alpha, size=None):
        """Dirichlet distribution.

        Returns an array of samples drawn from the dirichlet distribution. Its
        probability density function is defined as

        .. math::
            f(x) = \\frac{\\Gamma(\\sum_{i=1}^K\\alpha_i)} \
                {\\prod_{i=1}^{K}\\Gamma(\\alpha_i)} \
                \\prod_{i=1}^Kx_i^{\\alpha_i-1}.

        Args:
            alpha (array): Parameters of the dirichlet distribution
                :math:`\\alpha`.
            size (int or tuple of ints): The shape of the array. If ``None``,
                array of ``alpha.shape`` is generated

        Returns:
            cupy.ndarray: Samples drawn from the dirichlet distribution.

        .. seealso::
            :meth:`numpy.random.Generator.dirichlet`
        """

        if not isinstance(alpha, _ndarray_base):
            if type(alpha) in (float, int):
                alpha = cupy.asarray(alpha, numpy.float64)
            else:
                raise TypeError('alpha is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            alpha = alpha.astype('d', copy=False)

        if size is not None:
            if not isinstance(size, tuple):
                size = (size,)
            size += alpha.shape

        elif size is None:
            size = alpha.shape

        y = self.standard_gamma(alpha, size)
        y /= y.sum(axis=-1, keepdims=True)
        return y

    def exponential(self, scale=1.0, size=None):
        """Exponential distribution.

        Returns an array of samples drawn from the exponential distribution.
        Its probability density function is defined as

        .. math::
           f(x) = \\frac{1}{\\beta}\\exp (-\\frac{x}{\\beta}).

        Args:
            scale (float or array_like of floats): The scale parameter
                :math:`\\beta`.
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.

        Returns:
            cupy.ndarray: Samples drawn from the exponential distribution.

        .. seealso::
            :meth:`numpy.random.Generator.exponential`
        """
        return self.standard_exponential(size) * scale

    def f(self, dfnum, dfden, size=None):
        """F distribution.

        Returns an array of samples drawn from the f distribution. Its
        probability density function is defined as

        .. math::
            f(x) = \\frac{1}{B(\\frac{d_1}{2},\\frac{d_2}{2})} \
                \\left(\\frac{d_1}{d_2}\\right)^{\\frac{d_1}{2}} \
                x^{\\frac{d_1}{2}-1} \
                \\left(1+\\frac{d_1}{d_2}x\\right) \
                ^{-\\frac{d_1+d_2}{2}}.

        Args:
            dfnum (float or array_like of floats): Degrees of freedom in
                numerator, :math:`d_1`.
            dfden (float or array_like of floats): Degrees of freedom in
                denominator, :math:`d_2`.
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.

        Returns:
            cupy.ndarray: Samples drawn from the f distribution.

        .. seealso::
            :meth:`numpy.random.Generator.f`
        """
        if not isinstance(dfnum, _ndarray_base):
            if type(dfnum) in (float, int):
                dfnum = cupy.asarray(dfnum, numpy.float64)
            else:
                raise TypeError('dfnum is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            dfnum = dfnum.astype('d', copy=False)

        if not isinstance(dfden, _ndarray_base):
            if type(dfden) in (float, int):
                dfden = cupy.asarray(dfden, numpy.float64)
            else:
                raise TypeError('dfden is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            dfden = dfden.astype('d', copy=False)

        if size is None:
            size = cupy.broadcast(dfnum, dfden).shape

        y = (self.chisquare(dfnum, size) * dfden) / (
            self.chisquare(dfden, size) * dfnum)
        return y

    def geometric(self, p, size=None):
        """Geometric distribution.

        Returns an array of samples drawn from the geometric distribution. Its
        probability mass function is defined as

        .. math::
            f(x) = p(1-p)^{k-1}.

        Args:
            p (float or cupy.ndarray of floats): Success probability of
                the geometric distribution.
            size (int or tuple of ints, optional): The shape of the output
                array. If ``None`` (default), a single value is returned if
                ``p`` is scalar. Otherwise, ``p.size`` samples are drawn.

        Returns:
            cupy.ndarray: Samples drawn from the geometric distribution.

        .. seealso::
            :meth:`numpy.random.Generator.geometric`
        """
        cdef _ndarray_base y
        cdef _ndarray_base p_a

        if not isinstance(p, _ndarray_base):
            if type(p) in (float, int):
                p_a = _core.ndarray((), numpy.float64)
                p_a.fill(p)
                p = p_a
            else:
                raise TypeError('p is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            p = p.astype('d', copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size,)
        elif size is None:
            size = p.shape
        y = _core.ndarray(size if size is not None else (), numpy.int64)

        p = cupy.broadcast_to(p, y.shape)

        _launch_dist(self.bit_generator, geometric, y, (p,))
        return y

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """Hypergeometric distribution.

        Returns an array of samples drawn from the hypergeometric distribution.
        Its probability mass function is defined as

        .. math::
            f(x) = \\frac{\\binom{m}{n}\\binom{N-m}{n-x}}{\\binom{N}{n}}.

        Args:
            ngood (int or array_like of ints): Parameter of the hypergeometric
                distribution :math:`n`.
            nbad (int or array_like of ints): Parameter of the hypergeometric
                distribution :math:`m`.
            nsample (int or array_like of ints): Parameter of the
                hypergeometric distribution :math:`N`.
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.

        Returns:
            cupy.ndarray: Samples drawn from the hypergeometric distribution.

        .. seealso::
            :meth:`numpy.random.Generator.hypergeometric`
        """
        cdef _ndarray_base y
        cdef _ndarray_base ngood_a
        cdef _ndarray_base nbad_a
        cdef _ndarray_base nsample_a

        if not isinstance(ngood, _ndarray_base):
            if type(ngood) in (float, int):
                ngood_a = _core.ndarray((), numpy.int64)
                ngood_a.fill(ngood)
                ngood = ngood_a
            else:
                raise TypeError('ngood is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            ngood = ngood.astype(numpy.int64, copy=False)

        if not isinstance(nbad, _ndarray_base):
            if type(nbad) in (float, int):
                nbad_a = _core.ndarray((), numpy.int64)
                nbad_a.fill(nbad)
                nbad = nbad_a
            else:
                raise TypeError('nbad is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            nbad = nbad.astype(numpy.int64, copy=False)

        if not isinstance(nsample, _ndarray_base):
            if type(nsample) in (float, int):
                nsample_a = _core.ndarray((), numpy.int64)
                nsample_a.fill(nsample)
                nsample = nsample_a
            else:
                raise TypeError('nsample is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            nsample = nsample.astype(numpy.int64, copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size,)
        if size is None:
            size = cupy.broadcast(ngood, nbad, nsample).shape
        y = _core.ndarray(size, numpy.int64)

        ngood = cupy.broadcast_to(ngood, y.shape)
        nbad = cupy.broadcast_to(nbad, y.shape)
        nsample = cupy.broadcast_to(nsample, y.shape)

        _launch_dist(self.bit_generator, hypergeometric, y,
                     (ngood, nbad, nsample))
        return y

    def logseries(self, p, size=None):
        """Log series distribution.

        Returns an array of samples drawn from the log series distribution.
        Its probability mass function is defined as

        .. math::
           f(x) = \\frac{-p^x}{x\\ln(1-p)}.

        Args:
            p (float or cupy.ndarray of floats): Parameter of the log series
                distribution. Must be in the range (0, 1).
            size (int or tuple of ints, optional): The shape of the output
                array. If ``None`` (default), a single value is returned if
                ``p`` is scalar. Otherwise, ``p.size`` samples are drawn.

        Returns:
            cupy.ndarray: Samples drawn from the log series distribution.

        .. seealso::
            :meth:`numpy.random.Generator.logseries`
        """
        cdef _ndarray_base y

        if not isinstance(p, _ndarray_base):
            if type(p) in (float, int):
                p = cupy.asarray(p, numpy.float64)
            else:
                raise TypeError('p is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            p = p.astype('d', copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size,)
        elif size is None:
            size = p.shape

        y = _core.ndarray(size, numpy.int64)

        p = cupy.broadcast_to(p, y.shape)
        _launch_dist(self.bit_generator, logseries, y, (p,))
        return y

    def standard_exponential(
            self, size=None, dtype=numpy.float64,
            method='inv', out=None):
        """Standard exponential distribution.

        Returns an array of samples drawn from the standard exponential
        distribution. Its probability density function is defined as

          .. math::
             f(x) = e^{-x}.

        Args:
            size (int or tuple of ints): The shape of the array. If ``None``,
                a zero-dimensional array is generated.
            dtype: Data type specifier. Only :class:`numpy.float32` and
                :class:`numpy.float64` types are allowed.
            method (str): Method to sample. Currently only ``'inv'``, sampling
                from the default inverse CDF, is supported.
            out (cupy.ndarray, optional): If specified, values will be written
                to this array
        Returns:
            cupy.ndarray: Samples drawn from the standard exponential
            distribution.

        .. seealso::
            :meth:`numpy.random.Generator.standard_exponential`
        """
        cdef _ndarray_base y

        if method == 'zig':
            raise NotImplementedError('Ziggurat method is not supported')

        if out is not None:
            self._check_output_array(dtype, size, out)

        y = _core.ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, exponential, y, ())
        if out is not None:
            _core.elementwise_copy(y, out)
            y = out
        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)

    def poisson(self, lam=1.0, size=None):
        """Poisson distribution.

        Returns an array of samples drawn from the poisson distribution. Its
        probability mass function is defined as

        .. math::
            f(x) = \\frac{\\lambda^xe^{-\\lambda}}{x!}.

        Args:
            lam (float or array_like of floats): Parameter of
                the poisson distribution
                :math:`\\lambda`.
            size (int or tuple of ints): The shape of the array. If ``None``,
                this function generate an array whose shape is ``lam.shape``.

        Returns:
            cupy.ndarray: Samples drawn from the poisson distribution.

        .. seealso::
            :meth:`numpy.random.Generator.poisson`
        """
        cdef _ndarray_base y
        cdef _ndarray_base lam_a

        if not isinstance(lam, _ndarray_base):
            if type(lam) in (float, int):
                lam_a = _core.ndarray((), numpy.float64)
                lam_a.fill(lam)
                lam = lam_a
            else:
                raise TypeError('lam is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            # Check if size is broadcastable to shape
            # but size determines the output
            lam = lam.astype('d', copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size,)
        elif size is None:
            size = lam.shape

        y = _core.ndarray(size if size is not None else (), numpy.int64)

        lam = cupy.broadcast_to(lam, y.shape)
        _launch_dist(self.bit_generator, poisson, y, (lam,))
        return y

    def power(self, a, size=None):
        """Power distribution.

        Returns an array of samples drawn from the power distribution. Its
        probability density function is defined as

        .. math::
           f(x) = ax^{a-1}.

        Args:
            a (float or array_like of floats): Parameter of the power
                distribution :math:`a`.
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.

        Returns:
            cupy.ndarray: Samples drawn from the power distribution.

        .. seealso::
            :meth:`numpy.random.Generator.power`
        """

        if not isinstance(a, _ndarray_base):
            if type(a) in (float, int):
                a = cupy.asarray(a, numpy.float64)
            else:
                raise TypeError('a is required to be a cupy.ndarray'
                                ' or a scalar')
        else:
            a = a.astype('d', copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size, )
        elif size is None:
            size = a.shape

        x = self.standard_exponential(size)
        cupy.exp(-x, out=x)
        cupy.add(1, -x, out=x)
        cupy.power(x, 1./a, out=x)
        return x

    def standard_normal(self, size=None, dtype=numpy.float64, out=None):
        """Standard normal distribution.

        Returns an array of samples drawn from the standard normal
        distribution.

        Args:
            size (int or tuple of ints): The shape of the array. If ``None``, a
                zero-dimensional array is generated.
            dtype: Data type specifier.

            out (cupy.ndarray, optional): If specified, values will be written
                to this array

        Returns:
            cupy.ndarray: Samples drawn from the standard normal distribution.

        .. seealso::
            - :meth:`numpy.random.Generator.standard_normal`
        """
        cdef _ndarray_base y

        if out is not None:
            self._check_output_array(dtype, size, out)
            y = out
        else:
            y = _core.ndarray(size if size is not None else (), dtype)

        if y.dtype.char not in ('f', 'd'):
            raise TypeError(
                f'Unsupported dtype {y.dtype.name} for standard_normal')

        if y.dtype.char == 'd':
            _launch_dist(self.bit_generator, standard_normal, y, ())
        else:
            _launch_dist(self.bit_generator, standard_normal_float, y, ())

        return y

    def gamma(self, shape, scale=1.0, size=None):
        """Gamma distribution.

        Returns an array of samples drawn from the gamma distribution. Its
        probability density function is defined as

        .. math::
           f(x) = \\frac{1}{\\Gamma(k)\\theta^k}x^{k-1}e^{-x/\\theta}.


        Args:
            shape (float or array_like of float): The shape of the
                gamma distribution.  Must be non-negative.
            scale (float or array_like of float): The scale of the
                gamma distribution.  Must be non-negative.
                Default equals to 1
            size (int or tuple of ints): The shape of the array.
                If ``None``, a zero-dimensional array is generated.

        .. seealso::
            - :meth:`numpy.random.Generator.gamma`
        """
        if size is None:
            size = cupy.broadcast(shape, scale).shape
        y = self.standard_gamma(shape, size)
        y *= scale
        return y

    def standard_gamma(self, shape, size=None, dtype=numpy.float64, out=None):
        """Standard gamma distribution.

        Returns an array of samples drawn from the standard gamma distribution.
        Its probability density function is defined as

        .. math::
           f(x) = \\frac{1}{\\Gamma(k)}x^{k-1}e^{-x}.


        Args:
            shape (float or array_like of float): The shape of the
                gamma distribution.  Must be non-negative.
            size (int or tuple of ints): The shape of the array.
                If ``None``, a zero-dimensional array is generated.
            dtype: Data type specifier.
            out (cupy.ndarray, optional): If specified, values will be written
                to this array

        .. seealso::
            - :meth:`numpy.random.Generator.standard_gamma`
        """
        cdef _ndarray_base y
        cdef _ndarray_base shape_a

        if not isinstance(shape, _ndarray_base):
            if type(shape) in (float, int):
                shape_a = _core.ndarray((), numpy.float64)
                shape_a.fill(shape)
                shape = shape_a
            else:
                if shape is None:
                    raise TypeError('shape must be real number, not NoneType')
                raise ValueError('shape is required to be a cupy.ndarray'
                                 ' or a scalar')
        else:
            # Check if size is broadcastable to shape
            # but size determines the output
            shape = shape.astype('d', copy=False)

        if size is not None and not isinstance(size, tuple):
            size = (size,)
        elif size is None:
            size = shape.shape if out is None else out.shape

        y = None
        if out is not None:
            self._check_output_array(dtype, size, out, True)
            if out.dtype.char == 'd':
                y = out

        if y is None:
            y = _core.ndarray(size if size is not None else (), numpy.float64)

        if numpy.dtype(dtype).char not in ('f', 'd'):
            raise TypeError(
                f'Unsupported dtype {y.dtype.name} for standard_gamma')

        shape = cupy.broadcast_to(shape, y.shape)

        _launch_dist(self.bit_generator, standard_gamma, y, (shape,))
        if out is not None and y is not out:
            _core.elementwise_copy(y, out)
            y = out
        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)

    def binomial(self, n, p, size=None):
        """Binomial distribution.

        Returns an array of samples drawn from the binomial distribution. Its
        probability mass function is defined as

        .. math::
           f(x) = \\binom{n}{x}p^x(1-p)^(n-x).

        Args:
            n (int or cupy.ndarray of ints): Parameter of the distribution,
                >= 0. Floats are also accepted, but they will be truncated to
                integers.
            p (float or cupy.ndarray of floats): Parameter of the distribution,
                >= 0 and <= 1.
            size (int or tuple of ints, optional): The shape of the output
                array. If ``None`` (default), a single value is returned if
                ``n`` and ``p`` are both scalars. Otherwise,
                ``cupy.broadcast(n, p).size`` samples are drawn.

        Returns:
            cupy.ndarray: Samples drawn from the binomial distribution.

        .. seealso::
           :meth:`numpy.random.Generator.binomial`
        """
        cdef _ndarray_base y
        cdef intptr_t binomial_state_ptr

        if isinstance(n, _ndarray_base):
            n = n.astype(numpy.int64, copy=False)
        elif type(n) in (float, int):
            n = cupy.asarray(n, numpy.int64)
        else:
            raise TypeError('n is required to be a cupy.ndarray or a scalar')

        if isinstance(p, _ndarray_base):
            p = p.astype(numpy.float64, copy=False)
        elif type(p) is float:
            p = cupy.asarray(p, numpy.float64)
        else:
            raise TypeError('p is required to be a cupy.ndarray or a scalar')

        if size is None:
            size = cupy.broadcast(n, p).shape

        y = _core.ndarray(size if size is not None else (), numpy.int64)

        n = cupy.broadcast_to(n, y.shape)
        p = cupy.broadcast_to(p, y.shape)

        if self._binomial_state is None:
            state_size = self.bit_generator._state_size()
            self._binomial_state = cupy.zeros(
                sizeof(rk_binomial_state) * state_size, dtype=numpy.int8)
        binomial_state_ptr = <intptr_t>self._binomial_state.data.ptr
        _launch_dist(
            self.bit_generator, binomial, y,
            (n, p, binomial_state_ptr))
        return y


def init_curand(generator, state, seed, size):
    init_curand_generator(
        <int>generator,
        <intptr_t>state,
        <uint64_t>seed,
        <uint64_t>size,
        <intptr_t>stream.get_current_stream_ptr()
    )


def random_raw(generator, out):
    _launch_dist(generator, raw, out, ())


cdef void _launch(
        func, int generator, intptr_t state, intptr_t strm,
        int bsize, out, args):
    cdef ssize_t size = out.size
    if size == 0:
        # Avoid issues launching empty grids in CUDA 10.2
        return
    nargs = [
        _array_data(a)
        if isinstance(a, cupy.ndarray) else a for a in args]
    args_ptr = [
        <intptr_t>a.data.ptr
        if isinstance(a, cupy.ndarray) else a for a in nargs]

    func(generator, state, bsize,
         <intptr_t>out.data.ptr, size, strm, *args_ptr)


cdef void _launch_dist(bit_generator, func, out, args) except*:
    cdef intptr_t strm = stream.get_current_stream_ptr()
    cdef intptr_t state = <intptr_t>bit_generator.state()
    cdef int generator = bit_generator.generator
    cdef bsize = bit_generator._state_size()
    _launch(func, generator, state, strm, bsize, out, args)
