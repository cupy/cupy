# distutils: language = c++
import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t, int32_t, int64_t

import cupy
from cupy.cuda cimport stream
from cupy._core.core cimport ndarray
from cupy._core cimport internal


_UINT32_MAX = 0xffffffff
_UINT64_MAX = 0xffffffffffffffff

cdef extern from 'cupy_distributions.cuh' nogil:
    void init_curand_generator(
        int generator, intptr_t state_ptr, uint64_t seed,
        ssize_t size, intptr_t stream)
    void random_uniform(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream)
    void raw(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream)
    void interval_32(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream, int32_t mx, int32_t mask)
    void interval_64(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream, int64_t mx, int64_t mask)
    void beta(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream, double a, double b)
    void exponential(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream)
    void poisson(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream, double lam)
    void standard_normal(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream)
    void standard_normal_float(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream)
    void standard_gamma(
        int generator, intptr_t state, intptr_t out,
        ssize_t size, intptr_t stream, intptr_t shape)


cdef ndarray _array_data(ndarray x):
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
        self.bit_generator = bit_generator

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
        cdef ndarray y

        if out is not None:
            self._check_output_array(dtype, size, out)

        y = ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, random_uniform, y, ())
        if out is not None:
            out[...] = y
            y = out
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
        cdef ndarray y
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
        if not endpoint:
            diff -= 1

        cdef uint64_t mask = (1 << diff.bit_length()) - 1
        # TODO adjust dtype
        if diff <= _UINT32_MAX:
            pdtype = numpy.uint32
        elif diff <= _UINT64_MAX:
            pdtype = numpy.uint64
        else:
            raise ValueError(
                f'high - low must be within uint64 range (actual: {diff})')

        y = ndarray(size if size is not None else (), pdtype)

        if dtype is numpy.uint32:
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
        cdef ndarray y
        y = ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, beta, y, (a, b))
        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)

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
            method (str): Method to sample, Currently onlu 'inv', sample from
                the default inverse CDF is supported.
            out (cupy.ndarray, optional): If specified, values will be written
                to this array
        Returns:
            cupy.ndarray: Samples drawn from the standard exponential
                distribution.

        .. seealso::
            :meth:`numpy.random.Generator.standard_exponential`
        """
        cdef ndarray y

        if method == 'zig':
            raise NotImplementedError('Ziggurat method is not supported')

        if out is not None:
            self._check_output_array(dtype, size, out)

        y = ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, exponential, y, ())
        if out is not None:
            out[...] = y
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
            lam (array_like of floats): Parameter of the poisson distribution
                :math:`\\lambda`.
            size (int or tuple of ints): The shape of the array. If ``None``,
            this function generate an array whose shape is `lam.shape`.

        Returns:
            cupy.ndarray: Samples drawn from the poisson distribution.

        .. seealso::
            :meth:`numpy.random.Generator.poisson`
        """
        cdef ndarray y
        y = ndarray(size if size is not None else (), numpy.int64)
        _launch_dist(self.bit_generator, poisson, y, (lam, ))
        return y

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
        cdef ndarray y

        if out is not None:
            self._check_output_array(dtype, size, out)
            y = out
        else:
            y = ndarray(size if size is not None else (), dtype)

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
        cdef ndarray y
        cdef ndarray shape_arr

        if not isinstance(shape, ndarray):
            if type(shape) in (float, int):
                shape_a = ndarray(1, numpy.float64)
                shape_a.fill(shape)
                shape = shape_a
            else:
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
            y = ndarray(size if size is not None else (), numpy.float64)

        if numpy.dtype(dtype).char not in ('f', 'd'):
            raise TypeError(
                f'Unsupported dtype {y.dtype.name} for standard_gamma')

        shape = cupy.broadcast_to(shape, y.shape)
        shape_arr = _array_data(shape)
        shape_ptr = shape_arr.data.ptr

        _launch_dist(self.bit_generator, standard_gamma, y, (shape_ptr,))
        if out is not None and y is not out:
            out[...] = y
            y = out
        # we cast the array to a python object because
        # cython cant call astype with the default values for
        # omitted args.
        return (<object>y).astype(dtype, copy=False)


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


cdef void _launch_dist(bit_generator, func, out, args) except*:
    # The generator might only have state for a few number of threads,
    # what we do is to split the array filling in several chunks that are
    # generated sequentially using the same state
    cdef intptr_t strm = stream.get_current_stream_ptr()
    state_ptr = bit_generator.state()
    cdef state = <intptr_t>state_ptr
    cdef y_ptr = <intptr_t>out.data.ptr
    cdef ssize_t size = out.size
    cdef ndarray chunk
    cdef int generator = bit_generator.generator

    cdef bsize = bit_generator._state_size()
    if out.shape == () or bsize == 0:
        func(generator, state, y_ptr, out.size, strm, *args)
    else:
        chunks = (out.size + bsize - 1) // bsize
        for i in range(chunks):
            chunk = out[i*bsize:]
            y_ptr = <intptr_t>chunk.data.ptr
            func(generator, state, y_ptr, min(bsize, chunk.size), strm, *args)
