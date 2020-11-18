# distutils: language = c++
import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t

import cupy
from cupy.core.core cimport ndarray
from cupy.random._distributions_module import _get_distribution


_UINT32_MAX = 0xffffffff
_UINT64_MAX = 0xffffffffffffffff


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
            _launch_dist(self.bit_generator, 'interval_32', y, diff, mask)
        else:
            _launch_dist(self.bit_generator, 'interval_64', y, diff, mask)
        return (lo + y).astype(dtype)

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
            :meth:`numpy.random.Generator.beta
            <numpy.random.generator.Generator.beta>`
        """
        cdef ndarray y
        y = ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, 'beta', y, a, b)
        return y.astype(dtype)

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

        .. seealso:: :meth:`numpy.random.standard_exponential
                     <numpy.random.mtrand.RandomState.standard_exponential>`
        """
        cdef ndarray y

        if method == 'zig':
            raise NotImplementedError('Ziggurat method is not supported')

        y = ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, 'exponential', y)
        if out is not None:
            out[...] = y
            y = out
        return y.astype(dtype)


def _launch_dist(bit_generator, kernel_name, out, *args):
    kernel = _get_distribution(bit_generator, kernel_name)
    state_ptr = bit_generator.state()
    cdef state = <intptr_t>state_ptr
    cdef y_ptr = <intptr_t>out.data.ptr
    cdef ssize_t size = out.size
    cdef ndarray chunk
    cdef bsize = bit_generator._state_size()

    tpb = 256
    if out.shape == () or bsize == 0:
        bpg = (size + tpb - 1) // tpb
        kernel((bpg,), (tpb,), (state, y_ptr, size, *args))
    else:
        chunks = (out.size + bsize - 1) // bsize
        for i in range(chunks):
            chunk = out[i*bsize:]
            bpg = (bsize + tpb - 1) // tpb
            y_ptr = <intptr_t>chunk.data.ptr
            k_args = (state, y_ptr, min(bsize, chunk.size), ) + args
            kernel((bpg,), (tpb,), k_args)
