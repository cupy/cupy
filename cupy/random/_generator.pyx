# distutils: language = c++
import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t

import cupy
from cupy.core.core cimport ndarray
from cupy.random._distributions_module import _get_distribution


_UINT32_MAX = 0xffffffff
_UINT64_MAX = 0xffffffffffffffff


class Generator:
    def __init__(self, bit_generator):
        self.bit_generator = bit_generator

    def integers(
            self, low, high=None, size=None,
            dtype=numpy.int64, endpoint=False):
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
        """Returns an array of samples drawn from the beta distribution.

        .. seealso::
            :func:`cupy.random.beta` for full documentation,
            :meth:`numpy.random.RandomState.beta
            <numpy.random.mtrand.RandomState.beta>`
        """
        cdef ndarray y
        y = ndarray(size if size is not None else (), numpy.float64)
        _launch_dist(self.bit_generator, 'beta', y, a, b)
        return y.astype(dtype)

    def standard_exponential(
            self, size=None, dtype=numpy.float64,
            method='inv', out=None):
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
    cdef bsize = bit_generator.state_size()

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
