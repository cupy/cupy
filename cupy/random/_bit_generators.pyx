import threading

import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t

import cupy
from cupy.cuda import curand
from cupy.core.core cimport ndarray


cdef extern from 'cupy_distributions.h' nogil:
    void beta(intptr_t param, double a, double b, void* out, ssize_t size);
    void interval_32(intptr_t param, int mx, int mask, void* out, ssize_t size);
    void interval_64(intptr_t param, uint64_t mx, uint64_t mask, void* out, ssize_t size);
    void standard_exponential(intptr_t seed, void* out, ssize_t size);

_UINT32_MAX = 0xffffffff
_UINT64_MAX = 0xffffffffffffffff

class BitGenerator:
    def __init__(self, seed=None):
        self.lock = threading.Lock()
        # If None, then fresh, unpredictable entropy will be pulled from the OS.
        # If an int or array_like[ints] is passed, then it will be passed 
        # to ~`numpy.random.SeedSequence` to derive the initial BitGenerator state.
        # TODO(ecastill) port SeedSequence
        self._seed_seq = numpy.random.SeedSequence(seed)

    def random_raw(self, size=None, out=False):
        raise NotImplementedError(
            'Subclasses of `BitGenerator` must override `random_raw`')


class XORWOW(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed)
        self._seed = self._seed_seq.generate_state(1, numpy.uint64)[0]

    def random_raw(self, size=None, out=False):
        pass

class Generator:
    def __init__(self, bit_generator):
        self._bit_generator = bit_generator 

    def integers(self, low, high, size, dtype=numpy.int32, endpoint=False):
        cdef ndarray y
        cdef uint64_t param = <uint64_t>self._bit_generator._seed
        cdef void* y_ptr

        diff = high-low
        if not endpoint:
           diff -= 1

        cdef uint64_t mask = (1 << diff.bit_length()) - 1
        # TODO adjust dtype
        if diff <= _UINT32_MAX:
            dtype = numpy.uint32
        elif diff <= _UINT64_MAX:
            dtype = numpy.uint64
        else:
            raise ValueError(
                'high - low must be within uint64 range (actual: {})'.format(diff))

        y = ndarray(size if size is not None else (), dtype)
        y_ptr = <void *>y.data.ptr
        if dtype is numpy.uint32:
            # We know that the mask fits
            interval_32(<intptr_t>param, diff, <uint32_t>mask, y_ptr, y.size)        
        else:
            # we will only try and check the upper part
            interval_64(<intptr_t>param, diff, mask, y_ptr, y.size)        
        return low + y

    def beta(self, a, b, size=None, dtype=float):
        """Returns an array of samples drawn from the beta distribution.

        .. seealso::
            :func:`cupy.random.beta` for full documentation,
            :meth:`numpy.random.RandomState.beta
            <numpy.random.mtrand.RandomState.beta>`
        """
        cdef ndarray y
        cdef uint64_t param = <uint64_t>self._bit_generator._seed
        cdef void* y_ptr

        y = ndarray(size if size is not None else (), dtype)
        y_ptr = <void *>y.data.ptr
        beta(<intptr_t>param, a, b, y_ptr, y.size)        
        return y

    def standard_exponential(self, size=None, dtype=numpy.float64, method='inv', out=None):
        cdef ndarray y
        cdef uint64_t param = <uint64_t>self._bit_generator._seed
        cdef void* y_ptr

        if method == 'zig':
            raise NotImplementedError('Ziggurat method is not supported')
                 
        y = ndarray(size if size is not None else (), dtype)
        y_ptr = <void *>y.data.ptr

        standard_exponential(<intptr_t>param, y_ptr, y.size)        
        if out is not None:
            out[...] = y
            y = out
        return y

