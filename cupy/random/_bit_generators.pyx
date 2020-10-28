import threading

import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t

import cupy
from cupy.cuda import curand
from cupy.core.core cimport ndarray


cdef extern from 'cupy_distributions.cuh' nogil:
    cppclass curandState:
        pass
    void init_xor_generator(intptr_t state_ptr, uint64_t seed, ssize_t size);
    void interval_32(intptr_t state, int mx, int mask, intptr_t out, ssize_t size);
    void interval_64(intptr_t state, uint64_t mx, uint64_t mask, intptr_t out, ssize_t size);
    void beta(intptr_t state, double a, double b, intptr_t out, ssize_t size);
    void standard_exponential(intptr_t state, intptr_t out, ssize_t size);

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
    # Size is the number of threads that will be initialized
    def __init__(self, seed=None, size=1024*100):
        super().__init__(seed)
        self._seed = self._seed_seq.generate_state(1, numpy.uint64)[0]
        cdef ssize_t _size = sizeof(curandState) * size
        self._state = cupy.zeros(_size, dtype=numpy.int8)
        ptr = self._state.data.ptr
        cdef intptr_t state_ptr = <intptr_t>ptr
        cdef uint64_t c_seed = <uint64_t>self._seed
        # Initialize the state
        init_xor_generator(state_ptr, self._seed, size)
        print('state is',self._state)

    def random_raw(self, size=None, out=False):
        pass

    def state(self):
        return self._state.data.ptr

class Generator:
    def __init__(self, bit_generator):
        self._bit_generator = bit_generator 

    def integers(self, low, high, size, dtype=numpy.int32, endpoint=False):
        cdef ndarray y
        cdef intptr_t state
        cdef intptr_t y_ptr

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
        y_ptr = <intptr_t>y.data.ptr

        state_ptr = self._bit_generator.state()
        state = <intptr_t>state_ptr

        if dtype is numpy.uint32:
            # We know that the mask fits
            interval_32(state, diff, <uint32_t>mask, y_ptr, y.size)        
        else:
            # we will only try and check the upper part
            interval_64(state, diff, mask, y_ptr, y.size)        
        return low + y

    def beta(self, a, b, size=None, dtype=float):
        """Returns an array of samples drawn from the beta distribution.

        .. seealso::
            :func:`cupy.random.beta` for full documentation,
            :meth:`numpy.random.RandomState.beta
            <numpy.random.mtrand.RandomState.beta>`
        """
        cdef ndarray y
        # cdef uint64_t state = <uint64_t>self._bit_generator.state()
        cdef intptr_t state
        cdef intptr_t y_ptr

        state_ptr = self._bit_generator.state()
        state = <intptr_t>state_ptr

        y = ndarray(size if size is not None else (), dtype)
        y_ptr = <intptr_t>y.data.ptr
        beta(state, a, b, y_ptr, y.size)        
        return y

    def standard_exponential(self, size=None, dtype=numpy.float64, method='inv', out=None):
        cdef ndarray y
        cdef intptr_t state
        cdef intptr_t y_ptr

        if method == 'zig':
            raise NotImplementedError('Ziggurat method is not supported')
                 
        state_ptr = self._bit_generator.state()
        state = <intptr_t>state_ptr

        y = ndarray(size if size is not None else (), dtype)
        y_ptr = <intptr_t>y.data.ptr
        print(state_ptr)
        standard_exponential(state, y_ptr, y.size)
        if out is not None:
            out[...] = y
            y = out
        return y

