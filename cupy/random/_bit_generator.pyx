# distutils: language = c++

import threading

import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t

import cupy
from cupy.cuda import curand
from cupy.cuda cimport stream
from cupy.core.core cimport ndarray
from cupy.random._distributions_module import _get_distribution, _initialize_generator

cdef extern from 'cupy_distributions.cuh' nogil:
    cppclass curandState:
        pass
    cppclass curandStateMRG32k3a:
        pass
    cppclass curandStatePhilox4_32_10_t:
        pass


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
        dev = cupy.cuda.Device()
        self._current_device = dev.id

    def random_raw(self, size=None, out=False):
        raise NotImplementedError(
            'Subclasses of `BitGenerator` must override `random_raw`')

    def state_size(self):
        """Maximum number of samples that can be generated at once
        """
        return 0

    def _check_device(self):
        if cupy.cuda.Device().id != self._current_device:
            raise RuntimeError("This Generator state has been allocated in a different device")


class _cuRANDGenerator(BitGenerator):
    # Size is the number of threads that will be initialized
    def __init__(self, seed=None, size=1024*100):
        super().__init__(seed)
        # Raw kernel has problems with integers with the 64th bit set
        self._seed = self._seed_seq.generate_state(1, numpy.uint32)[0]
        self._size = size
        cdef uint64_t b_size = self._type_size() * size
        self._state = cupy.zeros(b_size, dtype=numpy.int8)
        ptr = self._state.data.ptr
        cdef intptr_t state_ptr = <intptr_t>ptr
        cdef uint64_t c_seed = <uint64_t>self._seed
        cdef intptr_t _strm = stream.get_current_stream_ptr()
        # Initialize the state
        tpb = 256
        bpg =  (size + tpb - 1) // tpb;
        _initialize_generator(self)((bpg,), (tpb,), (state_ptr, c_seed, size))

    def random_raw(self, size=None, out=False):
        pass

    def state(self):
        self._check_device()
        return self._state.data.ptr

    def state_size(self):
        return self._size

    def _type_size(self):
        return 0 


class XORWOW(_cuRANDGenerator):

    def _type_size(self):
        return sizeof(curandState)

    def _c_layer_generator(self):
        return "curand_pseudo_state<curandState>"


class MRG32k3a(_cuRANDGenerator):

    def _type_size(self):
        return sizeof(curandStateMRG32k3a)

    def _c_layer_generator(self):
        return "curand_pseudo_state<curandStateMRG32k3a>"


class Philox4x3210(_cuRANDGenerator):

    def _type_size(self):
        return sizeof(curandStatePhilox4_32_10_t)

    def _c_layer_generator(self):
        return "curand_pseudo_state<curandStatePhilox4_32_10_t>"


class Generator:
    def __init__(self, bit_generator):
        self._bit_generator = bit_generator 

    def integers(self, low, high, size, dtype=numpy.int32, endpoint=False):
        cdef ndarray y

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

        # if dtype is numpy.uint32:
        #    self._launch_distribution_kernel(interval_32, y, diff, mask)
        # else:
        #    self._launch_distribution_kernel(interval_64, y, diff, mask)
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
        y = ndarray(size if size is not None else (), dtype)
        # self._launch_distribution_kernel(beta, y, a, b)
        return y

    def standard_exponential(self, size=None, dtype=numpy.float64, method='inv', out=None):
        cdef ndarray y

        if method == 'zig':
            raise NotImplementedError('Ziggurat method is not supported')
                 
        y = ndarray(size if size is not None else (), dtype)
        self._launch_dist('exponential', y)
        if out is not None:
            out[...] = y
            y = out
        return y

    def _launch_dist(self, kernel_name, out, *args):
        kernel = _get_distribution(self._bit_generator, kernel_name)
        state_ptr = self._bit_generator.state()
        cdef state = <intptr_t>state_ptr
        cdef y_ptr = <intptr_t>out.data.ptr
        cdef ssize_t size = out.size
        cdef ndarray chunk
        cdef bsize = self._bit_generator.state_size()

        tpb = 256
        if bsize == 0:
            bpg =  (size + tpb - 1) // tpb;
            kernel((bpg,), (tpb,), (state, y_ptr, out.size, *args))
        else:
            chunks = (out.size + bsize - 1) // bsize
            for i in range(chunks):
                chunk = out[i*bsize:]
                bpg =  (bsize + tpb - 1) // tpb;
                y_ptr = <intptr_t>chunk.data.ptr
                kernel((bpg,), (tpb,), (state, y_ptr, bsize, *args))
