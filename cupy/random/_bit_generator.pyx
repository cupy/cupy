# distutils: language = c++
import threading

import numpy

from libc.stdint cimport intptr_t, uint64_t, uint32_t

import cupy
from cupy.cuda cimport stream
from cupy.core.core cimport ndarray
from cupy.random._generator import _launch_dist
from cupy.random._distributions_module import _initialize_generator

# We need access to the sizes here, so this is why we have this header
# in here instead of cupy backends
cdef extern from 'device_random.h' nogil:
    cppclass curandState:
        pass
    cppclass curandStateMRG32k3a:
        pass
    cppclass curandStatePhilox4_32_10_t:
        pass


class BitGenerator:
    def __init__(self, seed=None):
        self.lock = threading.Lock()
        # TODO(ecastill) port SeedSequence
        if isinstance(seed, numpy.random.SeedSequence):
            self._seed_seq = seed
        else:
            if isinstance(seed, cupy.ndarray):
                seed = cupy.asnumpy(seed)
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
            raise RuntimeError(
                'This Generator state is allocated in a different device')


class _cuRANDGenerator(BitGenerator):
    # Size is the number of threads that will be initialized
    def __init__(self, seed=None, size=10000*256):
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
        bpg = (size + tpb - 1) // tpb
        _initialize_generator(self)((bpg,), (tpb,), (state_ptr, c_seed, size))

    def random_raw(self, size=None, out=False):
        shape = size if size is not None else ()
        y = cupy.zeros(shape, dtype=numpy.int32)
        _launch_dist(self, 'raw', y)
        return y

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
