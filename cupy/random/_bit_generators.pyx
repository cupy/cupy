import threading

import numpy

import cupy
from cupy.cuda import curand

cdef extern from 'cupy_distributions.h' nogil:
    void standard_exponential();


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


class MT19937(BitGenerator):
    # Note that this can't be used to generate distributions inside the device
    # as there is no support for curand device api
    def __init__(self, seed=None):
        super().__init__(seed)
        method = cupy.cuda.curand.CURAND_RNG_PSEUDO_MT19937
        self._generator = curand.createGenerator(method)
        self._seed = self._seed_seq.generate_state(1, numpy.int64)[0]
        curand.setPseudoRandomGeneratorSeed(self._generator, self._seed)

    def random_raw(self, size=None, out=False):
        if size is None:
            size = ()
        sample = cupy.empty(size, dtype=cupy.uint32)
        # cupy.random only uses int32 random generator
        size_in_int = sample.dtype.itemsize // 4
        curand.generate(
            self._generator, sample.data.ptr, sample.size * size_in_int)
        return sample.astype(cupy.uint64)

    def jumped(self, jumps=1):
       # advances the state as-if 2^{128} random numbers have been generated
       # curand cant offset the mt19937 sequence.
       raise RuntimeError(
           'MT19937 does not currently support offsetting the generator')

class Generator:
    def __init__(self, bit_generator):
        self._bit_generator = bit_generator 

