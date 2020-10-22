import threading

import cupy
from cupy.cuda import curand

cdef extern from 'cupy_distributions.h' nogil:
    void test();


class BitGenerator:
    def __init__(self, seed=None):
        self.lock = threading.Lock()
        # If None, then fresh, unpredictable entropy will be pulled from the OS.
        # If an int or array_like[ints] is passed, then it will be passed 
        # to ~`numpy.random.SeedSequence` to derive the initial BitGenerator state.
        self._seed = seed

    def random_raw(self, size=None, out=False):
        raise NotImplementedError(
            'Subclasses of `BitGenerator` must override `random_raw`')


class MT19937(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed)
        # Lets get the generator
        method = cupy.cuda.curand.CURAND_RNG_PSEUDO_MT19937
        self._generator = curand.createGenerator(method)

    def random_raw(self, size=None, out=False):
        if size is None:
            size = ()
        sample = cupy.empty(size, dtype=cupy.uint32)
        # cupy.random only uses int32 random generator
        size_in_int = sample.dtype.itemsize // 4
        curand.generate(
            self._generator, sample.data.ptr, sample.size * size_in_int)
        return sample.astype(cupy.uint64)

def call_test():
    test()
