import atexit
import collections
import os
import time

import numpy
import six

import cupy
from cupy import cuda
from cupy.cuda import curand


class RandomState(object):

    def __init__(self, seed=None, method=curand.CURAND_RNG_PSEUDO_DEFAULT,
                 dtype=numpy.float64, allocator=cuda.alloc):
        self.float_type = numpy.dtype(dtype)
        self.allocator = allocator

        self._generator = curand.createGenerator(method)
        self.seed(seed)

    def __del__(self):
        curand.destroyGenerator(self._generator)

    def set_stream(self, stream=None):
        if stream is None:
            stream = cuda.Stream()
        curand.setStream(self._generator, stream.ptr)

    @property
    def float_type(self):
        return self._float_type

    @float_type.setter
    def float_type(self, dtype):
        dtype = numpy.dtype(dtype)
        if dtype not in (numpy.dtype('f'), numpy.dtype('d')):
            raise TypeError('cupy.random only supports float32 and float64.')
        self._float_type = dtype

    # NumPy compatible functions

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        size = _get_size(size)
        out = cupy.empty(size, dtype=self.float_type, allocator=self.allocator)
        if out.itemsize == 4:
            curand.generateLogNormal(self._generator, out._fptr, out.size,
                                     mean, sigma)
        else:
            curand.generateLogNormalDouble(self._generator, out._fptr,
                                           out.size, mean, sigma)
        return out

    def normal(self, loc=0.0, scale=1.0, size=None):
        size = _get_size(size)
        out = cupy.empty(size, dtype=self.float_type, allocator=self.allocator)
        if out.itemsize == 4:
            curand.generateNormal(self._generator, out._fptr, out.size, loc,
                                  scale)
        else:
            curand.generateNormalDouble(self._generator, out._fptr, out.size,
                                        loc, scale)
        return out

    def rand(self, *size):
        return self.random_sample(size=size)

    def randn(self, *size):
        return self.normal(size=size)

    def random_sample(self, size=None):
        out = cupy.empty(size, dtype=self.float_type, allocator=self.allocator)
        if out.itemsize == 4:
            curand.generateUniform(self._generator, out._fptr, out.size)
        else:
            curand.generateUniformDouble(self._generator, out._fptr, out.size)
        return out

    def seed(self, seed=None):
        if seed is None:
            try:
                seed_str = os.urandom(8)
                seed = numpy.uint64(long(seed_str.encode('hex'), 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.uint64(seed)

        curand.setPseudoRandomGeneratorSeed(self._generator, seed)

    def standard_normal(self, size=None):
        return self.normal(size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        size = _get_size(size)
        rand = self.rand(*size)
        return low + rand * (high - low)


def seed(seed=None):
    get_random_state().seed(seed)


# CuPy specific functions

_float_type = numpy.float64
_random_states = {}


def set_float_type(dtype, for_all_devices=True):
    if for_all_devices:
        global _float_type, _random_states
        for rs in six.itervalues(_random_states):
            rs.float_type = dtype
        _float_type = dtype
    else:
        rs = get_random_state()
        rs.float_type = dtype


@atexit.register
def reset_states():
    global _random_states
    _random_states = {}


def get_random_state():
    global _random_states
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        rs = RandomState(dtype=_float_type)
        _random_states[dev.id] = rs
    return rs


def _get_size(size):
    if size is None:
        return ()
    elif isinstance(size, collections.Iterable):
        return tuple(size)
    else:
        return size,
