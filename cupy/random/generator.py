import atexit
import binascii
import collections
import os
import time

import numpy

import cupy
from cupy import cuda
from cupy.cuda import curand
from cupy import elementwise


class RandomState(object):

    """Portable container of a pseudo-random number generator.

    An instance of this class holds the state of a random number generator. The
    state is available only on the device which has been current at the
    initialization of the instance.

    Functions of :mod:`cupy.random` use global instances of this class.
    Different instances are used for different devices. The global state for
    the current device can be obtained by the
    :func:`cupy.random.get_random_state` function.

    Args:
        seed (None or int): Seed of the random number generator. See the
            :meth:`~cupy.random.RandomState.seed` method for detail.
        method (int): Method of the random number generator. Following values
            are available::

               cupy.cuda.curand.CURAND_RNG_PSEUDO_DEFAULT
               cupy.cuda.curand.CURAND_RNG_XORWOW
               cupy.cuda.curand.CURAND_RNG_MRG32K3A
               cupy.cuda.curand.CURAND_RNG_MTGP32
               cupy.cuda.curand.CURAND_RNG_MT19937
               cupy.cuda.curand.CURAND_RNG_PHILOX4_32_10

    """
    def __init__(self, seed=None, method=curand.CURAND_RNG_PSEUDO_DEFAULT):
        self._generator = curand.createGenerator(method)
        self.seed(seed)

    def __del__(self):
        curand.destroyGenerator(self._generator)

    def set_stream(self, stream=None):
        if stream is None:
            stream = cuda.Stream()
        curand.setStream(self._generator, stream.ptr)

    # NumPy compatible functions

    def lognormal(self, mean=0.0, sigma=1.0, size=None, dtype=float,
                  allocator=cuda.alloc):
        """Returns an array of samples drawn from a log normal distribution.

        .. seealso::
            :func:`cupy.random.lognormal` for full documentation,
            :meth:`numpy.random.RandomState.lognormal`

        """
        dtype = _check_and_get_dtype(dtype)
        size = _get_size(size)
        out = cupy.empty(size, dtype=dtype, allocator=allocator)
        if dtype.type == numpy.float32:
            curand.generateLogNormal(self._generator, out._fptr, out.size,
                                     mean, sigma)
        else:
            curand.generateLogNormalDouble(self._generator, out._fptr,
                                           out.size, mean, sigma)
        return out

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=float,
               allocator=cuda.alloc):
        """Returns an array of normally distributed samples.

        .. seealso::
            :func:`cupy.random.normal` for full documentation,
            :meth:`numpy.random.RandomState.normal`

        """
        dtype = _check_and_get_dtype(dtype)
        size = _get_size(size)
        out = cupy.empty(size, dtype=dtype, allocator=allocator)
        if dtype.type == numpy.float32:
            curand.generateNormal(self._generator, out._fptr, out.size, loc,
                                  scale)
        else:
            curand.generateNormalDouble(self._generator, out._fptr, out.size,
                                        loc, scale)
        return out

    def rand(self, *size, **kwarg):
        """Returns uniform random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`cupy.random.rand` for full documentation,
            :meth:`numpy.random.RandomState.rand`

        """
        dtype = kwarg.pop('dtype', float)
        allocator = kwarg.pop('allocator', cuda.alloc)
        if kwarg:
            raise TypeError('rand() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.random_sample(size=size, dtype=dtype, allocator=allocator)

    def randn(self, *size, **kwarg):
        """Returns an array of standand normal random values.

        .. seealso::
            :func:`cupy.random.randn` for full documentation,
            :meth:`numpy.random.RandomState.randn`

        """
        dtype = kwarg.pop('dtype', float)
        allocator = kwarg.pop('allocator', cuda.alloc)
        if kwarg:
            raise TypeError('randn() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.normal(size=size, dtype=dtype, allocator=allocator)

    _1m_kernel = elementwise.ElementwiseKernel(
        '', 'T x', 'x = 1 - x', 'cupy_random_1_minus_x')

    def random_sample(self, size=None, dtype=float, allocator=cuda.alloc):
        """Returns an array of random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`cupy.random.random_sample` for full documentation,
            :meth:`numpy.random.RandomState.random_sample`

        """
        dtype = _check_and_get_dtype(dtype)
        size = _get_size(size)
        out = cupy.empty(size, dtype=dtype, allocator=allocator)
        if dtype.type == numpy.float32:
            curand.generateUniform(self._generator, out._fptr, out.size)
        else:
            curand.generateUniformDouble(self._generator, out._fptr, out.size)
        RandomState._1m_kernel(out)
        return out

    def seed(self, seed=None):
        """Resets the state of the random number generator with a seed.

        ..seealso::
            :func:`cupy.random.seed` for full documentation,
            :meth:`numpy.random.RandomState.seed`

        """
        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.uint64(seed)

        curand.setPseudoRandomGeneratorSeed(self._generator, seed)

    def standard_normal(self, size=None, dtype=float, allocator=cuda.alloc):
        """Returns samples drawn from the standard normal distribution.

        .. seealso::
            :func:`cupy.random.standard_normal` for full documentation,
            :meth:`numpy.random.RandomState.standard_normal`

        """
        return self.normal(size=size, dtype=dtype, allocator=allocator)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float,
                allocator=cuda.alloc):
        """Returns an array of uniformlly-distributed samples over an interval.

        .. seealso::
            :func:`cupy.random.uniform` for full documentation,
            :meth:`numpy.random.RandomState.uniform`

        """
        size = _get_size(size)
        rand = self.random_sample(size=size, dtype=dtype, allocator=allocator)
        return low + rand * (high - low)


def seed(seed=None):
    """Resets the state of the random number generator with a seed.

    This function resets the state of the global random number generator for
    the current device. Be careful that generators for other devices are not
    affected.

    Args:
        seed (None or int): Seed for the random number generator. If None, it
            uses :func:`os.urandom` if available or :func:`time.clock`
            otherwise. Note that this function does not support seeding by an
            integer array.

    """
    get_random_state().seed(seed)


# CuPy specific functions

_random_states = {}


@atexit.register
def reset_states():
    global _random_states
    _random_states = {}


def get_random_state():
    """Gets the state of the random number generator for the current device.

    If the state for the current device is not created yet, this function
    creates a new one, initializes it, and stores it as the state for the
    current device.

    Returns:
        RandomState: The state of the random number generator for the
        device.

    """
    global _random_states
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        rs = RandomState()
        _random_states[dev.id] = rs
    return rs


def _get_size(size):
    if size is None:
        return ()
    elif isinstance(size, collections.Iterable):
        return tuple(size)
    else:
        return size,


def _check_and_get_dtype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype.type not in (numpy.float32, numpy.float64):
        raise TypeError('cupy.random only supports float32 and float64')
    return dtype
