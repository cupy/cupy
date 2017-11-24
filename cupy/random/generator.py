import atexit
import binascii
import functools
import operator
import os
import time

import numpy
import six

import cupy
from cupy import core
from cupy import cuda
from cupy.cuda import curand


_gumbel_kernel = None


def _get_gumbel_kernel():
    global _gumbel_kernel
    if _gumbel_kernel is None:
        _gumbel_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = loc - log(-log(1 - x)) * scale',
            'gumbel_kernel'
        )
    return _gumbel_kernel


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
        # When createGenerator raises an error, _generator is not initialized
        if hasattr(self, '_generator'):
            curand.destroyGenerator(self._generator)

    def set_stream(self, stream=None):
        if stream is None:
            stream = cuda.Stream()
        curand.setStream(self._generator, stream.ptr)

    def _generate_normal(self, func, size, dtype, *args):
        # curand functions below don't support odd size.
        # * curand.generateNormal
        # * curand.generateNormalDouble
        # * curand.generateLogNormal
        # * curand.generateLogNormalDouble
        size = core.get_size(size)
        element_size = six.moves.reduce(operator.mul, size, 1)
        if element_size % 2 == 0:
            out = cupy.empty(size, dtype=dtype)
            func(self._generator, out.data.ptr, out.size, *args)
            return out
        else:
            out = cupy.empty((element_size + 1,), dtype=dtype)
            func(self._generator, out.data.ptr, out.size, *args)
            return out[:element_size].reshape(size)

    # NumPy compatible functions

    def lognormal(self, mean=0.0, sigma=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a log normal distribution.

        .. seealso::
            :func:`cupy.random.lognormal` for full documentation,
            :meth:`numpy.random.RandomState.lognormal`

        """
        dtype = _check_and_get_dtype(dtype)
        if dtype.char == 'f':
            func = curand.generateLogNormal
        else:
            func = curand.generateLogNormalDouble
        return self._generate_normal(func, size, dtype, mean, sigma)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of normally distributed samples.

        .. seealso::
            :func:`cupy.random.normal` for full documentation,
            :meth:`numpy.random.RandomState.normal`

        """
        dtype = _check_and_get_dtype(dtype)
        if dtype.char == 'f':
            func = curand.generateNormal
        else:
            func = curand.generateNormalDouble
        return self._generate_normal(func, size, dtype, loc, scale)

    def rand(self, *size, **kwarg):
        """Returns uniform random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`cupy.random.rand` for full documentation,
            :meth:`numpy.random.RandomState.rand`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('rand() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.random_sample(size=size, dtype=dtype)

    def randn(self, *size, **kwarg):
        """Returns an array of standard normal random values.

        .. seealso::
            :func:`cupy.random.randn` for full documentation,
            :meth:`numpy.random.RandomState.randn`

        """
        dtype = kwarg.pop('dtype', float)
        if kwarg:
            raise TypeError('randn() got unexpected keyword arguments %s'
                            % ', '.join(kwarg.keys()))
        return self.normal(size=size, dtype=dtype)

    _1m_kernel = core.ElementwiseKernel(
        '', 'T x', 'x = 1 - x', 'cupy_random_1_minus_x')

    def random_sample(self, size=None, dtype=float):
        """Returns an array of random values over the interval ``[0, 1)``.

        .. seealso::
            :func:`cupy.random.random_sample` for full documentation,
            :meth:`numpy.random.RandomState.random_sample`

        """
        dtype = _check_and_get_dtype(dtype)
        out = cupy.empty(size, dtype=dtype)
        if dtype.char == 'f':
            func = curand.generateUniform
        else:
            func = curand.generateUniformDouble
        func(self._generator, out.data.ptr, out.size)
        RandomState._1m_kernel(out)
        return out

    def interval(self, mx, size):
        """Generate multiple integers independently sampled uniformly from ``[0, mx]``.

        Args:
            mx (int): Upper bound of the interval
            size (None or int or tuple): Shape of the array or the scalar
                returned.
        Returns:
            int or cupy.ndarray: If ``None``, an :class:`cupy.ndarray` with
            shape ``()`` is returned.
            If ``int``, 1-D array of length size is returned.
            If ``tuple``, multi-dimensional array with shape
            ``size`` is returned.
            Currently, only 32 bit integers can be sampled.
            If 0 :math:`\\leq` ``mx`` :math:`\\leq` 0x7fffffff,
            a ``numpy.int32`` array is returned.
            If 0x80000000 :math:`\\leq` ``mx`` :math:`\\leq` 0xffffffff,
            a ``numpy.uint32`` array is returned.
        """
        if size is None:
            return self.interval(mx, 1).reshape(())
        elif isinstance(size, int):
            size = (size, )

        if mx == 0:
            return cupy.zeros(size, dtype=numpy.int32)

        if mx < 0:
            raise ValueError(
                'mx must be non-negative (actual: {})'.format(mx))
        elif mx <= 0x7fffffff:
            dtype = numpy.int32
        elif mx <= 0xffffffff:
            dtype = numpy.uint32
        else:
            raise ValueError(
                'mx must be within uint32 range (actual: {})'.format(mx))

        mask = (1 << mx.bit_length()) - 1
        mask = cupy.array(mask, dtype=dtype)

        n = functools.reduce(operator.mul, size, 1)

        sample = cupy.empty((n,), dtype=dtype)
        n_rem = n  # The number of remaining elements to sample
        ret = None
        while n_rem > 0:
            curand.generate(
                self._generator, sample.data.ptr, sample.size)
            # Drop the samples that exceed the upper limit
            sample &= mask
            success = sample <= mx

            if ret is None:
                # If the sampling has finished in the first iteration,
                # just return the sample.
                if success.all():
                    n_rem = 0
                    ret = sample
                    break

                # Allocate the return array.
                ret = cupy.empty((n,), dtype=dtype)

            n_succ = min(n_rem, int(success.sum()))
            ret[n - n_rem:n - n_rem + n_succ] = sample[success][:n_succ]
            n_rem -= n_succ

        assert n_rem == 0
        return ret.reshape(size)

    def seed(self, seed=None):
        """Resets the state of the random number generator with a seed.

        .. seealso::
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
            seed = numpy.asarray(seed).astype(numpy.uint64, casting='safe')

        curand.setPseudoRandomGeneratorSeed(self._generator, seed)
        curand.setGeneratorOffset(self._generator, 0)

    def standard_normal(self, size=None, dtype=float):
        """Returns samples drawn from the standard normal distribution.

        .. seealso::
            :func:`cupy.random.standard_normal` for full documentation,
            :meth:`numpy.random.RandomState.standard_normal`

        """
        return self.normal(size=size, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=float):
        """Returns an array of uniformly-distributed samples over an interval.

        .. seealso::
            :func:`cupy.random.uniform` for full documentation,
            :meth:`numpy.random.RandomState.uniform`

        """
        dtype = numpy.dtype(dtype)
        rand = self.random_sample(size=size, dtype=dtype)
        return dtype.type(low) + rand * dtype.type(high - low)

    def choice(self, a, size=None, replace=True, p=None):
        """Returns an array of random values from a given 1-D array.

        .. seealso::
            :func:`cupy.random.choice` for full document,
            :meth:`numpy.random.choice`

        """
        if a is None:
            raise ValueError('a must be 1-dimensional or an integer')
        if isinstance(a, cupy.ndarray) and a.ndim == 0:
            raise NotImplementedError
        if isinstance(a, six.integer_types):
            a_size = a
            if a_size <= 0:
                raise ValueError('a must be greater than 0')
        else:
            a = cupy.array(a, copy=False)
            if a.ndim != 1:
                raise ValueError('a must be 1-dimensional or an integer')
            else:
                a_size = len(a)
                if a_size == 0:
                    raise ValueError('a must be non-empty')

        if p is not None:
            p = cupy.array(p)
            if p.ndim != 1:
                raise ValueError('p must be 1-dimensional')
            if len(p) != a_size:
                raise ValueError('a and p must have same size')
            if not (p >= 0).all():
                raise ValueError('probabilities are not non-negative')
            p_sum = cupy.sum(p).get()
            if not numpy.allclose(p_sum, 1):
                raise ValueError('probabilities do not sum to 1')

        if size is None:
            raise NotImplementedError
        shape = size
        size = numpy.prod(shape)

        if not replace and p is None:
            if a_size < size:
                raise ValueError(
                    'Cannot take a larger sample than population when '
                    '\'replace=False\'')
            if isinstance(a, six.integer_types):
                indices = cupy.arange(a, dtype='l')
            else:
                indices = a.copy()
            self.shuffle(indices)
            return indices[:size].reshape(shape)

        if not replace:
            raise NotImplementedError

        if p is not None:
            p = cupy.broadcast_to(p, (size, a_size))
            index = cupy.argmax(cupy.log(p) +
                                self.gumbel(size=(size, a_size)),
                                axis=1)
            if not isinstance(shape, six.integer_types):
                index = cupy.reshape(index, shape)
        else:
            index = self.randint(0, a_size, size=shape)
            # Align the dtype with NumPy
            index = index.astype(cupy.int64, copy=False)

        if isinstance(a, six.integer_types):
            return index

        if index.ndim == 0:
            return cupy.array(a[index], dtype=a.dtype)

        return a[index]

    def shuffle(self, a):
        """Returns a shuffled array.

        .. seealso::
            :func:`cupy.random.shuffle` for full document,
            :meth:`numpy.random.shuffle`

        """
        if not isinstance(a, cupy.ndarray):
            raise TypeError('The array must be cupy.ndarray')

        if a.ndim == 0:
            raise TypeError('An array whose ndim is 0 is not supported')

        sample = cupy.zeros((len(a)), dtype=numpy.int32)
        curand.generate(self._generator, sample.data.ptr, sample.size)
        a[:] = a[cupy.argsort(sample)]

    def gumbel(self, loc=0.0, scale=1.0, size=None, dtype=float):
        """Returns an array of samples drawn from a Gumbel distribution.

        .. seealso::
            :func:`cupy.random.gumbel` for full documentation,
            :meth:`numpy.random.RandomState.gumbel`
        """
        x = self.uniform(size=size, dtype=dtype)
        # We use `1 - x` as input of `log` method to prevent overflow.
        # It obeys numpy implementation.
        _get_gumbel_kernel()(x, loc, scale, x)
        return x

    def randint(self, low, high=None, size=None, dtype='l'):
        """Returns a scalar or an array of integer values over ``[low, high)``.

        .. seealso::
            :func:`cupy.random.randint` for full documentation,
            :meth:`numpy.random.RandomState.randint`
        """
        if high is None:
            lo = 0
            hi = low
        else:
            lo = low
            hi = high

        if lo >= hi:
            raise ValueError('low >= high')
        if lo < cupy.iinfo(dtype).min:
            raise ValueError(
                'low is out of bounds for {}'.format(cupy.dtype(dtype).name))
        if hi > cupy.iinfo(dtype).max + 1:
            raise ValueError(
                'high is out of bounds for {}'.format(cupy.dtype(dtype).name))

        diff = hi - lo - 1
        if diff > cupy.iinfo(cupy.int32).max - cupy.iinfo(cupy.int32).min + 1:
            raise NotImplementedError(
                'Sampling from a range whose extent is larger than int32 '
                'range is currently not supported')
        x = self.interval(diff, size).astype(dtype, copy=False)
        cupy.add(x, lo, out=x)
        return x


def seed(seed=None):
    """Resets the state of the random number generator with a seed.

    This function resets the state of the global random number generator for
    the current device. Be careful that generators for other devices are not
    affected.

    Args:
        seed (None or int): Seed for the random number generator. If ``None``,
            it uses :func:`os.urandom` if available or :func:`time.clock`
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
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        seed = os.getenv('CUPY_SEED')
        if seed is None:
            seed = os.getenv('CHAINER_SEED')
        if seed is not None:
            seed = numpy.uint64(int(seed))
        rs = RandomState(seed)
        rs = _random_states.setdefault(dev.id, rs)
    return rs


def _check_and_get_dtype(dtype):
    dtype = numpy.dtype(dtype)
    if dtype.char not in ('f', 'd'):
        raise TypeError('cupy.random only supports float32 and float64')
    return dtype
