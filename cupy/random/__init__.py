import numpy as _numpy

import cupy as _cupy
from cupy_backends.cuda.api import runtime as _runtime


def bytes(length):
    """Returns random bytes.

    .. note:: This function is just a wrapper for :obj:`numpy.random.bytes`.
        The resulting bytes are generated on the host (NumPy), not GPU.

    .. seealso:: :meth:`numpy.random.bytes
                 <numpy.random.mtrand.RandomState.bytes>`
    """
    # TODO(kmaehashi): should it be provided in CuPy?
    return _numpy.random.bytes(length)


def default_rng(seed=None):  # NOQA: F811 (avoid redefinition of seed)
    """Construct a new Generator with the default BitGenerator (XORWOW).

    Args:
        seed (None, int, array_like[ints], numpy.random.SeedSequence, cupy.random.BitGenerator, cupy.random.Generator, optional):
            A seed to initialize the :class:`cupy.random.BitGenerator`. If an
            ``int`` or ``array_like[ints]`` or None is passed, then it will be
            passed to :class:`numpy.random.SeedSequence` to detive the initial
            :class:`BitGenerator` state. One may also pass in a `SeedSequence
            instance. Adiditionally, when passed :class:`BitGenerator`, it will
            be wrapped by :class:`Generator`. If passed a :class:`Generator`,
            it will be returned unaltered.

    Returns:
        Generator: The initialized generator object.
    """
    from cupy.random._generator_api import Generator

    if _runtime.is_hip and int(str(_runtime.runtimeGetVersion())[:3]) < 403:
        raise RuntimeError(
            "Generator API not supported in ROCm<4.3,"
            " please use the legacy one or update ROCm."
        )
    if isinstance(seed, BitGenerator):
        return Generator(seed)
    elif isinstance(seed, Generator):
        return seed
    return Generator(XORWOW(seed))


def __getattr__(key):
    # TODO(kmaehashi): Split cuRAND dependency from Generator class to allow
    # users use the class for type annotation.
    if key == "Generator":
        # Lazy import libraries depending on cuRAND
        import cupy.random._generator_api

        Generator = cupy.random._generator_api.Generator
        _cupy.random.Generator = Generator
        return Generator
    raise AttributeError(f"module '{__name__}' has no attribute '{key}'")


# import class and function
from cupy.random._distributions import beta
from cupy.random._distributions import binomial
from cupy.random._distributions import chisquare
from cupy.random._distributions import dirichlet
from cupy.random._distributions import exponential
from cupy.random._distributions import f
from cupy.random._distributions import gamma
from cupy.random._distributions import geometric
from cupy.random._distributions import gumbel
from cupy.random._distributions import hypergeometric
from cupy.random._distributions import laplace
from cupy.random._distributions import logistic
from cupy.random._distributions import lognormal
from cupy.random._distributions import logseries
from cupy.random._distributions import multivariate_normal
from cupy.random._distributions import negative_binomial
from cupy.random._distributions import noncentral_chisquare
from cupy.random._distributions import noncentral_f
from cupy.random._distributions import normal
from cupy.random._distributions import pareto
from cupy.random._distributions import poisson
from cupy.random._distributions import power
from cupy.random._distributions import rayleigh
from cupy.random._distributions import standard_cauchy
from cupy.random._distributions import standard_exponential
from cupy.random._distributions import standard_gamma
from cupy.random._distributions import standard_normal
from cupy.random._distributions import standard_t
from cupy.random._distributions import triangular
from cupy.random._distributions import uniform
from cupy.random._distributions import vonmises
from cupy.random._distributions import wald
from cupy.random._distributions import weibull
from cupy.random._distributions import zipf
from cupy.random._generator import get_random_state
from cupy.random._generator import RandomState
from cupy.random._generator import reset_states
from cupy.random._generator import seed
from cupy.random._generator import set_random_state
from cupy.random._permutations import permutation
from cupy.random._permutations import shuffle
from cupy.random._sample import choice
from cupy.random._sample import multinomial
from cupy.random._sample import rand
from cupy.random._sample import randint
from cupy.random._sample import randn
from cupy.random._sample import random_integers
from cupy.random._sample import random_sample
from cupy.random._sample import random_sample as random
from cupy.random._sample import random_sample as ranf
from cupy.random._sample import random_sample as sample
from cupy.random._bit_generator import BitGenerator
from cupy.random._bit_generator import XORWOW
from cupy.random._bit_generator import MRG32k3a
from cupy.random._bit_generator import Philox4x3210
