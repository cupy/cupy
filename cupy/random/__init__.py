import numpy as _numpy

from cupy_backends.cuda.api import runtime


def bytes(length):
    """Returns random bytes.

    .. note:: This function is just a wrapper for :obj:`numpy.random.bytes`.
        The resulting bytes are generated on the host (NumPy), not GPU.

    .. seealso:: :meth:`numpy.random.bytes
                 <numpy.random.mtrand.RandomState.bytes>`
    """
    # TODO(kmaehashi): should it be provided in CuPy?
    return _numpy.random.bytes(length)


def default_rng(seed=None):  # NOQA  avoid redefinition of seed
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
    """  # NOQA, list of types need to be in one line for sphinx
    if runtime.is_hip:
        raise RuntimeError('Generator API not supported in HIP,'
                           ' please use the legacy one.')
    if isinstance(seed, BitGenerator):
        return Generator(seed)
    elif isinstance(seed, Generator):
        return seed
    return Generator(XORWOW(seed))


# import class and function
from cupy.random._distributions import beta  # NOQA
from cupy.random._distributions import binomial  # NOQA
from cupy.random._distributions import chisquare  # NOQA
from cupy.random._distributions import dirichlet  # NOQA
from cupy.random._distributions import exponential  # NOQA
from cupy.random._distributions import f  # NOQA
from cupy.random._distributions import gamma  # NOQA
from cupy.random._distributions import geometric  # NOQA
from cupy.random._distributions import gumbel  # NOQA
from cupy.random._distributions import hypergeometric  # NOQA
from cupy.random._distributions import laplace  # NOQA
from cupy.random._distributions import logistic  # NOQA
from cupy.random._distributions import lognormal  # NOQA
from cupy.random._distributions import logseries  # NOQA
from cupy.random._distributions import multivariate_normal  # NOQA
from cupy.random._distributions import negative_binomial  # NOQA
from cupy.random._distributions import noncentral_chisquare  # NOQA
from cupy.random._distributions import noncentral_f  # NOQA
from cupy.random._distributions import normal  # NOQA
from cupy.random._distributions import pareto  # NOQA
from cupy.random._distributions import poisson  # NOQA
from cupy.random._distributions import power  # NOQA
from cupy.random._distributions import rayleigh  # NOQA
from cupy.random._distributions import standard_cauchy  # NOQA
from cupy.random._distributions import standard_exponential  # NOQA
from cupy.random._distributions import standard_gamma  # NOQA
from cupy.random._distributions import standard_normal  # NOQA
from cupy.random._distributions import standard_t  # NOQA
from cupy.random._distributions import triangular  # NOQA
from cupy.random._distributions import uniform  # NOQA
from cupy.random._distributions import vonmises  # NOQA
from cupy.random._distributions import wald  # NOQA
from cupy.random._distributions import weibull  # NOQA
from cupy.random._distributions import zipf  # NOQA
from cupy.random._generator import get_random_state  # NOQA
from cupy.random._generator import RandomState  # NOQA
from cupy.random._generator import reset_states  # NOQA
from cupy.random._generator import seed  # NOQA
from cupy.random._generator import set_random_state  # NOQA
from cupy.random._permutations import permutation  # NOQA
from cupy.random._permutations import shuffle  # NOQA
from cupy.random._sample import choice  # NOQA
from cupy.random._sample import multinomial  # NOQA
from cupy.random._sample import rand  # NOQA
from cupy.random._sample import randint  # NOQA
from cupy.random._sample import randn  # NOQA
from cupy.random._sample import random_integers  # NOQA
from cupy.random._sample import random_sample  # NOQA
from cupy.random._sample import random_sample as random  # NOQA
from cupy.random._sample import random_sample as ranf  # NOQA
from cupy.random._sample import random_sample as sample  # NOQA
if not runtime.is_hip:
    # This is disabled for HIP due to a problem when using
    # dynamic dispatching of kernels
    # see https://github.com/ROCm-Developer-Tools/HIP/issues/2186
    from cupy.random._bit_generator import BitGenerator  # NOQA
    from cupy.random._bit_generator import XORWOW  # NOQA
    from cupy.random._bit_generator import MRG32k3a  # NOQA
    from cupy.random._bit_generator import Philox4x3210  # NOQA
    from cupy.random._generator_api import Generator  # NOQA
