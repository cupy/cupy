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
from cupy.random._bit_generator import (
    XORWOW,
    BitGenerator,
    MRG32k3a,
    Philox4x3210,
)
from cupy.random._distributions import (
    beta,
    binomial,
    chisquare,
    dirichlet,
    exponential,
    f,
    gamma,
    geometric,
    gumbel,
    hypergeometric,
    laplace,
    logistic,
    lognormal,
    logseries,
    multivariate_normal,
    negative_binomial,
    noncentral_chisquare,
    noncentral_f,
    normal,
    pareto,
    poisson,
    power,
    rayleigh,
    standard_cauchy,
    standard_exponential,
    standard_gamma,
    standard_normal,
    standard_t,
    triangular,
    uniform,
    vonmises,
    wald,
    weibull,
    zipf,
)
from cupy.random._generator import (
    RandomState,
    get_random_state,
    reset_states,
    seed,
    set_random_state,
)
from cupy.random._permutations import permutation, shuffle
from cupy.random._sample import (
    choice,
    multinomial,
    rand,
    randint,
    randn,
    random_integers,
    random_sample,
)
from cupy.random._sample import random_sample as random
from cupy.random._sample import random_sample as ranf
from cupy.random._sample import random_sample as sample
