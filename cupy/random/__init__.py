import numpy as _numpy


def bytes(length):
    """Returns random bytes.

    .. seealso:: :meth:`numpy.random.bytes
                 <numpy.random.mtrand.RandomState.bytes>`
    """
    return _numpy.bytes(length)


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
