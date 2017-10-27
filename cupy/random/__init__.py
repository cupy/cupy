import numpy


def bytes(length):
    """Returns random bytes.

    .. seealso:: :func:`numpy.random.bytes`
    """
    return numpy.bytes(length)


from cupy.random import distributions  # NOQA
from cupy.random import generator  # NOQA
from cupy.random import permutations  # NOQA
from cupy.random import sample as sample_  # NOQA


# import class and function
from cupy.random.distributions import gumbel  # NOQA
from cupy.random.distributions import lognormal  # NOQA
from cupy.random.distributions import normal  # NOQA
from cupy.random.distributions import standard_normal  # NOQA
from cupy.random.distributions import uniform  # NOQA
from cupy.random.generator import get_random_state  # NOQA
from cupy.random.generator import RandomState  # NOQA
from cupy.random.generator import reset_states  # NOQA
from cupy.random.generator import seed  # NOQA
from cupy.random.permutations import shuffle  # NOQA
from cupy.random.sample import choice  # NOQA
from cupy.random.sample import multinomial  # NOQA
from cupy.random.sample import rand  # NOQA
from cupy.random.sample import randint  # NOQA
from cupy.random.sample import randn  # NOQA
from cupy.random.sample import random_integers  # NOQA
from cupy.random.sample import random_sample  # NOQA
from cupy.random.sample import random_sample as random  # NOQA
from cupy.random.sample import random_sample as ranf  # NOQA
from cupy.random.sample import random_sample as sample  # NOQA
