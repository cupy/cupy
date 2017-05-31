import numpy

import cupy


# In some tests (which utilize condition.repeat or condition.retry),
# setUp/tearDown is nested. setup_random() and teardown_random() do their
# work only in the outermost setUp/tearDown pair.
_nest_count = 0


def setup_random(numpy_seed=None, cupy_seed=None):
    """Sets up the deterministic random states of ``numpy`` and ``cupy``.

    Args:
         numpy_seed(int or None): Seed to initialize ``numpy.random``. If
         ``None``, predetermined constant value is used.
         cupy_seed(int or None): Seed to initialize ``cupy.random``. If
         ``None``, predetermined constant value is used.
    """
    global _nest_count
    if _nest_count == 0:
        numpy.random.seed(100 if numpy_seed is None else numpy_seed)
        cupy.random.seed(101 if cupy_seed is None else cupy_seed)
    _nest_count += 1


def teardown_random():
    """Tears down the deterministic random states set up by ``setup_random``.
    """
    global _nest_count
    assert _nest_count > 0
    _nest_count -= 1
    if _nest_count == 0:
        numpy.random.seed(None)
        cupy.random.reset_states()


def get_random_state(xp, seed=None):
    """Returns the random state whose seed is constant in each call.

    Args:
         xp(numpy or cupy): Array module to use.
         seed(int or None): Seed to initialize the random state. If ``None``,
         predetermined constant value is used.
    """
    if xp is numpy:
        return numpy.random.RandomState(102 if seed is None else seed)
    elif xp is cupy:
        return cupy.random.RandomState(103 if seed is None else seed)
    else:
        assert False
