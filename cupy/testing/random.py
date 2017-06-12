import atexit
import numpy

import cupy


# In some tests (which utilize condition.repeat or condition.retry),
# setUp/tearDown is nested. setup_random() and teardown_random() do their
# work only in the outermost setUp/tearDown pair.
_nest_count = 0


@atexit.register
def _check_teardown():
    assert _nest_count == 0, ('setup_random() and teardown_random() '
                              'must be called in pairs.')


def setup_random():
    """Sets up the deterministic random states of ``numpy`` and ``cupy``.

    """
    global _nest_count
    if _nest_count == 0:
        numpy.random.seed(100)
        cupy.random.seed(101)
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
