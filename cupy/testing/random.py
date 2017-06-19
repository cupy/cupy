import os
import atexit
import numpy

import cupy


# In some tests (which utilize condition.repeat or condition.retry),
# setUp/tearDown is nested. setup_random() and teardown_random() do their
# work only in the outermost setUp/tearDown pair.
_nest_count = 0

_old_numpy_random_state = None
_old_cupy_random_states = None


@atexit.register
def _check_teardown():
    assert _nest_count == 0, ('setup_random() and teardown_random() '
                              'must be called in pairs.')


def setup_random():
    """Sets up the deterministic random states of ``numpy`` and ``cupy``.

    """
    global _nest_count
    global _old_numpy_random_state
    global _old_cupy_random_states
    if _nest_count == 0:
        _old_numpy_random_state = numpy.random.get_state()
        _old_cupy_random_states = cupy.random.generator._random_states
        cupy.random.reset_states()
        # Check that _random_state has been recreated in
        # cupy.random.reset_states(). Otherwise the contents of
        # _old_cupy_random_states would be overwritten.
        assert (cupy.random.generator._random_states is not
                _old_cupy_random_states)

        nondeterministic = bool(int(os.environ.get(
            'CUPY_TEST_RANDOM_NONDETERMINISTIC', '0')))
        if nondeterministic:
            numpy.random.seed()
            cupy.random.seed()
        else:
            numpy.random.seed(100)
            cupy.random.seed(101)
    _nest_count += 1


def teardown_random():
    """Tears down the deterministic random states set up by ``setup_random``.

    """
    global _nest_count
    global _old_numpy_random_state
    global _old_cupy_random_states
    assert _nest_count > 0
    _nest_count -= 1
    if _nest_count == 0:
        numpy.random.set_state(_old_numpy_random_state)
        cupy.random.generator._random_states = _old_cupy_random_states
        _old_numpy_random_state = None
        _old_cupy_random_states = None
