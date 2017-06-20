import os
import atexit
import functools
import numpy
import types
import unittest

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
    assert _nest_count > 0, 'setup_random has not been called'
    _nest_count -= 1
    if _nest_count == 0:
        numpy.random.set_state(_old_numpy_random_state)
        cupy.random.generator._random_states = _old_cupy_random_states
        _old_numpy_random_state = None
        _old_cupy_random_states = None


def generate_seed():
    assert _nest_count > 0, 'random is not set up'
    return numpy.random.randint(0xffffffff)


def fix_random():
    # TODO(niboshi): Prevent this decorator from being applied within
    #    condition.repeat or condition.retry decorators. That would repeat
    #    tests with the same random seeds. It's okay to apply this outside
    #    these decorators.

    def decorator(impl):
        if type(impl) is types.FunctionType:
            # Applied to test method
            @functools.wraps(impl)
            def test_func(self, *args, **kw):
                setup_random()
                impl(self, *args, **kw)
                teardown_random()
            return test_func
        elif type(impl) is type and issubclass(impl, unittest.TestCase):
            # Applied to test case class
            klass = impl

            def wrap_setUp(f):
                def func(self):
                    setup_random()
                    f(self)
                return func

            def wrap_tearDown(f):
                def func(self):
                    f(self)
                    teardown_random()
                return func

            klass.setUp = wrap_setUp(klass.setUp)
            klass.tearDown = wrap_tearDown(klass.tearDown)
            return klass
        else:
            raise ValueError('Invalid object {}'.format(type(impl)))

    return decorator
