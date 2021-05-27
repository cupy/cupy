import atexit
import functools
import numpy
import os
import random
import types
import unittest

import cupy


_old_python_random_state = None
_old_numpy_random_state = None
_old_cupy_random_states = None


def do_setup(deterministic=True):
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_cupy_random_states
    _old_python_random_state = random.getstate()
    _old_numpy_random_state = numpy.random.get_state()
    _old_cupy_random_states = cupy.random._generator._random_states
    cupy.random.reset_states()
    # Check that _random_state has been recreated in
    # cupy.random.reset_states(). Otherwise the contents of
    # _old_cupy_random_states would be overwritten.
    assert (cupy.random._generator._random_states is not
            _old_cupy_random_states)

    if not deterministic:
        random.seed()
        numpy.random.seed()
        cupy.random.seed()
    else:
        random.seed(99)
        numpy.random.seed(100)
        cupy.random.seed(101)


def do_teardown():
    global _old_python_random_state
    global _old_numpy_random_state
    global _old_cupy_random_states
    random.setstate(_old_python_random_state)
    numpy.random.set_state(_old_numpy_random_state)
    cupy.random._generator._random_states = _old_cupy_random_states
    _old_python_random_state = None
    _old_numpy_random_state = None
    _old_cupy_random_states = None


# In some tests (which utilize condition.repeat or condition.retry),
# setUp/tearDown is nested. _setup_random() and _teardown_random() do their
# work only in the outermost setUp/tearDown pair.
_nest_count = 0


@atexit.register
def _check_teardown():
    assert _nest_count == 0, ('_setup_random() and _teardown_random() '
                              'must be called in pairs.')


def _setup_random():
    """Sets up the deterministic random states of ``numpy`` and ``cupy``.

    """
    global _nest_count
    if _nest_count == 0:
        nondeterministic = bool(int(os.environ.get(
            'CUPY_TEST_RANDOM_NONDETERMINISTIC', '0')))
        do_setup(not nondeterministic)
    _nest_count += 1


def _teardown_random():
    """Tears down the deterministic random states set up by ``_setup_random``.

    """
    global _nest_count
    assert _nest_count > 0, '_setup_random has not been called'
    _nest_count -= 1
    if _nest_count == 0:
        do_teardown()


def generate_seed():
    assert _nest_count > 0, 'random is not set up'
    return numpy.random.randint(0x7fffffff)


def fix_random():
    """Decorator that fixes random numbers in a test.

    This decorator can be applied to either a test case class or a test method.
    It should not be applied within ``condition.retry`` or
    ``condition.repeat``.
    """

    # TODO(niboshi): Prevent this decorator from being applied within
    #    condition.repeat or condition.retry decorators. That would repeat
    #    tests with the same random seeds. It's okay to apply this outside
    #    these decorators.

    def decorator(impl):
        if (isinstance(impl, types.FunctionType) and
                impl.__name__.startswith('test_')):
            # Applied to test method
            @functools.wraps(impl)
            def test_func(self, *args, **kw):
                _setup_random()
                try:
                    impl(self, *args, **kw)
                finally:
                    _teardown_random()
            return test_func
        elif isinstance(impl, type) and issubclass(impl, unittest.TestCase):
            # Applied to test case class
            klass = impl

            def wrap_setUp(f):
                def func(self):
                    _setup_random()
                    f(self)
                return func

            def wrap_tearDown(f):
                def func(self):
                    try:
                        f(self)
                    finally:
                        _teardown_random()
                return func

            klass.setUp = wrap_setUp(klass.setUp)
            klass.tearDown = wrap_tearDown(klass.tearDown)
            return klass
        else:
            raise ValueError('Can\'t apply fix_random to {}'.format(impl))

    return decorator
