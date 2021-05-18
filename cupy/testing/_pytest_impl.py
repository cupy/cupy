import unittest

import cupy.testing._parameterized

try:
    import pytest
    _error = None
except ImportError as e:
    _error = e


def is_available():
    return _error is None


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


if is_available():

    class _TestingParameterizeMixin:

        def __repr__(self):
            return '<{}  parameter: {}>'.format(
                super().__repr__(),
                self.__dict__,
            )

        @pytest.fixture(autouse=True)
        def _cupy_testing_parameterize(self, _cupy_testing_param):
            assert not self.__dict__, \
                'There should not be another hack with instance attribute.'
            self.__dict__.update(_cupy_testing_param)


def parameterize(*params):
    check_available()
    param_name = cupy.testing._parameterized._make_class_name
    # TODO(kataoka): Give better names (`id`).
    # For now, use legacy `_make_class_name` just for consistency. Here,
    # a generated name is `TestFoo::test_bar[_param_0_{...}]`, whereas
    # a legacy name is `TestFoo_param_0_{...}::test_bar
    params = [
        pytest.param(param, id=param_name('', i, param))
        for i, param in enumerate(params)
    ]

    def f(cls):
        assert not issubclass(cls, unittest.TestCase)
        if issubclass(cls, _TestingParameterizeMixin):
            raise RuntimeError('do not `@testing.parameterize` twice')
        module_name = cls.__module__
        cls = type(cls.__name__, (_TestingParameterizeMixin, cls), {})
        cls.__module__ = module_name
        cls = pytest.mark.parametrize('_cupy_testing_param', params)(cls)
        return cls
    return f
