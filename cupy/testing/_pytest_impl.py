import unittest

import cupy.testing.parameterized

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


class _TestingParameterizeMixin:

    @pytest.fixture(autouse=True)
    def _cupy_testing_parameterize(self, _cupy_testing_param):
        assert not self.__dict__
        self.__dict__ = _cupy_testing_param


def parameterize(*params):
    check_available()
    param_name = cupy.testing.parameterized._make_class_name
    # TODO(kataoka): Give better names (`id`).
    # For now, use legacy `_make_class_name` just for consistency. Here,
    # a generated name is `TestFoo::test_bar[_param_0_{...}]`, whereas
    # a legacy name is `TestFoo_param_0_{...}::test_bar
    params = [
        pytest.param(param, id=param_name("", i, param))
        for i, param in enumerate(params)
    ]

    def f(cls):
        assert not issubclass(cls, unittest.TestCase)
        if issubclass(cls, _TestingParameterizeMixin):
            raise RuntimeError("do not `@testing.parameterize` twice")
        cls = type(
            cls.__name__, (_TestingParameterizeMixin, cls),
            # `_TestingParameterizeMixin` will replace this empty dict
            {},
        )
        cls = pytest.mark.parametrize("_cupy_testing_param", params)(cls)
        return cls
    return f
