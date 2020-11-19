import itertools
import types
import typing as tp  # NOQA
import unittest

from cupy.testing import _bundle
from cupy.testing import _pytest_impl


def _param_to_str(obj):
    if isinstance(obj, type):
        return obj.__name__
    elif hasattr(obj, '__name__') and isinstance(obj.__name__, str):
        # print __name__ attribute for classes, functions and modules
        return obj.__name__
    return repr(obj)


def _shorten(s, maxlen):
    # Shortens the string down to maxlen, by replacing the middle part with
    # a 3-dots string '...'.
    ellipsis = '...'
    if len(s) <= maxlen:
        return s
    n1 = (maxlen - len(ellipsis)) // 2
    n2 = maxlen - len(ellipsis) - n1
    s = s[:n1] + ellipsis + s[-n2:]
    assert len(s) == maxlen
    return s


def _make_class_name(base_class_name, i_param, param):
    # Creates a class name for a single combination of parameters.
    SINGLE_PARAM_MAXLEN = 100  # Length limit of a single parameter value
    PARAMS_MAXLEN = 5000  # Length limit of the whole parameters part
    param_strs = [
        '{}={}'.format(k, _shorten(_param_to_str(v), SINGLE_PARAM_MAXLEN))
        for k, v in sorted(param.items())]
    param_strs = _shorten(', '.join(param_strs), PARAMS_MAXLEN)
    cls_name = '{}_param_{}_{{{}}}'.format(
        base_class_name, i_param, param_strs)
    return cls_name


def _parameterize_test_case_generator(base, params):
    # Defines the logic to generate parameterized test case classes.

    for i, param in enumerate(params):
        yield _parameterize_test_case(base, i, param)


def _parameterize_test_case(base, i, param):
    cls_name = _make_class_name(base.__name__, i, param)

    def __repr__(self):
        name = base.__repr__(self)
        return '<%s  parameter: %s>' % (name, param)

    mb = {'__repr__': __repr__}
    for k, v in sorted(param.items()):
        if isinstance(v, types.FunctionType):

            def create_new_v():
                f = v

                def new_v(self, *args, **kwargs):
                    return f(*args, **kwargs)
                return new_v

            mb[k] = create_new_v()
        else:
            mb[k] = v

    return (cls_name, mb, lambda method: method)


def parameterize(*params):
    """Generates test classes with given sets of additional attributes

    >>> @parameterize({"a": 1}, {"b": 2, "c": 3})
    ... class TestX(unittest.TestCase):
    ...     def test_y(self):
    ...         pass

    generates two classes `TestX_param_0_...`, `TestX_param_1_...` and
    removes the original class `TestX`.

    The specification is subject to change, which applies to all the non-NumPy
    `testing` features.

    """
    def f(cls):
        if issubclass(cls, unittest.TestCase):
            deco = _bundle.make_decorator(
                lambda base: _parameterize_test_case_generator(base, params))
        else:
            deco = _pytest_impl.parameterize(*params)
        return deco(cls)
    return f


def product(parameter):
    # TODO(kataoka): Add documentation
    assert isinstance(parameter, dict)
    keys = sorted(parameter)
    values = [parameter[key] for key in keys]
    values_product = itertools.product(*values)
    return [dict(zip(keys, vals)) for vals in values_product]


def product_dict(*parameters):
    # TODO(kataoka): Add documentation
    return [
        {k: v for dic in dicts for k, v in dic.items()}
        for dicts in itertools.product(*parameters)]
