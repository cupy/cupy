import enum
import functools
import inspect
import itertools
import re
import sys
import types
import unittest

import six

STRING_TYPES = bytes, str

# The type of re.compile objects is not exposed in Python.
REGEX_TYPE = type(re.compile(""))

_non_printable_ascii_translate_table = {
    i: "\\x{:02x}".format(i) for i in range(128) if i not in range(32, 127)
}
_non_printable_ascii_translate_table.update(
    {ord("\t"): "\\t", ord("\r"): "\\r", ord("\n"): "\\n"}
)


def _translate_non_printable(s):
    return s.translate(_non_printable_ascii_translate_table)


def _ascii_escaped(val):
    """If val is pure ascii, returns it as a str().  Otherwise, escapes
    bytes objects into a sequence of escaped bytes:

    b'\xc3\xb4\xc5\xd6' -> '\\xc3\\xb4\\xc5\\xd6'

    and escapes unicode objects into a sequence of escaped unicode
    ids, e.g.:

    '4\\nV\\U00043efa\\x0eMXWB\\x1e\\u3028\\u15fd\\xcd\\U0007d944'

    note:
       the obvious "v.decode('unicode-escape')" will return
       valid utf-8 unicode if it finds them in bytes, but we
       want to return escaped bytes for any byte, even if they match
       a utf-8 string.

    """
    if isinstance(val, bytes):
        ret = _bytes_to_ascii(val)
    else:
        ret = val.encode("unicode_escape").decode("ascii")
    return _translate_non_printable(ret)


def _bytes_to_ascii(val):
    return val.decode("ascii", "backslashreplace")


def _idval(val, argname, i):
    if isinstance(val, STRING_TYPES):
        return _ascii_escaped(val)
    elif val is None or isinstance(val, (float, int, bool, enum.Enum)):
        return str(val)
    elif isinstance(val, REGEX_TYPE):
        return _ascii_escaped(val.pattern)
    elif ((inspect.isclass(val) or inspect.isfunction(val)) and
            hasattr(val, "__name__")):
        return val.__name__
    return str(argname) + str(i)


def _idmaker(param, i):
    """Generate a descriptive id for the parameter set.

    This is simplified from private code internal to pytest.
    """
    ids = [_idval(v, k, i) for k, v in param.items()]
    return '[' + '-'.join(ids) + ']'


def _gen_case(base, module, i, param):
    cls_name = '%s%s' % (base.__name__, _idmaker(param, i))

    # Add parameters as members

    def __str__(self):
        name = base.__str__(self)
        return '%s  parameter: %s' % (name, param)

    mb = {'__str__': __str__}
    for k, v in six.iteritems(param):
        if isinstance(v, types.FunctionType):

            def create_new_v():
                f = v

                def new_v(self, *args, **kwargs):
                    return f(*args, **kwargs)
                return new_v

            mb[k] = create_new_v()
        else:
            mb[k] = v

    cls = type(cls_name, (base,), mb)

    # Wrap test methods to generate useful error message

    def wrap_test_method(method):
        @functools.wraps(method)
        def wrap(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except unittest.SkipTest:
                raise
            except Exception as e:
                s = six.StringIO()
                s.write('Parameterized test failed.\n\n')
                s.write('Base test method: {}.{}\n'.format(
                    base.__name__, method.__name__))
                s.write('Test parameters:\n')
                for k, v in six.iteritems(param):
                    s.write('  {}: {}\n'.format(k, v))
                s.write('\n')
                s.write('{}: {}\n'.format(type(e).__name__, e))
                e_new = AssertionError(s.getvalue())
                if sys.version_info < (3,):
                    six.reraise(AssertionError, e_new, sys.exc_info()[2])
                else:
                    six.raise_from(e_new.with_traceback(e.__traceback__), None)
        return wrap

    # ismethod for Python 2 and isfunction for Python 3
    members = inspect.getmembers(
        cls, predicate=lambda _: inspect.ismethod(_) or inspect.isfunction(_))
    for name, method in members:
        if name.startswith('test_'):
            setattr(cls, name, wrap_test_method(method))

    # Add new test class to module
    setattr(module, cls_name, cls)


def _gen_cases(name, base, params):
    module = sys.modules[name]
    for i, param in enumerate(params):
        _gen_case(base, module, i, param)


def parameterize(*params):
    def f(klass):
        assert issubclass(klass, unittest.TestCase)
        _gen_cases(klass.__module__, klass, params)
        # Remove original base class
        return None
    return f


def product(parameter):
    keys = sorted(parameter)
    values = [parameter[key] for key in keys]
    values_product = itertools.product(*values)
    return [dict(zip(keys, vals)) for vals in values_product]


def product_dict(*parameters):
    return [
        {k: v for dic in dicts for k, v in six.iteritems(dic)}
        for dicts in itertools.product(*parameters)]
