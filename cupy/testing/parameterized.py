import functools
import inspect
import itertools
import sys
import types
import unittest

import six


def _gen_case(base, module, i, param):
    cls_name = '%s_param_%d' % (base.__name__, i)

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
