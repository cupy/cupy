import itertools
import sys
import unittest


def _gen_case(base, module, i, param):
    cls_name = '%s_param_%d' % (base.__name__, i)

    def __str__(self):
        name = base.__str__(self)
        return '%s  parameter: %s' % (name, param)

    mb = dict(param)

    mb['__str__'] = __str__
    cls = type(cls_name, (base,), mb)
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
