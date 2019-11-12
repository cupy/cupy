import functools
import itertools
import types
import typing as tp  # NOQA
import unittest

import numpy
import six

from chainer.testing import _bundle
from chainer import utils


def _param_to_str(obj):
    if isinstance(obj, type):
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

    def __str__(self):
        name = base.__str__(self)
        return '%s  parameter: %s' % (name, param)

    mb = {'__str__': __str__}
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

    def method_generator(base_method):
        # Generates a wrapped test method

        @functools.wraps(base_method)
        def new_method(self, *args, **kwargs):
            try:
                return base_method(self, *args, **kwargs)
            except unittest.SkipTest:
                raise
            except Exception as e:
                s = six.StringIO()
                s.write('Parameterized test failed.\n\n')
                s.write('Base test method: {}.{}\n'.format(
                    base.__name__, base_method.__name__))
                s.write('Test parameters:\n')
                for k, v in sorted(param.items()):
                    s.write('  {}: {}\n'.format(k, v))
                utils._raise_from(e.__class__, s.getvalue(), e)
        return new_method

    return (cls_name, mb, method_generator)


def parameterize(*params):
    # TODO(niboshi): Add documentation
    return _bundle.make_decorator(
        lambda base: _parameterize_test_case_generator(base, params))


def _values_to_dicts(names, values):
    assert isinstance(names, six.string_types)
    assert isinstance(values, (tuple, list))

    def safe_zip(ns, vs):
        if len(ns) == 1:
            return [(ns[0], vs)]
        assert isinstance(vs, (tuple, list)) and len(ns) == len(vs)
        return zip(ns, vs)

    names = names.split(',')
    params = [dict(safe_zip(names, value_list)) for value_list in values]
    return params


def from_pytest_parameterize(names, values):
    # Pytest-style parameterization.
    # TODO(niboshi): Add documentation
    return _values_to_dicts(names, values)


def parameterize_pytest(names, values):
    # Pytest-style parameterization.
    # TODO(niboshi): Add documentation
    return parameterize(*from_pytest_parameterize(names, values))


def product(parameter):
    # TODO(niboshi): Add documentation
    if isinstance(parameter, dict):
        return product_dict(*[
            _values_to_dicts(names, values)
            for names, values in sorted(parameter.items())])

    elif isinstance(parameter, list):
        # list of lists of dicts
        if not all(isinstance(_, list) for _ in parameter):
            raise TypeError('parameter must be list of lists of dicts')
        if not all(isinstance(_, dict) for l in parameter for _ in l):
            raise TypeError('parameter must be list of lists of dicts')
        return product_dict(*parameter)

    else:
        raise TypeError(
            'parameter must be either dict or list. Actual: {}'.format(
                type(parameter)))


def product_dict(*parameters):
    # TODO(niboshi): Add documentation
    return [
        {k: v for dic in dicts for k, v in six.iteritems(dic)}
        for dicts in itertools.product(*parameters)]


# TODO(kataoka): product_dict is patched by tests/conftest.py while tests are
# collected if CHAINER_TEST_PAIRWISE_PARAMETERIZATION is configured
# accordingly. Also used in
# tests/chainer_tests/testing_tests/test_parameterized.py
_product_dict_orig = product_dict


def _pairwise_product_dict(*parameters):
    if len(parameters) <= 2:
        return _product_dict_orig(*parameters)
    return list(_pairwise_product_dict_iter(*parameters))


def _pairwise_product_dict_iter(
        *parameters: tp.Iterable[tp.Dict[str, tp.Any]]
) -> tp.Iterator[tp.Dict[str, tp.Any]]:
    """Generate combinations that cover all pairs.

    The argument is the same as `chainer.testing.product_dict`.

    """
    parameter_lists = [list(dicts) for dicts in parameters]  # type: tp.List[tp.List[tp.Dict[str, tp.Any]]]  # NOQA

    for nd_index in sorted(_nd_indices_to_cover_each_2d(
            [len(dicts) for dicts in parameter_lists])):
        yield {
            k: v
            for i, dicts in zip(nd_index, parameter_lists)
            for k, v in dicts[i].items()}


def _nd_indices_to_cover_each_2d(
        shape: tp.Sequence[int]
) -> tp.Iterator[tp.Tuple[int, ...]]:
    rs = numpy.random.RandomState(seed=0)
    n = len(shape)
    indices = [list(range(length)) for length in shape]  # type: tp.List[tp.List[int]]  # NOQA

    # `(k_i, k_j) in uncovered[(i, j)]` iff it has not been yielded
    # `nd_index` such that `(nd_index[i], nd_inde[j]) == (k_i, k_j)`.
    uncovered = {}  # type: tp.Dict[tp.Tuple[int, int], tp.Set[tp.Tuple[int, int]]]  # NOQA
    for i, j in itertools.combinations(range(n), 2):
        uncovered[(i, j)] = set(itertools.product(indices[i], indices[j]))

    nd_indices = list(itertools.product(*indices))  # type: tp.List[tp.Tuple[int, ...]]  # NOQA
    rs.shuffle(nd_indices)
    for nd_index in nd_indices:
        count = 0
        for i, j in itertools.combinations(range(n), 2):
            try:
                uncovered[(i, j)].remove((nd_index[i], nd_index[j]))
            except KeyError:
                pass
            else:
                count += 1
        if count > 0:
            yield nd_index
