import collections
import re
import textwrap
import unittest

import pytest

from cupy import testing


@testing.parameterize(
    {'actual': {'a': [1, 2], 'b': [3, 4, 5]},
     'expect': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 5},
                {'a': 2, 'b': 3}, {'a': 2, 'b': 4}, {'a': 2, 'b': 5}]},
    {'actual': {'a': [1, 2]}, 'expect': [{'a': 1}, {'a': 2}]},
    {'actual': {'a': [1, 2], 'b': []}, 'expect': []},
    {'actual': {'a': []}, 'expect': []},
    {'actual': {}, 'expect': [{}]})
class TestProduct(unittest.TestCase):

    def test_product(self):
        assert testing.product(self.actual) == self.expect


@testing.parameterize(
    {'actual': [[{'a': 1, 'b': 3}, {'a': 2, 'b': 4}], [{'c': 5}, {'c': 6}]],
     'expect': [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6},
                {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]},
    {'actual': [[{'a': 1}, {'a': 2}], [{'b': 3}, {'b': 4}, {'b': 5}]],
     'expect': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 5},
                {'a': 2, 'b': 3}, {'a': 2, 'b': 4}, {'a': 2, 'b': 5}]},
    {'actual': [[{'a': 1}, {'a': 2}]], 'expect': [{'a': 1}, {'a': 2}]},
    {'actual': [[{'a': 1}, {'a': 2}], []], 'expect': []},
    {'actual': [[]], 'expect': []},
    {'actual': [], 'expect': [{}]})
class TestProductDict(unittest.TestCase):

    def test_product_dict(self):
        assert testing.product_dict(*self.actual) == self.expect


def f(x):
    return x


class C:

    def __repr__(self):
        return '<C object>'

    def __call__(self, x):
        return x

    def method(self, x):
        return x


@testing.parameterize(
    {'callable': f},
    {'callable': lambda x: x},
    {'callable': C()},
    {'callable': C().method}
)
class TestParameterize(unittest.TestCase):

    def test_callable(self):
        y = self.callable(1)
        assert y == 1

    def test_skip(self):
        # Skipping the test case should not report error.
        self.skipTest('skip')


@testing.parameterize(
    {'callable': f},
    {'callable': lambda x: x},
    {'callable': C()},
    {'callable': C().method}
)
class TestParameterizePytestImpl:

    def test_callable(self):
        y = self.callable(1)
        assert y == 1

    def test_skip(self):
        # Skipping the test case should not report error.
        pytest.skip('skip')


@pytest.mark.parametrize(('src', 'outcomes'), [
    (  # simple
        textwrap.dedent('''\
        @testing.parameterize({'a': 1}, {'a': 2})
        class TestA:
            def test_a(self):
                assert self.a > 0
        '''), [
            ("::TestA::test_a[_param_0_{a=1}]", 'PASSED'),
            ("::TestA::test_a[_param_1_{a=2}]", 'PASSED'),
        ]
    ),
    (  # simple fail
        textwrap.dedent('''\
        @testing.parameterize({'a': 1}, {'a': 2})
        class TestA:
            def test_a(self):
                assert self.a == 1
        '''), [
            ("::TestA::test_a[_param_0_{a=1}]", 'PASSED'),
            ("::TestA::test_a[_param_1_{a=2}]", 'FAILED'),
        ]
    ),
    (  # set of params can be different
        textwrap.dedent('''\
        @testing.parameterize({'a': 1}, {'b': 2})
        class TestA:
            def test_a(self):
                a = getattr(self, 'a', 3)
                b = getattr(self, 'b', 4)
                assert (a, b) in [(1, 4), (3, 2)]
        '''), [
            ("::TestA::test_a[_param_0_{a=1}]", 'PASSED'),
            ("::TestA::test_a[_param_1_{b=2}]", 'PASSED'),
        ]
    ),
    (  # do not copy each value but do not share the param dict
        textwrap.dedent('''\
        import numpy
        @testing.parameterize({'a': numpy.array(1)}, {'a': 1})
        class TestA:
            def test_first(self):
                assert self.a == 1
                self.a += 2
            def test_second(self):
                assert self.a == 1
                self.a += 2
        '''), [
            ("::TestA::test_first[_param_0_{a=array(1)}]", 'PASSED'),
            ("::TestA::test_first[_param_1_{a=1}]", 'PASSED'),
            ("::TestA::test_second[_param_0_{a=array(1)}]", 'FAILED'),
            ("::TestA::test_second[_param_1_{a=1}]", 'PASSED'),
        ]
    ),
    (  # multiple params and class attr
        textwrap.dedent('''\
        @testing.parameterize({'a': 1, 'b': 4}, {'a': 2, 'b': 3})
        class TestA:
            c = 5
            def test_a(self):
                assert self.a + self.b == self.c
        '''), [
            ("::TestA::test_a[_param_0_{a=1, b=4}]", 'PASSED'),
            ("::TestA::test_a[_param_1_{a=2, b=3}]", 'PASSED'),
        ]
    ),
    (  # combine pytest.mark.parameterize
        textwrap.dedent('''\
        import pytest
        @pytest.mark.parametrize("outer", ["E", "e"])
        @testing.parameterize({"x": "D"}, {"x": "d"})
        @pytest.mark.parametrize("inner", ["c", "C"])
        class TestA:
            @pytest.mark.parametrize(
                ("fn1", "fn2"), [("A", "b"), ("a", "B")])
            def test_a(self, fn2, inner, outer, fn1):
                assert (
                    (fn1 + fn2 + inner + self.x + outer).lower()
                    == "abcde")
            @pytest.mark.parametrize(
                "fn", ["A", "a"])
            def test_b(self, outer, fn, inner):
                assert sum(
                    c.isupper() for c in [fn, inner, self.x, outer]
                ) != 2
        '''), [
            ("::TestA::test_a[A-b-_param_0_{x='D'}-E-c]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_0_{x='D'}-E-C]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_0_{x='D'}-e-c]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_0_{x='D'}-e-C]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_1_{x='d'}-E-c]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_1_{x='d'}-E-C]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_1_{x='d'}-e-c]", 'PASSED'),
            ("::TestA::test_a[A-b-_param_1_{x='d'}-e-C]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_0_{x='D'}-E-c]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_0_{x='D'}-E-C]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_0_{x='D'}-e-c]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_0_{x='D'}-e-C]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_1_{x='d'}-E-c]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_1_{x='d'}-E-C]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_1_{x='d'}-e-c]", 'PASSED'),
            ("::TestA::test_a[a-B-_param_1_{x='d'}-e-C]", 'PASSED'),
            ("::TestA::test_b[A-_param_0_{x='D'}-E-c]", 'PASSED'),
            ("::TestA::test_b[A-_param_0_{x='D'}-E-C]", 'PASSED'),
            ("::TestA::test_b[A-_param_0_{x='D'}-e-c]", 'FAILED'),
            ("::TestA::test_b[A-_param_0_{x='D'}-e-C]", 'PASSED'),
            ("::TestA::test_b[A-_param_1_{x='d'}-E-c]", 'FAILED'),
            ("::TestA::test_b[A-_param_1_{x='d'}-E-C]", 'PASSED'),
            ("::TestA::test_b[A-_param_1_{x='d'}-e-c]", 'PASSED'),
            ("::TestA::test_b[A-_param_1_{x='d'}-e-C]", 'FAILED'),
            ("::TestA::test_b[a-_param_0_{x='D'}-E-c]", 'FAILED'),
            ("::TestA::test_b[a-_param_0_{x='D'}-E-C]", 'PASSED'),
            ("::TestA::test_b[a-_param_0_{x='D'}-e-c]", 'PASSED'),
            ("::TestA::test_b[a-_param_0_{x='D'}-e-C]", 'FAILED'),
            ("::TestA::test_b[a-_param_1_{x='d'}-E-c]", 'PASSED'),
            ("::TestA::test_b[a-_param_1_{x='d'}-E-C]", 'FAILED'),
            ("::TestA::test_b[a-_param_1_{x='d'}-e-c]", 'PASSED'),
            ("::TestA::test_b[a-_param_1_{x='d'}-e-C]", 'PASSED'),
        ]
    ),
])
# Pytest 7.0.1 creates test with names test_b[A-c-_param_0_{x='D'}-e]
#  instead of test_b[A-_param_0_{x='D'}-e-c]
@pytest.mark.skipif(
    pytest.__version__ > "7.4.2", reason="test name not compatible"
)
def test_parameterize_pytest_impl(testdir, src, outcomes):
    testdir.makepyfile('from cupy import testing\n' + src)
    result = testdir.runpytest('-v', '--tb=no')
    expected_lines = [
        '.*{} {}.*'.format(re.escape(name), res)
        for name, res in outcomes
    ]
    print("Result", pytest.__version__)
    print("---")
    print("Expected", '\n'.join(expected_lines))
    result.stdout.re_match_lines(expected_lines)
    expected_count = collections.Counter(
        [res.lower() for _, res in outcomes]
    )
    result.assert_outcomes(**expected_count)
