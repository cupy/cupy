import unittest

from cupy import testing


@testing.parameterize(
    {'actual': {'a': [1, 2], 'b': [3, 4, 5]},
     'expect': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 5},
                {'a': 2, 'b': 3}, {'a': 2, 'b': 4}, {'a': 2, 'b': 5}]},
    {'actual': {'a': [1, 2]}, 'expect': [{'a': 1}, {'a': 2}]},
    {'actual': {'a': [1, 2], 'b': []}, 'expect': []},
    {'actual': {'a': []}, 'expect': []},
    {'actual': {}, 'expect': [{}]})
class ProductTest(unittest.TestCase):

    def test_product(self):
        self.assertListEqual(testing.product(self.actual), self.expect)


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
class ProductDictTest(unittest.TestCase):

    def test_product_dict(self):
        self.assertListEqual(testing.product_dict(*self.actual), self.expect)


def f(x):
    return x


class C(object):

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
        self.assertEqual(y, 1)

    def test_skip(self):
        # Skipping the test case should not report error.
        self.skipTest('skip')


@testing.parameterize(
    {'param1': 1},
    {'param1': 2})
@testing.parameterize(
    {'param2': 3},
    {'param2': 4})
class TestParameterizeTwice(unittest.TestCase):
    # This test only checks if each of the parameterized combinations is a
    # member of the expected combinations. This test does not check if each
    # of the expected combinations is actually visited by the parameterization,
    # as there are no way to test that in a robust manner.

    def test_twice(self):
        assert hasattr(self, 'param1')
        assert hasattr(self, 'param2')
        assert (self.param1, self.param2) in (
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4))
