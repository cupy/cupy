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
