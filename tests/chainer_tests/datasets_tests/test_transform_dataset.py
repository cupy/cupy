import unittest

import numpy

from chainer import datasets
from chainer import testing


def _create_list_tuples(shape1, shape2, length):
    return [(numpy.random.uniform(shape1), numpy.random.uniform(shape2)) for
            _ in range(length)]


@testing.parameterize(
    {'dataset': numpy.random.uniform(size=(2, 3, 32, 32))},
    {'dataset': _create_list_tuples((3, 32, 32), (32, 32), 5)}
)
class TestTransformDataset(unittest.TestCase):

    def setUp(self):
        def transform(in_data):
            if isinstance(in_data, tuple):
                return tuple([example * 3 for example in in_data])
            else:
                return in_data * 3
        self.transform = transform

    def test_transform_dataset(self):
        td = datasets.TransformDataset(self.dataset, self.transform)
        self.assertEqual(len(td), len(self.dataset))

        for i in range(len(td)):
            example = td[i]
            if isinstance(example, tuple):
                for j, arr in enumerate(example):
                    numpy.testing.assert_array_equal(
                        arr, self.transform(self.dataset[i][j]))
            else:
                numpy.testing.assert_array_equal(
                    example, self.transform(self.dataset[i]))

    def test_transform_dataset_overrun(self):
        td = datasets.TransformDataset(self.dataset, self.transform)
        with self.assertRaises(IndexError):
            td[len(td) + 1]


testing.run_module(__name__, __file__)
