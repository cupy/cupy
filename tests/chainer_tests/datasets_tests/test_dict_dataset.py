import unittest

import numpy

from chainer import cuda
from chainer import datasets
from chainer import testing
from chainer.testing import attr


class TestDictDataset(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.rand(3, 4)
        self.y = numpy.random.rand(3, 5)
        self.z = numpy.random.rand(4, 4)

    def check_dict_dataset(self, x, y):
        dd = datasets.DictDataset(x=x, y=y)
        self.assertEqual(len(dd), len(x))

        for i in range(len(x)):
            example = dd[i]
            self.assertIn('x', example)
            self.assertIn('y', example)

            numpy.testing.assert_array_equal(
                cuda.to_cpu(example['x']), cuda.to_cpu(x[i]))
            numpy.testing.assert_array_equal(
                cuda.to_cpu(example['y']), cuda.to_cpu(y[i]))

    def test_dict_dataset_cpu(self):
        self.check_dict_dataset(self.x, self.y)

    @attr.gpu
    def test_dict_dataset_gpu(self):
        self.check_dict_dataset(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

    def test_dict_dataset_len_mismatch(self):
        with self.assertRaises(ValueError):
            datasets.DictDataset(x=self.x, z=self.z)

    def test_dict_dtaset_overrun(self):
        dd = datasets.DictDataset(x=self.x, y=self.y)
        with self.assertRaises(IndexError):
            dd[3]


testing.run_module(__name__, __file__)
