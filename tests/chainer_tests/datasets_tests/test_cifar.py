import os
import unittest

import mock
import numpy

from chainer.dataset import download
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.datasets import tuple_dataset
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'withlabel': [True, False],
    'ndim': [1, 3],
    'scale': [1., 255.]
}))
class TestCifar(unittest.TestCase):

    def setUp(self):
        self.root = download.get_dataset_directory(
            os.path.join('pfnet', 'chainer', 'cifar'))

    def tearDown(self):
        if hasattr(self, 'cached_file') and os.path.exists(self.cached_file):
            os.remove(self.cached_file)

    @attr.slow
    def test_get_cifar10(self):
        self.check_retrieval_once('cifar-10.npz', get_cifar10)

    @attr.slow
    def test_get_cifar100(self):
        self.check_retrieval_once('cifar-100.npz', get_cifar100)

    def check_retrieval_once(self, name, retrieval_func):
        self.cached_file = os.path.join(self.root, name)
        train, test = retrieval_func(withlabel=self.withlabel, ndim=self.ndim,
                                     scale=self.scale)

        for cifar_dataset in (train, test):
            if self.withlabel:
                self.assertIsInstance(cifar_dataset,
                                      tuple_dataset.TupleDataset)
                cifar_dataset = cifar_dataset._datasets[0]
            else:
                self.assertIsInstance(cifar_dataset, numpy.ndarray)

            if self.ndim == 1:
                self.assertEqual(cifar_dataset.ndim, 2)
            else:
                # self.ndim == 3
                self.assertEqual(cifar_dataset.ndim, 4)
                self.assertEqual(cifar_dataset.shape[2],
                                 cifar_dataset.shape[3])  # 32

    # test caching - call twice
    @attr.slow
    def test_get_cifar10_cached(self):
        self.check_retrieval_twice('cifar-10.npz', get_cifar10)

    @attr.slow
    def test_get_cifar100_cached(self):
        self.check_retrieval_twice('cifar-100.npz', get_cifar100)

    def check_retrieval_twice(self, name, retrieval_func):
        self.cached_file = os.path.join(self.root, name)
        train, test = retrieval_func(withlabel=self.withlabel, ndim=self.ndim,
                                     scale=self.scale)

        with mock.patch('chainer.datasets.cifar.numpy', autospec=True) as \
                mnumpy:
            train, test = retrieval_func(withlabel=self.withlabel,
                                         ndim=self.ndim,
                                         scale=self.scale)
        mnumpy.savez_compressed.assert_not_called()  # creator() not called
        self.assertEqual(mnumpy.load.call_count, 1)


testing.run_module(__name__, __file__)
