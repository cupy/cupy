import os
import unittest

from chainer import dataset
from chainer import testing


class TestGetSetDatasetRoot(unittest.TestCase):

    def test_set_dataset_root(self):
        orig_root = dataset.get_dataset_root()
        new_root = '/tmp/dataset_root'
        try:
            dataset.set_dataset_root(new_root)
            self.assertEqual(dataset.get_dataset_root(), new_root)
        finally:
            dataset.set_dataset_root(orig_root)


class TestGetDatasetDirectory(unittest.TestCase):

    def test_get_dataset_directory(self):
        root = dataset.get_dataset_root()
        path = dataset.get_dataset_directory('test', False)
        self.assertEqual(path, os.path.join(root, 'test'))


testing.run_module(__name__, __file__)
