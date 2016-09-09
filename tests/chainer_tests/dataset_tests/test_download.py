import os
import shutil
import tempfile
import unittest

import mock

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


class TestCacheOrLoadFile(unittest.TestCase):

    def setUp(self):
        self.default_dataset_root = dataset.get_dataset_root()
        self.temp_dir = tempfile.mkdtemp()
        dataset.set_dataset_root(self.temp_dir)

    def tearDown(self):
        dataset.set_dataset_root(self.default_dataset_root)
        shutil.rmtree(self.temp_dir)

    def test_cache_exists(self):
        creator = mock.Mock()
        loader = mock.Mock()

        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()

        try:
            dataset.cache_or_load_file(f.name, creator, loader)
        finally:
            os.remove(f.name)

        self.assertFalse(creator.called)
        loader.assert_called_once_with(f.name)

    def test_new_file(self):
        def create(path):
            with open(path, 'w') as f:
                f.write('test')

        creator = mock.Mock()
        creator.side_effect = create
        loader = mock.Mock()

        dir_path = tempfile.mkdtemp()
        # This file always does not exists as the directory is new.
        path = os.path.join(dir_path, 'cahche')

        try:
            dataset.cache_or_load_file(path, creator, loader)

            self.assertEqual(creator.call_count, 1)
            self.assertFalse(loader.called)

            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                self.assertEqual(f.read(), 'test')

        finally:
            shutil.rmtree(dir_path)


class TestCacheOrLoadFileFileExists(unittest.TestCase):

    def setUp(self):
        self.default_dataset_root = dataset.get_dataset_root()
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        dataset.set_dataset_root(self.temp_file.name)
        self.dir_path = tempfile.mkdtemp()

    def tearDown(self):
        dataset.set_dataset_root(self.default_dataset_root)
        os.remove(self.temp_file.name)
        shutil.rmtree(self.dir_path)

    def test_file_exists(self):
        creator = mock.Mock()
        loader = mock.Mock()

        # This file always does not exists as the directory is new.
        path = os.path.join(self.dir_path, 'cahche')

        with self.assertRaises(RuntimeError):
            dataset.cache_or_load_file(path, creator, loader)


class TestCachedDownload(unittest.TestCase):

    def setUp(self):
        self.default_dataset_root = dataset.get_dataset_root()
        self.temp_dir = tempfile.mkdtemp()
        dataset.set_dataset_root(self.temp_dir)

    def tearDown(self):
        dataset.set_dataset_root(self.default_dataset_root)
        shutil.rmtree(self.temp_dir)

    def test_fail_to_make_dir(self):
        with mock.patch('os.makedirs') as f:
            f.side_effect = OSError()
            with self.assertRaises(RuntimeError):
                dataset.cached_download('http://example.com')

    def test_file_exists(self):
        # Make an empty file which has the same name as the cache directory
        with open(os.path.join(self.temp_dir, '_dl_cache'), 'w'):
            pass
        with self.assertRaises(RuntimeError):
            dataset.cached_download('http://example.com')

    def test_cached_download(self):
        with mock.patch('six.moves.urllib.request.urlretrieve') as f:
            def download(url, path):
                with open(path, 'w') as f:
                    f.write('test')
            f.side_effect = download

            cache_path = dataset.cached_download('http://example.com')

        self.assertEqual(f.call_count, 1)
        args, kwargs = f.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 2)
        # The second argument is a temporary path, and it is removed
        self.assertEqual(args[0], 'http://example.com')

        self.assertTrue(os.path.exists(cache_path))
        with open(cache_path) as f:
            stored_data = f.read()
        self.assertEqual(stored_data, 'test')


testing.run_module(__name__, __file__)
