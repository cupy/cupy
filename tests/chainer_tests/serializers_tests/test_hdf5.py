import os
import tempfile
import unittest

import h5py
import mock
import numpy

from chainer import cuda
from chainer.serializers import hdf5
from chainer import testing
from chainer.testing import attr


class TestHDF5Serializer(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        self.hdf5file = h5py.File(path, 'w')
        self.serializer = hdf5.HDF5Serializer(self.hdf5file, compression=3)

        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def tearDown(self):
        if hasattr(self, 'hdf5file'):
            self.hdf5file.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.serializer['x']
        self.assertIsInstance(child, hdf5.HDF5Serializer)
        self.assertEqual(child.group.name, '/x')
        self.assertEqual(child.compression, 3)

    def check_serialize(self, data):
        ret = self.serializer('w', data)
        dset = self.hdf5file['w']

        self.assertIsInstance(dset, h5py.Dataset)
        self.assertEqual(dset.shape, data.shape)
        self.assertEqual(dset.size, data.size)
        self.assertEqual(dset.dtype, data.dtype)
        read = numpy.empty((2, 3), dtype=numpy.float32)
        dset.read_direct(read)
        numpy.testing.assert_array_equal(read, cuda.to_cpu(data))

        self.assertEqual(dset.compression_opts, 3)

        self.assertIs(ret, data)

    def test_serialize_cpu(self):
        self.check_serialize(self.data)

    @attr.gpu
    def test_serialize_gpu(self):
        self.check_serialize(cuda.to_gpu(self.data))

    def test_serialize_scalar(self):
        ret = self.serializer('x', 10)
        dset = self.hdf5file['x']

        self.assertIsInstance(dset, h5py.Dataset)
        self.assertEqual(dset.shape, ())
        self.assertEqual(dset.size, 1)
        self.assertEqual(dset.dtype, int)
        read = numpy.empty((), dtype=numpy.int32)
        dset.read_direct(read)
        self.assertEqual(read, 10)

        self.assertEqual(dset.compression_opts, None)

        self.assertIs(ret, 10)


class TestHDF5Deserializer(unittest.TestCase):

    def setUp(self):
        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with h5py.File(path, 'w') as f:
            f.require_group('x')
            f.create_dataset('y', data=self.data)
            f.create_dataset('z', data=numpy.asarray(10))

        self.hdf5file = h5py.File(path, 'r')
        self.deserializer = hdf5.HDF5Deserializer(self.hdf5file)

    def tearDown(self):
        if hasattr(self, 'hdf5file'):
            self.hdf5file.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.deserializer['x']
        self.assertIsInstance(child, hdf5.HDF5Deserializer)
        self.assertEqual(child.group.name, '/x')

    def check_deserialize(self, y):
        ret = self.deserializer('y', y)
        numpy.testing.assert_array_equal(cuda.to_cpu(y), self.data)
        self.assertIs(ret, y)

    def test_deserialize_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(y)

    @attr.gpu
    def test_deserialize_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(cuda.to_gpu(y))

    def test_deserialize_scalar(self):
        z = 5
        ret = self.deserializer('z', z)
        self.assertEqual(ret, 10)


class TestSaveHDF5(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_save(self):
        obj = mock.MagicMock()
        hdf5.save_hdf5(self.temp_file_path, obj, compression=3)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, hdf5.HDF5Serializer)
        self.assertEqual(serializer.compression, 3)


class TestLoadHDF5(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        # Make a hdf5 file with empty data
        h5py.File(path, 'w')

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_load(self):
        obj = mock.MagicMock()
        hdf5.load_hdf5(self.temp_file_path, obj)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, hdf5.HDF5Deserializer)


testing.run_module(__name__, __file__)
