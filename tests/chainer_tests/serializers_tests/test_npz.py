import os
import tempfile
import unittest

import mock
import numpy

from chainer import cuda
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr


class TestDictionarySerializer(unittest.TestCase):

    def setUp(self):
        self.serializer = npz.DictionarySerializer({})

        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def test_get_item(self):
        child = self.serializer['x']
        self.assertIsInstance(child, npz.DictionarySerializer)
        self.assertEqual(child.path[-2:], 'x/')

    def check_serialize(self, data):
        ret = self.serializer('w', data)
        dset = self.serializer.dict['w']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, data.shape)
        self.assertEqual(dset.size, data.size)
        self.assertEqual(dset.dtype, data.dtype)
        numpy.testing.assert_array_equal(dset, cuda.to_cpu(data))

        self.assertIs(ret, data)

    def test_serialize_cpu(self):
        self.check_serialize(self.data)

    @attr.gpu
    def test_serialize_gpu(self):
        self.check_serialize(cuda.to_gpu(self.data))

    def test_serialize_scalar(self):
        ret = self.serializer('x', 10)
        dset = self.serializer.dict['x']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, ())
        self.assertEqual(dset.size, 1)
        self.assertEqual(dset.dtype, int)
        self.assertEqual(dset[()], 10)

        self.assertIs(ret, 10)


class TestNPZDeserializer(unittest.TestCase):

    def setUp(self):
        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            numpy.savez(
                f, **{'x/': None, 'y': self.data, 'z': numpy.asarray(10)})

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NPZDeserializer(self.npzfile)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.deserializer['x']
        self.assertIsInstance(child, npz.NPZDeserializer)
        self.assertEqual(child.path[-2:], 'x/')

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


class TestSaveNPZ(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_save(self):
        obj = mock.MagicMock()
        npz.save_npz(self.temp_file_path, obj)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, npz.DictionarySerializer)


class TestLoadNPZ(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            numpy.savez(f, None)

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_load(self):
        obj = mock.MagicMock()
        npz.load_npz(self.temp_file_path, obj)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, npz.NPZDeserializer)


testing.run_module(__name__, __file__)
