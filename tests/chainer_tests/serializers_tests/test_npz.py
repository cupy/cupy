import os
import tempfile
import unittest

import mock
import numpy

from chainer import cuda
from chainer import link
from chainer import links
from chainer import optimizers
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
        self.assertEqual(child.path, 'x/')

    def test_get_item_strip_slashes(self):
        child = self.serializer['/x/']
        self.assertEqual(child.path, 'x/')

    def check_serialize(self, data, query):
        ret = self.serializer(query, data)
        dset = self.serializer.target['w']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, data.shape)
        self.assertEqual(dset.size, data.size)
        self.assertEqual(dset.dtype, data.dtype)
        numpy.testing.assert_array_equal(dset, cuda.to_cpu(data))

        self.assertIs(ret, data)

    def test_serialize_cpu(self):
        self.check_serialize(self.data, 'w')

    @attr.gpu
    def test_serialize_gpu(self):
        self.check_serialize(cuda.to_gpu(self.data), 'w')

    def test_serialize_cpu_strip_slashes(self):
        self.check_serialize(self.data, '/w')

    @attr.gpu
    def test_serialize_gpu_strip_slashes(self):
        self.check_serialize(cuda.to_gpu(self.data), '/w')

    def test_serialize_scalar(self):
        ret = self.serializer('x', 10)
        dset = self.serializer.target['x']

        self.assertIsInstance(dset, numpy.ndarray)
        self.assertEqual(dset.shape, ())
        self.assertEqual(dset.size, 1)
        self.assertEqual(dset.dtype, int)
        self.assertEqual(dset[()], 10)

        self.assertIs(ret, 10)


@testing.parameterize(*testing.product({'compress': [False, True]}))
class TestNpzDeserializer(unittest.TestCase):

    def setUp(self):
        self.data = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            savez = numpy.savez_compressed if self.compress else numpy.savez
            savez(
                f, **{'x/': None, 'y': self.data, 'z': numpy.asarray(10)})

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NpzDeserializer(self.npzfile)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_get_item(self):
        child = self.deserializer['x']
        self.assertIsInstance(child, npz.NpzDeserializer)
        self.assertEqual(child.path[-2:], 'x/')

    def test_get_item_strip_slashes(self):
        child = self.deserializer['/x/']
        self.assertEqual(child.path, 'x/')

    def check_deserialize(self, y, query):
        ret = self.deserializer(query, y)
        numpy.testing.assert_array_equal(cuda.to_cpu(y), self.data)
        self.assertIs(ret, y)

    def check_deserialize_none_value(self, y, query):
        ret = self.deserializer(query, None)
        numpy.testing.assert_array_equal(cuda.to_cpu(ret), self.data)

    def test_deserialize_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(y, 'y')

    def test_deserialize_none_value_cpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize_none_value(y, 'y')

    @attr.gpu
    def test_deserialize_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(cuda.to_gpu(y), 'y')

    @attr.gpu
    def test_deserialize_none_value_gpu(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize_none_value(cuda.to_gpu(y), 'y')

    def test_deserialize_cpu_strip_slashes(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(y, '/y')

    @attr.gpu
    def test_deserialize_gpu_strip_slashes(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        self.check_deserialize(cuda.to_gpu(y), '/y')

    def test_deserialize_scalar(self):
        z = 5
        ret = self.deserializer('z', z)
        self.assertEqual(ret, 10)


class TestNpzDeserializerNonStrict(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            numpy.savez(
                f, **{'x': numpy.asarray(10)})

        self.npzfile = numpy.load(path)
        self.deserializer = npz.NpzDeserializer(self.npzfile, strict=False)

    def tearDown(self):
        if hasattr(self, 'npzfile'):
            self.npzfile.close()
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_deserialize_partial(self):
        y = numpy.empty((2, 3), dtype=numpy.float32)
        ret = self.deserializer('y', y)
        self.assertIs(ret, y)


@testing.parameterize(*testing.product({'compress': [False, True]}))
class TestSaveNpz(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_save(self):
        obj = mock.MagicMock()
        npz.save_npz(self.temp_file_path, obj, self.compress)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, npz.DictionarySerializer)


@testing.parameterize(*testing.product({'compress': [False, True]}))
class TestLoadNpz(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path
        with open(path, 'wb') as f:
            savez = numpy.savez_compressed if self.compress else numpy.savez
            savez(f, None)

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def test_load(self):
        obj = mock.MagicMock()
        npz.load_npz(self.temp_file_path, obj)

        self.assertEqual(obj.serialize.call_count, 1)
        (serializer,), _ = obj.serialize.call_args
        self.assertIsInstance(serializer, npz.NpzDeserializer)


@testing.parameterize(*testing.product({'compress': [False, True]}))
class TestGroupHierachy(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = path

        child = link.Chain(linear=links.Linear(2, 3))
        child.add_param('Wc', (2, 3))
        self.parent = link.Chain(child=child)
        self.parent.add_param('Wp', (2, 3))

        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self.parent)

        self.savez = numpy.savez_compressed if self.compress else numpy.savez

    def _save(self, target, obj, name):
        serializer = npz.DictionarySerializer(target, name)
        serializer.save(obj)

    def tearDown(self):
        if hasattr(self, 'temp_file_path'):
            os.remove(self.temp_file_path)

    def _check_chain_group(self, file, state, prefix=''):
        keys = ('child/linear/W',
                'child/linear/b',
                'child/Wc') + state
        self.assertSetEqual(set(file.keys()), {prefix + x for x in keys})

    def _check_optimizer_group(self, file, state, prefix=''):
        keys = ('child/linear/W/msg',
                'child/linear/W/msdx',
                'child/linear/b/msg',
                'child/linear/b/msdx',
                'child/Wc/msg',
                'child/Wc/msdx') + state
        self.assertSetEqual(set(file.keys()),
                            {prefix + x for x in keys})

    def test_save_chain(self):
        d = {}
        self._save(d, self.parent, 'test/')
        with open(self.temp_file_path, 'wb') as f:
            self.savez(f, **d)
        with numpy.load(self.temp_file_path) as f:
            self._check_chain_group(f, ('Wp',), 'test/')

    def test_save_optimizer(self):
        d = {}
        self._save(d, self.optimizer, 'test/')
        with open(self.temp_file_path, 'wb') as f:
            self.savez(f, **d)
        with numpy.load(self.temp_file_path) as f:
            self._check_optimizer_group(
                f, ('Wp/msg', 'Wp/msdx', 'epoch', 't'), 'test/')

    def test_save_chain2(self):
        npz.save_npz(self.temp_file_path, self.parent, self.compress)
        with numpy.load(self.temp_file_path) as f:
            self._check_chain_group(f, ('Wp',))

    def test_save_optimizer2(self):
        npz.save_npz(self.temp_file_path, self.optimizer, self.compress)
        with numpy.load(self.temp_file_path) as f:
            self._check_optimizer_group(f, ('Wp/msg', 'Wp/msdx', 'epoch', 't'))

    def test_load_optimizer(self):
        for param in self.parent.params():
            param.data.fill(1)
        npz.save_npz(self.temp_file_path, self.parent, self.compress)
        for param in self.parent.params():
            param.data.fill(0)
        npz.load_npz(self.temp_file_path, self.parent)
        for param in self.parent.params():
            self.assertTrue((param.data == 1).all())


testing.run_module(__name__, __file__)
