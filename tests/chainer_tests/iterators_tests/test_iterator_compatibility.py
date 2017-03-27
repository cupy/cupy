from __future__ import division
import unittest

import itertools
import numpy

from chainer import iterators
from chainer import serializer
from chainer import testing


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, self.target[key])
        else:
            value = type(value)(numpy.asarray(self.target[key]))
        return value


@testing.parameterize(*testing.product({
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
}))
class TestIteratorCompatibility(unittest.TestCase):

    def setUp(self):
        self.n_processes = 2
        self.options = {'n_processes': self.n_processes,
                        'n_prefetch': self.n_prefetch,
                        'shared_mem': self.shared_mem}

    def test_iterator_compatibilty(self):
        dataset = [1, 2, 3, 4, 5, 6]

        iters = (
            lambda: iterators.SerialIterator(dataset, 2),
            lambda: iterators.MultiprocessIterator(dataset, 2, **self.options),
        )

        for it_before, it_after in itertools.permutations(iters, 2):
            it = it_before()

            self.assertEqual(it.epoch, 0)
            self.assertAlmostEqual(it.epoch_detail, 0 / 6)
            batch1 = it.next()
            self.assertEqual(len(batch1), 2)
            self.assertIsInstance(batch1, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, 2 / 6)
            batch2 = it.next()
            self.assertEqual(len(batch2), 2)
            self.assertIsInstance(batch2, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, 4 / 6)

            target = dict()
            it.serialize(DummySerializer(target))

            it = it_after()
            it.serialize(DummyDeserializer(target))
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, 4 / 6)

            batch3 = it.next()
            self.assertEqual(len(batch3), 2)
            self.assertIsInstance(batch3, list)
            self.assertTrue(it.is_new_epoch)
            self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
            self.assertAlmostEqual(it.epoch_detail, 6 / 6)


testing.run_module(__name__, __file__)
