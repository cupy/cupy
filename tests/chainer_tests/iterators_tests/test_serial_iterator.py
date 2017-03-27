from __future__ import division
import unittest

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


class TestSerialIterator(unittest.TestCase):

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2, shuffle=False)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i + 0 / 6)
            if i == 0:
                self.assertIsNone(it.previous_epoch_detail)
            else:
                self.assertAlmostEqual(it.previous_epoch_detail, i - 2 / 6)
            self.assertEqual(it.next(), [1, 2])
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 2 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 0 / 6)
            self.assertEqual(it.next(), [3, 4])
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 4 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 2 / 6)
            self.assertEqual(it.next(), [5, 6])
            self.assertTrue(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 6 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 4 / 6)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2, shuffle=False)

        self.assertEqual(it.epoch, 0)
        self.assertAlmostEqual(it.epoch_detail, 0 / 5)
        self.assertIsNone(it.previous_epoch_detail)
        self.assertEqual(it.next(), [1, 2])
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 5)
        self.assertEqual(it.next(), [3, 4])
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 5)
        self.assertEqual(it.next(), [5, 1])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        self.assertAlmostEqual(it.epoch_detail, 6 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 5)

        self.assertEqual(it.next(), [2, 3])
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 8 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 6 / 5)
        self.assertEqual(it.next(), [4, 5])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 2)
        self.assertAlmostEqual(it.epoch_detail, 10 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 8 / 5)

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2, repeat=False, shuffle=False)

        self.assertAlmostEqual(it.epoch_detail, 0 / 6)
        self.assertIsNone(it.previous_epoch_detail)
        self.assertEqual(it.next(), [1, 2])
        self.assertAlmostEqual(it.epoch_detail, 2 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 6)
        self.assertEqual(it.next(), [3, 4])
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)
        self.assertEqual(it.next(), [5, 6])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        self.assertAlmostEqual(it.epoch_detail, 6 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 6)
        for i in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2, repeat=False, shuffle=False)

        self.assertAlmostEqual(it.epoch_detail, 0 / 5)
        self.assertIsNone(it.previous_epoch_detail)
        self.assertEqual(it.next(), [1, 2])
        self.assertAlmostEqual(it.epoch_detail, 2 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 5)
        self.assertEqual(it.next(), [3, 4])
        self.assertAlmostEqual(it.epoch_detail, 4 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 5)
        self.assertEqual(it.next(), [5])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        self.assertAlmostEqual(it.epoch_detail, 5 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 5)
        self.assertRaises(StopIteration, it.next)


class TestSerialIteratorShuffled(unittest.TestCase):

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i + 0 / 6)
            if i == 0:
                self.assertIsNone(it.previous_epoch_detail)
            else:
                self.assertAlmostEqual(it.previous_epoch_detail, i - 2 / 6)
            batch1 = it.next()
            self.assertEqual(len(batch1), 2)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 2 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 0 / 6)
            batch2 = it.next()
            self.assertEqual(len(batch2), 2)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 4 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 2 / 6)
            batch3 = it.next()
            self.assertEqual(len(batch3), 2)
            self.assertTrue(it.is_new_epoch)
            self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
            self.assertAlmostEqual(it.epoch_detail, i + 6 / 6)
            self.assertAlmostEqual(it.previous_epoch_detail, i + 4 / 6)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2)

        batches = sum([it.next() for _ in range(5)], [])
        self.assertEqual(sorted(batches[:5]), dataset)
        self.assertEqual(sorted(batches[5:]), dataset)

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2, repeat=False)

        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2, repeat=False)

        self.assertAlmostEqual(it.epoch_detail, 0 / 5)
        self.assertIsNone(it.previous_epoch_detail)
        batch1 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 2 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 5)
        batch2 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 4 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 5)
        batch3 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 5 / 5)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 5)
        self.assertRaises(StopIteration, it.next)

        self.assertEqual(len(batch3), 1)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)

    def test_iterator_shuffle_divisible(self):
        dataset = list(range(10))
        it = iterators.SerialIterator(dataset, 10)
        self.assertNotEqual(it.next(), it.next())

    def test_iterator_shuffle_nondivisible(self):
        dataset = list(range(10))
        it = iterators.SerialIterator(dataset, 3)
        out = sum([it.next() for _ in range(7)], [])
        self.assertNotEqual(out[0:10], out[10:20])

    def test_reset(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2, repeat=False)

        for trial in range(4):
            batches = sum([it.next() for _ in range(3)], [])
            self.assertEqual(sorted(batches), dataset)
            for _ in range(2):
                self.assertRaises(StopIteration, it.next)
            it.reset()


class TestSerialIteratorSerialize(unittest.TestCase):

    def test_iterator_serialize(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2)

        self.assertEqual(it.epoch, 0)
        self.assertAlmostEqual(it.epoch_detail, 0 / 6)
        self.assertIsNone(it.previous_epoch_detail)
        batch1 = it.next()
        self.assertEqual(len(batch1), 2)
        self.assertIsInstance(batch1, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 6)
        batch2 = it.next()
        self.assertEqual(len(batch2), 2)
        self.assertIsInstance(batch2, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)

        target = dict()
        it.serialize(DummySerializer(target))

        it = iterators.SerialIterator(dataset, 2)
        it.serialize(DummyDeserializer(target))
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)

        batch3 = it.next()
        self.assertEqual(len(batch3), 2)
        self.assertIsInstance(batch3, list)
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
        self.assertAlmostEqual(it.epoch_detail, 6 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 6)

    def test_iterator_serialize_backward_compat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2)

        self.assertEqual(it.epoch, 0)
        self.assertAlmostEqual(it.epoch_detail, 0 / 6)
        self.assertIsNone(it.previous_epoch_detail)
        batch1 = it.next()
        self.assertEqual(len(batch1), 2)
        self.assertIsInstance(batch1, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 2 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 0 / 6)
        batch2 = it.next()
        self.assertEqual(len(batch2), 2)
        self.assertIsInstance(batch2, list)
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)

        target = dict()
        it.serialize(DummySerializer(target))
        # older version does not have previous_epoch_detail
        del target['previous_epoch_detail']

        it = iterators.SerialIterator(dataset, 2)
        it.serialize(DummyDeserializer(target))
        self.assertFalse(it.is_new_epoch)
        self.assertAlmostEqual(it.epoch_detail, 4 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 2 / 6)

        batch3 = it.next()
        self.assertEqual(len(batch3), 2)
        self.assertIsInstance(batch3, list)
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
        self.assertAlmostEqual(it.epoch_detail, 6 / 6)
        self.assertAlmostEqual(it.previous_epoch_detail, 4 / 6)

    def test_iterator_serialize_backward_compat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2)

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
        # older version uses '_order'
        target['_order'] = target['order']
        del target['order']

        it = iterators.SerialIterator(dataset, 2)
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
