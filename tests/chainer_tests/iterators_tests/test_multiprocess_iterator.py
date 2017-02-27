from __future__ import division
import copy
import unittest

import numpy
import six

from chainer import iterators
from chainer import testing


@testing.parameterize(*testing.product({
    'n_prefetch': [1, 2],
    'shared_mem': [None, 1000000],
}))
class TestMultiprocessIterator(unittest.TestCase):

    def setUp(self):
        self.n_processes = 2
        self.options = {'n_processes': self.n_processes,
                        'n_prefetch': self.n_prefetch,
                        'shared_mem': self.shared_mem}

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i + 0 / 6)
            batch1 = it.next()
            self.assertEqual(len(batch1), 2)
            self.assertIsInstance(batch1, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 2 / 6)
            batch2 = it.next()
            self.assertEqual(len(batch2), 2)
            self.assertIsInstance(batch2, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, i + 4 / 6)
            batch3 = it.next()
            self.assertEqual(len(batch3), 2)
            self.assertIsInstance(batch3, list)
            self.assertTrue(it.is_new_epoch)
            self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)
            self.assertAlmostEqual(it.epoch_detail, i + 6 / 6)

    def test_iterator_list_type(self):
        dataset = [[i, numpy.zeros((10,)) + i] for i in range(6)]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i)
            batches = {}
            for j in range(3):
                batch = it.next()
                self.assertEqual(len(batch), 2)
                if j != 2:
                    self.assertFalse(it.is_new_epoch)
                else:
                    self.assertTrue(it.is_new_epoch)
                self.assertAlmostEqual(
                    it.epoch_detail, (3 * i + j + 1) * 2 / 6)
                for x in batch:
                    self.assertIsInstance(x, list)
                    self.assertIsInstance(x[1], numpy.ndarray)
                    batches[x[0]] = x[1]

            self.assertEqual(len(batches), len(dataset))
            for k, v in six.iteritems(batches):
                numpy.testing.assert_allclose(dataset[k][1], v)

    def test_iterator_tuple_type(self):
        dataset = [(i, numpy.zeros((10,)) + i) for i in range(6)]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i)
            batches = {}
            for j in range(3):
                batch = it.next()
                self.assertEqual(len(batch), 2)
                if j != 2:
                    self.assertFalse(it.is_new_epoch)
                else:
                    self.assertTrue(it.is_new_epoch)
                self.assertAlmostEqual(
                    it.epoch_detail, (3 * i + j + 1) * 2 / 6)
                for x in batch:
                    self.assertIsInstance(x, tuple)
                    self.assertIsInstance(x[1], numpy.ndarray)
                    batches[x[0]] = x[1]

            self.assertEqual(len(batches), len(dataset))
            for k, v in six.iteritems(batches):
                numpy.testing.assert_allclose(dataset[k][1], v)

    def test_iterator_dict_type(self):
        dataset = [{i: numpy.zeros((10,)) + i} for i in range(6)]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertAlmostEqual(it.epoch_detail, i)
            batches = {}
            for j in range(3):
                batch = it.next()
                self.assertEqual(len(batch), 2)
                if j != 2:
                    self.assertFalse(it.is_new_epoch)
                else:
                    self.assertTrue(it.is_new_epoch)
                self.assertAlmostEqual(
                    it.epoch_detail, (3 * i + j + 1) * 2 / 6)
                for x in batch:
                    self.assertIsInstance(x, dict)
                    k = tuple(x)[0]
                    v = x[k]
                    self.assertIsInstance(v, numpy.ndarray)
                    batches[k] = v

            self.assertEqual(len(batches), len(dataset))
            for k, v in six.iteritems(batches):
                x = dataset[k][tuple(dataset[k])[0]]
                numpy.testing.assert_allclose(x, v)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(dataset, 2, **self.options)

        batches = sum([it.next() for _ in range(5)], [])
        self.assertEqual(sorted(batches), sorted(dataset * 2))

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)

        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)

        self.assertAlmostEqual(it.epoch_detail, 0 / 5)
        batch1 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 2 / 5)
        batch2 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 4 / 5)
        batch3 = it.next()
        self.assertAlmostEqual(it.epoch_detail, 5 / 5)
        self.assertRaises(StopIteration, it.next)

        self.assertEqual(len(batch3), 1)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)

    def test_iterator_shuffle_divisible(self):
        dataset = list(range(10))
        it = iterators.MultiprocessIterator(
            dataset, 10, **self.options)
        self.assertNotEqual(it.next(), it.next())

    def test_iterator_shuffle_nondivisible(self):
        dataset = list(range(10))
        it = iterators.MultiprocessIterator(
            dataset, 3, **self.options)
        out = sum([it.next() for _ in range(7)], [])
        self.assertNotEqual(out[0:10], out[10:20])

    def test_copy_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, **self.options)
        copy_it = copy.copy(it)
        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)
        it = None

        batches = sum([copy_it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, copy_it.next)

testing.run_module(__name__, __file__)
