import unittest

from chainer import iterators
from chainer import testing


class TestMultiprocessIterator(unittest.TestCase):

    def setUp(self):
        self.n_processes = 2

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.MultiprocessIterator(
            dataset, 2, n_processes=self.n_processes)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            batch1 = it.next()
            self.assertEqual(len(batch1), 2)
            self.assertFalse(it.is_new_epoch)
            batch2 = it.next()
            self.assertEqual(len(batch2), 2)
            self.assertFalse(it.is_new_epoch)
            batch3 = it.next()
            self.assertEqual(len(batch3), 2)
            self.assertTrue(it.is_new_epoch)
            self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, n_processes=self.n_processes)

        batches = sum([it.next() for _ in range(5)], [])
        self.assertEqual(sorted(batches), sorted(dataset * 2))

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, n_processes=self.n_processes)

        batches = sum([it.next() for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.MultiprocessIterator(
            dataset, 2, repeat=False, n_processes=self.n_processes)

        batch1 = it.next()
        batch2 = it.next()
        batch3 = it.next()
        self.assertRaises(StopIteration, it.next)

        self.assertEqual(len(batch3), 1)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)

    def test_iterator_shuffle_divisible(self):
        dataset = list(range(10))
        it = iterators.MultiprocessIterator(
            dataset, 10, n_processes=self.n_processes)
        self.assertNotEqual(it.next(), it.next())

    def test_iterator_shuffle_nondivisible(self):
        dataset = list(range(10))
        it = iterators.MultiprocessIterator(
            dataset, 3, n_processes=self.n_processes)
        out = sum([it.next() for _ in range(7)], [])
        self.assertNotEqual(out[0:10], out[10:20])


testing.run_module(__name__, __file__)
