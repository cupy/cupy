import unittest

from chainer import iterators
from chainer import testing


class TestSerialIterator(unittest.TestCase):

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2, shuffle=False)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertEqual(it.next(), [1, 2])
            self.assertFalse(it.is_new_epoch)
            self.assertEqual(it.next(), [3, 4])
            self.assertFalse(it.is_new_epoch)
            self.assertEqual(it.next(), [5, 6])
            self.assertTrue(it.is_new_epoch)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2, shuffle=False)

        self.assertEqual(it.epoch, 0)
        self.assertEqual(it.next(), [1, 2])
        self.assertFalse(it.is_new_epoch)
        self.assertEqual(it.next(), [3, 4])
        self.assertFalse(it.is_new_epoch)
        self.assertEqual(it.next(), [5, 1])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)

        self.assertEqual(it.next(), [2, 3])
        self.assertFalse(it.is_new_epoch)
        self.assertEqual(it.next(), [4, 5])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 2)

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2, repeat=False, shuffle=False)

        self.assertEqual(it.next(), [1, 2])
        self.assertEqual(it.next(), [3, 4])
        self.assertEqual(it.next(), [5, 6])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        for i in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SerialIterator(dataset, 2, repeat=False, shuffle=False)

        self.assertEqual(it.next(), [1, 2])
        self.assertEqual(it.next(), [3, 4])
        self.assertEqual(it.next(), [5])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        self.assertRaises(StopIteration, it.next)


class TestSerialIteratorShuffled(unittest.TestCase):

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SerialIterator(dataset, 2)
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

        batch1 = it.next()
        batch2 = it.next()
        batch3 = it.next()
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


testing.run_module(__name__, __file__)
