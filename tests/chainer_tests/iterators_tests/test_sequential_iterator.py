import unittest

from chainer import iterators
from chainer import testing


class TestSequentialIterator(unittest.TestCase):

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SequentialIterator(dataset, 2)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            self.assertEqual(it.next()[:], [1, 2])
            self.assertFalse(it.is_new_epoch)
            self.assertEqual(it.next()[:], [3, 4])
            self.assertFalse(it.is_new_epoch)
            self.assertEqual(it.next()[:], [5, 6])
            self.assertTrue(it.is_new_epoch)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SequentialIterator(dataset, 2)

        self.assertEqual(it.epoch, 0)
        self.assertEqual(it.next()[:], [1, 2])
        self.assertFalse(it.is_new_epoch)
        self.assertEqual(it.next()[:], [3, 4])
        self.assertFalse(it.is_new_epoch)
        self.assertEqual(it.next()[:], [5, 1])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)

        self.assertEqual(it.next()[:], [2, 3])
        self.assertFalse(it.is_new_epoch)
        self.assertEqual(it.next()[:], [4, 5])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 2)

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.SequentialIterator(dataset, 2, repeat=False)

        self.assertEqual(it.next()[:], [1, 2])
        self.assertEqual(it.next()[:], [3, 4])
        self.assertEqual(it.next()[:], [5, 6])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        for i in range(2):
            self.assertRaises(StopIteration, it.next)

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.SequentialIterator(dataset, 2, repeat=False)

        self.assertEqual(it.next()[:], [1, 2])
        self.assertEqual(it.next()[:], [3, 4])
        self.assertEqual(it.next()[:], [5])
        self.assertTrue(it.is_new_epoch)
        self.assertEqual(it.epoch, 1)
        self.assertRaises(StopIteration, it.next)


testing.run_module(__name__, __file__)
