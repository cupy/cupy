import unittest

from chainer import iterators
from chainer import testing


class TestShuffledIterator(unittest.TestCase):

    def test_iterator_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.ShuffledIterator(dataset, 2)
        for i in range(3):
            self.assertEqual(it.epoch, i)
            batch1 = it.next()[:]
            self.assertEqual(len(batch1), 2)
            self.assertFalse(it.is_new_epoch)
            batch2 = it.next()[:]
            self.assertEqual(len(batch2), 2)
            self.assertFalse(it.is_new_epoch)
            batch3 = it.next()[:]
            self.assertEqual(len(batch3), 2)
            self.assertTrue(it.is_new_epoch)
            self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)

    def test_iterator_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.ShuffledIterator(dataset, 2)

        batches = sum([it.next()[:] for _ in range(5)], [])
        self.assertEqual(sorted(batches[:5]), dataset)
        self.assertEqual(sorted(batches[5:]), dataset)

    def test_iterator_not_repeat(self):
        dataset = [1, 2, 3, 4, 5, 6]
        it = iterators.ShuffledIterator(dataset, 2, repeat=False)

        batches = sum([it.next()[:] for _ in range(3)], [])
        self.assertEqual(sorted(batches), dataset)
        for _ in range(2):
            with self.assertRaises(StopIteration):
                it.next()

    def test_iterator_not_repeat_not_even(self):
        dataset = [1, 2, 3, 4, 5]
        it = iterators.ShuffledIterator(dataset, 2, repeat=False)

        batch1 = it.next()[:]
        batch2 = it.next()[:]
        batch3 = it.next()[:]
        with self.assertRaises(StopIteration):
            it.next()

        self.assertEqual(len(batch3), 1)
        self.assertEqual(sorted(batch1 + batch2 + batch3), dataset)


testing.run_module(__name__, __file__)
