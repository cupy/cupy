import unittest

from chainer import datasets


class TestSubDataset(unittest.TestCase):

    def test_sub_dataset(self):
        original = [1, 2, 3, 4, 5]
        subset = datasets.SubDataset(original, 1, 4)
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0], 2)
        self.assertEqual(subset[1], 3)
        self.assertEqual(subset[2], 4)

    def test_permuted_sub_dataset(self):
        original = [1, 2, 3, 4, 5]
        subset = datasets.SubDataset(original, 1, 4, [2, 0, 3, 1, 4])
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0], 1)
        self.assertEqual(subset[1], 4)
        self.assertEqual(subset[2], 2)

    def test_split_dataset(self):
        original = [1, 2, 3, 4, 5]
        subset1, subset2 = datasets.split_dataset(original, 2)
        self.assertEqual(len(subset1), 2)
        self.assertEqual(subset1[0], 1)
        self.assertEqual(subset1[1], 2)
        self.assertEqual(len(subset2), 3)
        self.assertEqual(subset2[0], 3)
        self.assertEqual(subset2[1], 4)
        self.assertEqual(subset2[2], 5)

    def test_permuted_split_dataset(self):
        original = [1, 2, 3, 4, 5]
        subset1, subset2 = datasets.split_dataset(original, 2, [2, 0, 3, 1, 4])
        self.assertEqual(len(subset1), 2)
        self.assertEqual(subset1[0], 3)
        self.assertEqual(subset1[1], 1)
        self.assertEqual(len(subset2), 3)
        self.assertEqual(subset2[0], 4)
        self.assertEqual(subset2[1], 2)
        self.assertEqual(subset2[2], 5)

    def test_get_cross_validation_datasets(self):
        original = [1, 2, 3, 4, 5, 6]
        cv1, cv2, cv3 = datasets.get_cross_validation_datasets(original, 3)

        tr1, te1 = cv1
        self.assertEqual(len(tr1), 4)
        self.assertEqual(tr1[0], 1)
        self.assertEqual(tr1[1], 2)
        self.assertEqual(tr1[2], 3)
        self.assertEqual(tr1[3], 4)
        self.assertEqual(len(te1), 2)
        self.assertEqual(te1[0], 5)
        self.assertEqual(te1[1], 6)

        tr2, te2 = cv2
        self.assertEqual(len(tr2), 4)
        self.assertEqual(tr2[0], 5)
        self.assertEqual(tr2[1], 6)
        self.assertEqual(tr2[2], 1)
        self.assertEqual(tr2[3], 2)
        self.assertEqual(len(te2), 2)
        self.assertEqual(te2[0], 3)
        self.assertEqual(te2[1], 4)

        tr3, te3 = cv3
        self.assertEqual(len(tr3), 4)
        self.assertEqual(tr3[0], 3)
        self.assertEqual(tr3[1], 4)
        self.assertEqual(tr3[2], 5)
        self.assertEqual(tr3[3], 6)
        self.assertEqual(len(te3), 2)
        self.assertEqual(te3[0], 1)
        self.assertEqual(te3[1], 2)

    def test_get_cross_validation_datasets_2(self):
        original = [1, 2, 3, 4, 5, 6, 7]
        cv1, cv2, cv3 = datasets.get_cross_validation_datasets(original, 3)

        tr1, te1 = cv1
        self.assertEqual(len(tr1), 4)
        self.assertEqual(tr1[0], 1)
        self.assertEqual(tr1[1], 2)
        self.assertEqual(tr1[2], 3)
        self.assertEqual(tr1[3], 4)
        self.assertEqual(len(te1), 3)
        self.assertEqual(te1[0], 5)
        self.assertEqual(te1[1], 6)
        self.assertEqual(te1[2], 7)

        tr2, te2 = cv2
        self.assertEqual(len(tr2), 5)
        self.assertEqual(tr2[0], 5)
        self.assertEqual(tr2[1], 6)
        self.assertEqual(tr2[2], 7)
        self.assertEqual(tr2[3], 1)
        self.assertEqual(tr2[4], 2)
        self.assertEqual(len(te2), 2)
        self.assertEqual(te2[0], 3)
        self.assertEqual(te2[1], 4)

        tr3, te3 = cv3
        self.assertEqual(len(tr3), 5)
        self.assertEqual(tr3[0], 3)
        self.assertEqual(tr3[1], 4)
        self.assertEqual(tr3[2], 5)
        self.assertEqual(tr3[3], 6)
        self.assertEqual(tr3[4], 7)
        self.assertEqual(len(te3), 2)
        self.assertEqual(te3[0], 1)
        self.assertEqual(te3[1], 2)


testing.run_module(__name__, __file__)
