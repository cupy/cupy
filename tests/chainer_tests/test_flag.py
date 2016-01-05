import unittest

import chainer
from chainer import flag
from chainer import testing


class TestFlag(unittest.TestCase):

    def test_construction(self):
        self.assertIs(chainer.ON, chainer.Flag('on'))
        self.assertIs(chainer.ON, chainer.Flag('ON'))
        self.assertIs(chainer.ON, chainer.Flag(True))
        self.assertIs(chainer.OFF, chainer.Flag('off'))
        self.assertIs(chainer.OFF, chainer.Flag('OFF'))
        self.assertIs(chainer.OFF, chainer.Flag(False))
        self.assertIs(chainer.AUTO, chainer.Flag('auto'))
        self.assertIs(chainer.AUTO, chainer.Flag('AUTO'))

        self.assertIsNot(chainer.ON, chainer.OFF)
        self.assertIsNot(chainer.ON, chainer.AUTO)
        self.assertIsNot(chainer.OFF, chainer.ON)

    def test_aggregate_flags(self):
        self.assertIs(
            flag.aggregate_flags([chainer.ON, chainer.AUTO, chainer.ON]),
            chainer.ON)
        self.assertIs(
            flag.aggregate_flags([
                chainer.OFF, chainer.OFF, chainer.AUTO, chainer.AUTO]),
            chainer.OFF)

    def test_mix_on_and_off(self):
        with self.assertRaises(ValueError):
            flag.aggregate_flags([chainer.ON, chainer.AUTO, chainer.OFF])

    def test_repr(self):
        self.assertEqual(repr(chainer.ON), 'ON')
        self.assertEqual(repr(chainer.OFF), 'OFF')
        self.assertEqual(repr(chainer.AUTO), 'AUTO')

    def test_equality(self):
        self.assertEqual(chainer.ON, 'on')
        self.assertEqual(chainer.ON, 'ON')
        self.assertEqual(chainer.ON, True)
        self.assertEqual(chainer.OFF, 'off')
        self.assertEqual(chainer.OFF, 'OFF')
        self.assertEqual(chainer.OFF, False)
        self.assertEqual(chainer.AUTO, 'auto')
        self.assertEqual(chainer.AUTO, 'AUTO')
        self.assertEqual(chainer.AUTO, None)


testing.run_module(__name__, __file__)
